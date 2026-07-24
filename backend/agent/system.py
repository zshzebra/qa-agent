from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from openinference.semconv.trace import SpanAttributes
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Link
from pydantic_ai import (
    Agent,
    AgentCapability,
    ModelMessage,
    ModelMessagesTypeAdapter,
    RunContext,
    ToolOutput,
)
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_core import to_jsonable_python

from .models import (
    AuditResult,
    DialogueMessage,
    NextMessage,
    PlanStatus,
    ResearchJob,
    ResearchOutput,
    SystemOutput,
    render_dialogue,
)
from .tools.buffer import BufferToolset
from .tools.exa import ExaWithoutCitation


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required but is not set in the environment")
    return value


tracer = trace.get_tracer("lead-form")

exa_without_citation = ExaWithoutCitation(require_env("EXA_API_KEY"))
smart_model = GoogleModel("gemini-3.1-flash-lite-preview")
research_model = smart_model
auditor_model = smart_model


def load_instructions(agent: str) -> str:
    with open(
        os.path.join(os.path.dirname(__file__), f"instructions/{agent}.md"),
        "r",
        encoding="utf-8",
    ) as file:
        return file.read()


def instrumented_capabilities(extra: list[AgentCapability]) -> list[AgentCapability]:
    return [Instrumentation(InstrumentationSettings(version=2)), *extra]


def tool_output[OutputT](t: type[OutputT]) -> ToolOutput[OutputT]:
    return ToolOutput(t, name=t.__name__)


MAX_AUDIT_RETRIES = 2
AUDIT_FEEDBACK = load_instructions("audit_feedback")


class AgentSystem:
    def __init__(self, state: dict[str, Any] | None = None) -> None:
        state = state or {}

        # Serializable logical state.
        self.session_id: str = state.get("session_id", str(uuid4()))
        self.buffer: str = state.get("buffer", "")
        self.dialogue: list[DialogueMessage] = [
            DialogueMessage(**message) for message in state.get("dialogue", [])
        ]
        self.planner_history: list[ModelMessage] = (
            ModelMessagesTypeAdapter.validate_python(state.get("planner_history", []))
        )
        self.speaker_history: list[ModelMessage] = (
            ModelMessagesTypeAdapter.validate_python(state.get("speaker_history", []))
        )
        self.planner_sent: int = state.get("planner_sent", 0)
        self.speaker_sent: int = state.get("speaker_sent", 0)
        self.research_jobs = [
            ResearchJob(**job) for job in state.get("research_jobs", [])
        ]
        self.research_completion_order: int = state.get("research_completion_order", 0)
        self.planner_pending: bool = state.get("planner_pending", False)
        self.speaker_pending: bool = state.get("speaker_pending", False)
        self.waiting_for_user: bool = state.get("waiting_for_user", False)
        self.is_done: bool = state.get("is_done", False)
        self.done_note: str = state.get("done_note", "")
        self.failure: str | None = state.get("failure")

        # Runtime-only state.
        self._state_lock = asyncio.Lock()
        self._planner_cycle_lock = asyncio.Lock()
        self._speaker_cycle_lock = asyncio.Lock()
        self._planner_wakeup = asyncio.Event()
        self._speaker_wakeup = asyncio.Event()
        self._done = asyncio.Event()
        self._outputs: asyncio.Queue[SystemOutput] = asyncio.Queue()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._research_tasks: dict[str, asyncio.Task[None]] = {}
        self._started = False
        self._closing = False

    @classmethod
    def deserialize(cls, value: str) -> AgentSystem:
        return cls(json.loads(value))

    async def serialize(self) -> str:
        """Return a safe snapshot at a planner/speaker cycle boundary."""
        async with self._planner_cycle_lock, self._speaker_cycle_lock, self._state_lock:
            state = {
                "session_id": self.session_id,
                "buffer": self.buffer,
                "dialogue": [asdict(message) for message in self.dialogue],
                "planner_history": to_jsonable_python(self.planner_history),
                "speaker_history": to_jsonable_python(self.speaker_history),
                "planner_sent": self.planner_sent,
                "speaker_sent": self.speaker_sent,
                "research_jobs": [asdict(job) for job in self.research_jobs],
                "research_completion_order": self.research_completion_order,
                "planner_pending": self.planner_pending,
                "speaker_pending": self.speaker_pending,
                "waiting_for_user": self.waiting_for_user,
                "is_done": self.is_done,
                "done_note": self.done_note,
                "failure": self.failure,
            }
        return json.dumps(state)

    async def run(self) -> None:
        """Start or resume the system and return once the runtime is active."""
        if self._started:
            return
        self._started = True
        self._closing = False

        if self.failure is not None:
            self._done.set()
            await self._outputs.put(SystemOutput(kind="error", error=self.failure))
            return
        if self.is_done:
            self._done.set()
            await self._outputs.put(
                SystemOutput(
                    kind="done", message=self.done_note, final_plan=self.buffer
                )
            )
            return

        # loops start first, then the opener runs.
        self._spawn(self._planner_loop(), f"planner:{self.session_id}")
        self._spawn(self._speaker_loop(), f"speaker:{self.session_id}")

        if not self.dialogue:
            await self._open_session()

        await self._restore_research()
        async with self._state_lock:
            if any(
                job.status in {"completed", "failed"} and not job.consumed
                for job in self.research_jobs
            ):
                self.planner_pending = True
            planner_pending = self.planner_pending
            speaker_pending = self.speaker_pending

        if planner_pending:
            self._planner_wakeup.set()
        if speaker_pending:
            self._speaker_wakeup.set()

    async def send(self, message: str) -> None:
        message = message.strip()
        if not message:
            raise ValueError("Message cannot be empty")
        if not self._started:
            await self.run()

        async with self._state_lock:
            if self.is_done:
                raise RuntimeError("The agent system has completed")
            if self.failure is not None:
                raise RuntimeError(self.failure)
            if not self.waiting_for_user:
                raise RuntimeError("The agent system is not waiting for user input")
            self.dialogue.append(
                DialogueMessage(
                    role="user",
                    content=message,
                )
            )
            self.waiting_for_user = False
            self.planner_pending = True
        self._planner_wakeup.set()

    async def receive(self) -> SystemOutput:
        return await self._outputs.get()

    async def wait_done(self) -> None:
        await self._done.wait()

    async def close(self) -> None:
        """Gracefully stop at a planner/speaker cycle boundary."""
        if not self._started:
            return

        # Waiting here avoids persisting half a planner or speaker call.
        async with self._planner_cycle_lock, self._speaker_cycle_lock:
            self._closing = True

        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._tasks.clear()
        self._research_tasks.clear()
        self._started = False
        self._closing = False

    async def run_research(self, target: str) -> str:
        dispatch_ctx = trace.get_current_span().get_span_context()
        async with self._state_lock:
            job = ResearchJob(id=str(uuid4()), target=target)
            self.research_jobs.append(job)
        self._start_research(job.id, dispatch_ctx)
        return (
            f"Research dispatched for '{target}'; result will arrive in a later cycle."
        )

    def _spawn(self, coroutine: Any, name: str, job_id: str | None = None) -> None:
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        if job_id is not None:
            self._research_tasks[job_id] = task

        def done(finished: asyncio.Task[Any]) -> None:
            self._tasks.discard(finished)
            if job_id is not None:
                self._research_tasks.pop(job_id, None)
            if not finished.cancelled():
                finished.exception()

        task.add_done_callback(done)

    async def _open_session(self) -> None:
        async with self._speaker_cycle_lock:
            with tracer.start_as_current_span(
                "chat", attributes={SpanAttributes.SESSION_ID: self.session_id}
            ):
                opener = await user_agent.run(
                    "Greet the user warmly in one line and ask which company they "
                    "work for. Ask only that, and set speak=true.",
                    deps=self,
                    conversation_id=self.session_id,
                )

            async with self._state_lock:
                # The original opener was not added to speaker history.
                self.dialogue.append(
                    DialogueMessage(role="assistant", content=opener.output.message)
                )
                self.waiting_for_user = True

        await self._outputs.put(
            SystemOutput(
                kind="message", message=opener.output.message, expects_reply=True
            )
        )

    async def _planner_loop(self) -> None:
        try:
            while not self._stopped:
                await self._planner_wakeup.wait()
                self._planner_wakeup.clear()
                if self._stopped:
                    return
                async with self._planner_cycle_lock:
                    await self._planner_cycle()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail("planner", exc)

    async def _planner_cycle(self) -> None:
        async with self._state_lock:
            if not self.planner_pending:
                return
            self.planner_pending = False
            dialogue_end = len(self.dialogue)
            new_lines = list(self.dialogue[self.planner_sent : dialogue_end])
            findings = sorted(
                (
                    job
                    for job in self.research_jobs
                    if job.status in {"completed", "failed"} and not job.consumed
                ),
                key=lambda job: job.completion_order or 0,
            )
            history = list(self.planner_history)

        parts = []
        if new_lines:
            parts.append("New in the conversation:\n" + render_dialogue(new_lines))
        if findings:
            parts.append(
                "New research results:\n\n"
                + "\n\n".join(self._render_research(job) for job in findings)
            )
        if not parts:
            parts.append(
                "Nothing new arrived. Re-check the plan and decide the next step."
            )

        result = await self._call_planner("\n\n".join(parts), history)
        finding_ids = {job.id for job in findings}
        async with self._state_lock:
            self.planner_history = result.all_messages()
            self.planner_sent = dialogue_end
            for job in self.research_jobs:
                if job.id in finding_ids:
                    job.consumed = True
            self.speaker_pending = True
        self._speaker_wakeup.set()

        audit_retries = 0
        while result.output.done:
            audit = await self._run_audit()
            if audit.output.approved:
                await self._complete(result.output.note)
                return

            audit_retries += 1
            if audit_retries > MAX_AUDIT_RETRIES:
                await self._complete(
                    "Planner finishing with unresolved audit issues: "
                    f"{audit.output.issues}"
                )
                return

            result = await self._call_planner(
                AUDIT_FEEDBACK.format(issues=audit.output.issues),
                list(self.planner_history),
            )
            async with self._state_lock:
                self.planner_history = result.all_messages()
                self.speaker_pending = True
            self._speaker_wakeup.set()

    async def _call_planner(self, prompt: str, history: list[ModelMessage]) -> Any:
        with tracer.start_as_current_span(
            "planner", attributes={SpanAttributes.SESSION_ID: self.session_id}
        ):
            return await planning_agent.run(
                prompt,
                message_history=history,
                deps=self,
                conversation_id=self.session_id,
            )

    async def _run_audit(self) -> Any:
        async with self._state_lock:
            transcript = render_dialogue(self.dialogue)
            plan = self.buffer
        with tracer.start_as_current_span(
            "auditor", attributes={SpanAttributes.SESSION_ID: self.session_id}
        ):
            return await auditor_agent.run(
                f"Conversation:\n{transcript}\n\nPlan:\n{plan}",
                conversation_id=self.session_id,
            )

    async def _speaker_loop(self) -> None:
        try:
            while not self._stopped:
                await self._speaker_wakeup.wait()
                self._speaker_wakeup.clear()
                if self._stopped:
                    return
                async with self._speaker_cycle_lock:
                    await self._speaker_cycle()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail("speaker", exc)

    async def _speaker_cycle(self) -> None:
        async with self._state_lock:
            if not self.speaker_pending:
                return
            self.speaker_pending = False
            if self.waiting_for_user:  # Original agent_floor check.
                return

            dialogue_end = len(self.dialogue)
            new_lines = list(self.dialogue[self.speaker_sent : dialogue_end])
            history = list(self.speaker_history)

        prompt = (
            "The plan changed. Decide the single next thing to say to the user, "
            "or stay silent."
        )
        if new_lines:
            prompt += "\n\nNew since your last check:\n" + render_dialogue(new_lines)

        with tracer.start_as_current_span(
            "chat", attributes={SpanAttributes.SESSION_ID: self.session_id}
        ):
            result = await user_agent.run(
                prompt,
                message_history=history,
                deps=self,
                conversation_id=self.session_id,
            )

        out = result.output
        async with self._state_lock:
            self.speaker_history = result.all_messages()
            self.speaker_sent = dialogue_end
            if out.speak and out.message.strip():
                self.dialogue.append(
                    DialogueMessage(role="assistant", content=out.message)
                )
                if out.expects_reply:
                    self.waiting_for_user = True

        if out.speak and out.message.strip():
            await self._outputs.put(
                SystemOutput(
                    kind="message",
                    message=out.message,
                    expects_reply=out.expects_reply,
                )
            )

    def _start_research(self, job_id: str, dispatch_ctx: Any | None = None) -> None:
        if job_id in self._research_tasks:
            return
        self._spawn(
            self._research_worker(job_id, dispatch_ctx),
            f"research:{job_id}",
            job_id,
        )

    async def _restore_research(self) -> None:
        async with self._state_lock:
            job_ids = []
            for job in self.research_jobs:
                if job.status in {"queued", "running"}:
                    job.status = "queued"
                    job_ids.append(job.id)
        for job_id in job_ids:
            self._start_research(job_id)

    async def _research_worker(self, job_id: str, dispatch_ctx: Any | None) -> None:
        try:
            async with self._state_lock:
                job = self._job(job_id)
                job.status = "running"
                target = job.target

            links = [Link(dispatch_ctx)] if dispatch_ctx is not None else []
            with tracer.start_as_current_span(
                "researcher",
                context=otel_context.Context(),
                links=links,
                attributes={
                    SpanAttributes.SESSION_ID: self.session_id,
                    "research.target": target,
                },
            ):
                run = await research_agent.run(
                    f"Please find the following information: {target}",
                    conversation_id=self.session_id,
                )

            out = run.output
            message = (
                f"Research result for '{target}':\n"
                f"answer: {out.answer}\nreasoning: {out.reasoning}"
            )
            async with self._state_lock:
                job = self._job(job_id)
                job.status = "completed"
                job.result = message
                job.completion_order = self.research_completion_order
                self.research_completion_order += 1
                self.planner_pending = True

        except asyncio.CancelledError:
            async with self._state_lock:
                job = self._job(job_id)
                if job.status == "running":
                    job.status = "queued"
            raise
        except Exception as exc:
            async with self._state_lock:
                job = self._job(job_id)
                job.status = "failed"
                job.error = str(exc)
                job.completion_order = self.research_completion_order
                self.research_completion_order += 1
                self.planner_pending = True
        finally:
            if not self._closing:
                self._planner_wakeup.set()

    def _job(self, job_id: str) -> ResearchJob:
        return next(job for job in self.research_jobs if job.id == job_id)

    @staticmethod
    def _render_research(job: ResearchJob) -> str:
        if job.status == "completed":
            return job.result or ""
        return f"Research for '{job.target}' failed: {job.error}"

    @property
    def _stopped(self) -> bool:
        return self._closing or self.is_done or self.failure is not None

    async def _complete(self, note: str) -> None:
        async with self._state_lock:
            self.is_done = True
            self.done_note = note
            final_plan = self.buffer
        self._done.set()
        self._planner_wakeup.set()
        self._speaker_wakeup.set()
        await self._outputs.put(
            SystemOutput(kind="done", message=note, final_plan=final_plan)
        )

    async def _fail(self, source: str, exc: BaseException) -> None:
        error = f"{source} failed: {exc}"
        async with self._state_lock:
            self.failure = error
        self._done.set()
        self._planner_wakeup.set()
        self._speaker_wakeup.set()
        await self._outputs.put(SystemOutput(kind="error", error=error))


planning_agent = Agent(
    smart_model,
    output_type=tool_output(PlanStatus),
    instructions=load_instructions("planner"),
    deps_type=AgentSystem,
    toolsets=[BufferToolset()],
    capabilities=[Instrumentation(InstrumentationSettings(version=2))],
    retries=3,
)

research_agent = Agent(
    research_model,
    instructions=load_instructions("research"),
    output_type=tool_output(ResearchOutput),
    toolsets=[exa_without_citation.exa_toolset],
    capabilities=instrumented_capabilities([exa_without_citation.exa_hooks]),
    retries=3,
)

auditor_agent = Agent(
    auditor_model,
    output_type=tool_output(AuditResult),
    instructions=load_instructions("auditor"),
    capabilities=instrumented_capabilities([]),
    retries=3,
)

user_agent = Agent(
    smart_model,
    deps_type=AgentSystem,
    output_type=ToolOutput(NextMessage, name="NextMessage"),
    instructions=load_instructions("user"),
    capabilities=instrumented_capabilities([]),
    retries=3,
)


@user_agent.instructions
def current_plan(ctx: RunContext[AgentSystem]) -> str:
    if not ctx.deps.buffer:
        return "No plan yet. Warmly greet the user and ask which company they work for. Ask only that."
    return f"Current plan:\n{ctx.deps.buffer}"


@planning_agent.tool
async def spawn_research_agent(ctx: RunContext[AgentSystem], target: str) -> str:
    return await ctx.deps.run_research(target)
