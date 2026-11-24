from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from time import monotonic
from typing import Any, cast
from uuid import uuid4

from openinference.semconv.trace import SpanAttributes
from opentelemetry import context as otel_context
from opentelemetry import metrics
from opentelemetry import trace
from opentelemetry.trace import Link
from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    AgentCapability,
    AgentRunResult,
    ModelMessage,
    ModelMessagesTypeAdapter,
    RunContext,
    ToolOutput,
)
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models import Model
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_core import to_jsonable_python

from .models import (
    ActiveResearcher,
    AuditIssue,
    AuditResult,
    ContinuePlanning,
    ConversationState,
    DialogueMessage,
    NextMessage,
    ResearchJob,
    ResearchOutput,
    SystemOutput,
    render_dialogue,
)
from .prompts import PromptRenderer
from .target import describe_target
from .tools.buffer import BufferToolset
from .tools.exa import ExaWithoutCitation


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required but is not set in the environment")
    return value


tracer = trace.get_tracer("lead-form")
meter = metrics.get_meter("lead-form")
quiescence_detections = meter.create_counter(
    "lead_form.quiescence.detections",
    description="Number of times an unfinished session had no owner or active work.",
)
reconciliation_outcomes = meter.create_counter(
    "lead_form.reconciliation.outcomes",
    description="Outcomes of planner reconciliation cycles.",
)
reconciliation_duration = meter.create_histogram(
    "lead_form.reconciliation.duration",
    unit="s",
    description="Time from quiescence detection to a live session outcome.",
)

def instrumented_capabilities(extra: list[AgentCapability]) -> list[AgentCapability]:
    return [Instrumentation(InstrumentationSettings(version=2)), *extra]


def tool_output[OutputT](t: type[OutputT]) -> ToolOutput[OutputT]:
    return ToolOutput(t, name=t.__name__)


MAX_AUDIT_RETRIES = 2


class AgentSystem[OutputT: BaseModel]:
    def __init__(
        self,
        *,
        output_type: type[OutputT],
        planner_model: Model,
        speaker_model: Model,
        researcher_model: Model,
        auditor_model: Model,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.output_type = output_type
        self.target = describe_target(output_type)
        self.prompts = PromptRenderer()
        state = state or {}

        # Serializable logical state.
        self.session_id: str = state.get("session_id", str(uuid4()))
        self.buffer: str = (
            state["buffer"]
            if "buffer" in state
            else self.prompts.render("initial_buffer.md.j2", target=self.target)
        )
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
        self.planner_trigger: str = state.get("planner_trigger", "normal")
        self.reconciliation_attempted: bool = state.get(
            "reconciliation_attempted", False
        )
        self.waiting_for_user: bool = state.get("waiting_for_user", False)
        self.is_done: bool = state.get("is_done", False)
        self.done_note: str = state.get("done_note", "")
        serialized_result = state.get("result")
        self.result: OutputT | None = (
            output_type.model_validate(serialized_result)
            if serialized_result is not None
            else None
        )
        self.failure: str | None = state.get("failure")
        if self.is_done and self.result is None:
            raise ValueError("Completed state is missing its structured result")

        # Runtime-only state.
        self._state_lock = asyncio.Lock()
        self._planner_cycle_lock = asyncio.Lock()
        self._speaker_cycle_lock = asyncio.Lock()
        self._planner_wakeup = asyncio.Event()
        self._speaker_wakeup = asyncio.Event()
        self._done = asyncio.Event()
        self._outputs: asyncio.Queue[SystemOutput[OutputT]] = asyncio.Queue()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._research_tasks: dict[str, asyncio.Task[None]] = {}
        self._planner_active = False
        self._speaker_active = False
        self._reconciliation_id: str | None = None
        self._reconciliation_started_at: float | None = None
        self._reconciliation_link: Any | None = None
        self._started = False
        self._closing = False

        self._create_agents(
            planner_model=planner_model,
            speaker_model=speaker_model,
            researcher_model=researcher_model,
            auditor_model=auditor_model,
        )

    def _create_agents(
        self,
        *,
        planner_model: Model,
        speaker_model: Model,
        researcher_model: Model,
        auditor_model: Model,
    ) -> None:
        self._exa_without_citation = ExaWithoutCitation(
            require_env("EXA_API_KEY"), self._record_sources
        )

        self.planning_agent: Agent[
            AgentSystem[Any],
            ContinuePlanning | OutputT,
        ] = Agent(
            planner_model,
            output_type=[
                ToolOutput(
                    ContinuePlanning,
                    name="continue_planning",
                ),
                ToolOutput(
                    self.output_type,
                    name="submit_result",
                    description=(
                        "Submit the complete target object for independent auditing."
                    ),
                ),
            ],
            instructions=self.prompts.render(
                "planner.md.j2",
                target=self.target,
            ),
            deps_type=AgentSystem,
            toolsets=[
                BufferToolset(
                    instructions=self.prompts.render("buffer_tools.md.j2"),
                )
            ],
            capabilities=[Instrumentation(InstrumentationSettings(version=2))],
            retries=3,
        )
        self.planning_agent.tool(name="spawn_research_agent")(
            self._spawn_research_agent
        )

        self.research_agent = Agent(
            researcher_model,
            instructions=self.prompts.render(
                "researcher.md.j2",
                target=self.target,
            ),
            deps_type=str,
            output_type=tool_output(ResearchOutput),
            toolsets=[self._exa_without_citation.exa_toolset],
            capabilities=instrumented_capabilities(
                [self._exa_without_citation.exa_hooks]
            ),
            retries=3,
        )

        self.auditor_agent = Agent(
            auditor_model,
            output_type=tool_output(AuditResult),
            instructions=self.prompts.render(
                "auditor.md.j2",
                target=self.target,
            ),
            capabilities=instrumented_capabilities([]),
            retries=3,
        )

        self.speaker_agent = Agent(
            speaker_model,
            deps_type=AgentSystem,
            output_type=ToolOutput(NextMessage, name="NextMessage"),
            instructions=self.prompts.render(
                "speaker.md.j2",
                target=self.target,
            ),
            capabilities=instrumented_capabilities([]),
            retries=3,
        )
        self.speaker_agent.instructions(self._current_plan_instruction)

    @classmethod
    def deserialize(
        cls,
        value: str,
        *,
        output_type: type[OutputT],
        planner_model: Model,
        speaker_model: Model,
        researcher_model: Model,
        auditor_model: Model,
    ) -> AgentSystem[OutputT]:
        return cls(
            output_type=output_type,
            planner_model=planner_model,
            speaker_model=speaker_model,
            researcher_model=researcher_model,
            auditor_model=auditor_model,
            state=json.loads(value),
        )

    def _current_plan_instruction(
        self, ctx: RunContext[AgentSystem[Any]]
    ) -> str:
        return self.prompts.render(
            "speaker_context.md.j2",
            buffer=ctx.deps.buffer,
        )

    async def _spawn_research_agent(
        self, ctx: RunContext[AgentSystem[Any]], target: str
    ) -> str:
        return await ctx.deps.run_research(target)

    async def conversation_state(self) -> ConversationState[OutputT]:
        async with self._state_lock:
            return ConversationState(
                dialogue=tuple(
                    DialogueMessage(role=message.role, content=message.content)
                    for message in self.dialogue
                ),
                waiting_for_user=self.waiting_for_user,
                is_done=self.is_done,
                done_note=self.done_note,
                result=self.result,
                failure=self.failure,
                researchers=tuple(
                    ActiveResearcher(id=job.id, domains=tuple(job.domains))
                    for job in sorted(
                        (j for j in self.research_jobs if j.status == "running"),
                        key=lambda j: j.started_at or 0.0,
                    )
                ),
            )

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
                "planner_trigger": self.planner_trigger,
                "reconciliation_attempted": self.reconciliation_attempted,
                "waiting_for_user": self.waiting_for_user,
                "is_done": self.is_done,
                "done_note": self.done_note,
                "result": (
                    self.result.model_dump(mode="json")
                    if self.result is not None
                    else None
                ),
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
                SystemOutput(kind="done", message=self.done_note, result=self.result)
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
        await self._check_quiescence("startup")

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
            if not self.waiting_for_user and not self._is_quiescent_locked():
                raise RuntimeError("The agent system is not waiting for user input")
            self.dialogue.append(
                DialogueMessage(
                    role="user",
                    content=message,
                )
            )
            self.waiting_for_user = False
            self.reconciliation_attempted = False
            self._clear_reconciliation_locked()
            self.planner_trigger = "normal"
            self.planner_pending = True
        self._planner_wakeup.set()

    async def receive(self) -> SystemOutput[OutputT]:
        return await self._outputs.get()

    async def wait_done(self) -> None:
        await self._done.wait()

    def _is_quiescent_locked(self) -> bool:
        return (
            self._started
            and not self._closing
            and not self.is_done
            and self.failure is None
            and not self.waiting_for_user
            and not self.planner_pending
            and not self.speaker_pending
            and not self._planner_active
            and not self._speaker_active
            and not any(
                job.status in {"queued", "running"} for job in self.research_jobs
            )
            and not any(
                job.status in {"completed", "failed"} and not job.consumed
                for job in self.research_jobs
            )
        )

    async def _check_quiescence(self, source: str = "unknown") -> None:
        schedule_reconciliation = False
        force_user_handoff = False
        reconciliation = None
        detection_attributes: dict[str, Any] = {}

        async with self._state_lock:
            if not self._is_quiescent_locked():
                return

            queued = sum(job.status == "queued" for job in self.research_jobs)
            running = sum(job.status == "running" for job in self.research_jobs)
            unconsumed = sum(
                job.status in {"completed", "failed"} and not job.consumed
                for job in self.research_jobs
            )
            detection_attributes = {
                SpanAttributes.SESSION_ID: self.session_id,
                "lead_form.quiescence.source": source,
                "lead_form.quiescence.reconciliation_attempted": (
                    self.reconciliation_attempted
                ),
                "lead_form.planner.pending": self.planner_pending,
                "lead_form.speaker.pending": self.speaker_pending,
                "lead_form.planner.active": self._planner_active,
                "lead_form.speaker.active": self._speaker_active,
                "lead_form.research.queued": queued,
                "lead_form.research.running": running,
                "lead_form.research.unconsumed": unconsumed,
            }

            if self.reconciliation_attempted:
                self.waiting_for_user = True
                reconciliation = self._finish_reconciliation_locked(
                    "forced_user_handoff"
                )
                force_user_handoff = True
            else:
                self.reconciliation_attempted = True
                self.planner_trigger = "reconciliation"
                self.planner_pending = True
                self._reconciliation_id = str(uuid4())
                self._reconciliation_started_at = monotonic()
                detection_attributes["lead_form.reconciliation.id"] = (
                    self._reconciliation_id
                )
                schedule_reconciliation = True

        quiescence_detections.add(
            1,
            {
                "source": source,
                "reconciliation_attempted": str(
                    detection_attributes[
                        "lead_form.quiescence.reconciliation_attempted"
                    ]
                ).lower(),
            },
        )
        with tracer.start_as_current_span(
            "lead_form.quiescence.detected",
            attributes=detection_attributes,
        ) as span:
            span.add_event(
                "reconciliation.scheduled"
                if schedule_reconciliation
                else "user_handoff.forced"
            )
            if schedule_reconciliation:
                span_context = span.get_span_context()
                async with self._state_lock:
                    if self.reconciliation_attempted:
                        self._reconciliation_link = span_context

        if schedule_reconciliation:
            self._planner_wakeup.set()
        if force_user_handoff:
            self._record_reconciliation_outcome(reconciliation)
            await self._outputs.put(
                SystemOutput(kind="state", expects_reply=True)
            )

    def _clear_reconciliation_locked(self) -> None:
        self._reconciliation_id = None
        self._reconciliation_started_at = None
        self._reconciliation_link = None

    def _finish_reconciliation_locked(
        self, outcome: str
    ) -> dict[str, Any] | None:
        if not self.reconciliation_attempted:
            return None

        reconciliation = {
            "id": self._reconciliation_id,
            "started_at": self._reconciliation_started_at,
            "link": self._reconciliation_link,
            "outcome": outcome,
        }
        self.reconciliation_attempted = False
        self._clear_reconciliation_locked()
        return reconciliation

    def _record_reconciliation_outcome(
        self, reconciliation: dict[str, Any] | None
    ) -> None:
        if reconciliation is None:
            return

        outcome = str(reconciliation["outcome"])
        started_at = reconciliation["started_at"]
        duration = monotonic() - started_at if started_at is not None else None
        link = reconciliation["link"]
        links = [Link(link)] if link is not None and link.is_valid else []
        attributes: dict[str, Any] = {
            SpanAttributes.SESSION_ID: self.session_id,
            "lead_form.reconciliation.outcome": outcome,
        }
        if reconciliation["id"] is not None:
            attributes["lead_form.reconciliation.id"] = reconciliation["id"]
        if duration is not None:
            attributes["lead_form.reconciliation.duration_seconds"] = duration

        reconciliation_outcomes.add(1, {"outcome": outcome})
        if duration is not None:
            reconciliation_duration.record(duration, {"outcome": outcome})
        with tracer.start_as_current_span(
            "lead_form.reconciliation.outcome",
            links=links,
            attributes=attributes,
        ):
            pass

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
        reconciliation = None
        async with self._state_lock:
            job = ResearchJob(id=str(uuid4()), target=target)
            self.research_jobs.append(job)
            reconciliation = self._finish_reconciliation_locked("research_started")
        self._start_research(job.id, dispatch_ctx)
        self._record_reconciliation_outcome(reconciliation)
        return self.prompts.render(
            "research_dispatched.md.j2",
            research_target=target,
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
                opener = await self.speaker_agent.run(
                    self.prompts.render(
                        "opener.md.j2",
                        target=self.target,
                    ),
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
                    async with self._state_lock:
                        self._planner_active = True
                    try:
                        await self._planner_cycle()
                    finally:
                        async with self._state_lock:
                            self._planner_active = False
                await self._check_quiescence("planner_cycle")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail("planner", exc)

    async def _planner_cycle(self) -> None:
        async with self._state_lock:
            if not self.planner_pending:
                return
            self.planner_pending = False
            trigger = self.planner_trigger
            self.planner_trigger = "normal"
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
            transcript = render_dialogue(self.dialogue)
            plan = self.buffer
            research_state = [
                {
                    "target": job.target,
                    "status": job.status,
                    "consumed": job.consumed,
                }
                for job in self.research_jobs
            ]
            reconciliation_link = self._reconciliation_link
            reconciliation_id = self._reconciliation_id

        if trigger == "reconciliation":
            prompt = self.prompts.render(
                "reconcile.md.j2",
                target=self.target,
                conversation=transcript,
                buffer=plan,
                research_state=research_state,
            )
        else:
            prompt = self.prompts.render(
                "planner_turn.md.j2",
                new_dialogue=render_dialogue(new_lines),
                findings=[self._render_research(job) for job in findings],
            )

        planner_run = await self._call_planner(
            prompt,
            history,
            trigger=trigger,
            reconciliation_link=reconciliation_link,
            reconciliation_id=reconciliation_id,
        )
        finding_ids = {job.id for job in findings}
        async with self._state_lock:
            self.planner_history = planner_run.all_messages()
            self.planner_sent = dialogue_end
            for job in self.research_jobs:
                if job.id in finding_ids:
                    job.consumed = True

        audit_retries = 0
        while not isinstance(planner_run.output, ContinuePlanning):
            candidate = cast(OutputT, planner_run.output)

            audit = await self._run_audit(candidate)
            if not audit.output.issues:
                await self._complete(candidate)
                return

            audit_retries += 1
            if audit_retries > MAX_AUDIT_RETRIES:
                await self._fail(
                    "audit",
                    RuntimeError(
                        "structured completion was rejected after "
                        f"{audit_retries} attempts: "
                        f"{self._render_audit_issues(audit.output.issues)}"
                    ),
                )
                return

            planner_run = await self._call_planner(
                self.prompts.render(
                    "audit_feedback.md.j2",
                    candidate=candidate.model_dump(mode="json"),
                    issues=[
                        issue.model_dump(mode="json")
                        for issue in audit.output.issues
                    ],
                ),
                list(self.planner_history),
                trigger=trigger,
                reconciliation_link=reconciliation_link,
                reconciliation_id=reconciliation_id,
            )
            async with self._state_lock:
                self.planner_history = planner_run.all_messages()

        async with self._state_lock:
            self.speaker_pending = True
        self._speaker_wakeup.set()

    async def _call_planner(
        self,
        prompt: str,
        history: list[ModelMessage],
        *,
        trigger: str = "normal",
        reconciliation_link: Any | None = None,
        reconciliation_id: str | None = None,
    ) -> AgentRunResult[ContinuePlanning | OutputT]:
        links = (
            [Link(reconciliation_link)]
            if reconciliation_link is not None and reconciliation_link.is_valid
            else []
        )
        attributes: dict[str, Any] = {
            SpanAttributes.SESSION_ID: self.session_id,
            "lead_form.planner.trigger": trigger,
        }
        if reconciliation_id is not None:
            attributes["lead_form.reconciliation.id"] = reconciliation_id
        with tracer.start_as_current_span(
            "planner",
            links=links,
            attributes=attributes,
        ):
            return await self.planning_agent.run(
                prompt,
                message_history=history,
                deps=self,
                conversation_id=self.session_id,
            )

    async def _run_audit(self, candidate: OutputT) -> Any:
        async with self._state_lock:
            transcript = render_dialogue(self.dialogue)
            plan = self.buffer
            research_state = [
                {
                    "target": job.target,
                    "status": job.status,
                    "consumed": job.consumed,
                }
                for job in self.research_jobs
            ]
        with tracer.start_as_current_span(
            "auditor", attributes={SpanAttributes.SESSION_ID: self.session_id}
        ):
            return await self.auditor_agent.run(
                self.prompts.render(
                    "audit_request.md.j2",
                    target=self.target,
                    candidate=candidate.model_dump(mode="json"),
                    conversation=transcript,
                    buffer=plan,
                    research_state=research_state,
                ),
                conversation_id=self.session_id,
            )

    @staticmethod
    def _render_audit_issues(issues: list[AuditIssue]) -> str:
        return "; ".join(
            (
                f"{issue.field}: {issue.reason}"
                if issue.field is not None
                else issue.reason
            )
            for issue in issues
        )

    async def _speaker_loop(self) -> None:
        try:
            while not self._stopped:
                await self._speaker_wakeup.wait()
                self._speaker_wakeup.clear()
                if self._stopped:
                    return
                async with self._speaker_cycle_lock:
                    async with self._state_lock:
                        self._speaker_active = True
                    try:
                        await self._speaker_cycle()
                    finally:
                        async with self._state_lock:
                            self._speaker_active = False
                await self._check_quiescence("speaker_cycle")
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
            speaker_trigger = (
                "reconciliation" if self.reconciliation_attempted else "normal"
            )
            reconciliation_link = self._reconciliation_link
            reconciliation_id = self._reconciliation_id

        prompt = self.prompts.render(
            "speaker_turn.md.j2",
            new_dialogue=render_dialogue(new_lines),
        )

        attributes: dict[str, Any] = {
            SpanAttributes.SESSION_ID: self.session_id,
            "lead_form.chat.trigger": speaker_trigger,
        }
        if reconciliation_id is not None:
            attributes["lead_form.reconciliation.id"] = reconciliation_id
        with tracer.start_as_current_span(
            "chat",
            links=(
                [Link(reconciliation_link)]
                if reconciliation_link is not None and reconciliation_link.is_valid
                else []
            ),
            attributes=attributes,
        ):
            result = await self.speaker_agent.run(
                prompt,
                message_history=history,
                deps=self,
                conversation_id=self.session_id,
            )

        out = result.output
        reconciliation = None
        async with self._state_lock:
            self.speaker_history = result.all_messages()
            self.speaker_sent = dialogue_end
            if out.speak and out.message.strip():
                self.dialogue.append(
                    DialogueMessage(role="assistant", content=out.message)
                )
                if out.expects_reply:
                    self.waiting_for_user = True
                    reconciliation = self._finish_reconciliation_locked(
                        "waiting_for_user"
                    )

        if out.speak and out.message.strip():
            await self._outputs.put(
                SystemOutput(
                    kind="message",
                    message=out.message,
                    expects_reply=out.expects_reply,
                )
            )
        self._record_reconciliation_outcome(reconciliation)

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
                job.started_at = monotonic()
                target = job.target
            await self._progress()

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
                run = await self.research_agent.run(
                    self.prompts.render(
                        "research_request.md.j2",
                        research_target=target,
                    ),
                    conversation_id=self.session_id,
                    deps=job_id,
                )

            out = run.output
            message = self.prompts.render(
                "research_result.md.j2",
                research_target=target,
                answer=out.answer,
                reasoning=out.reasoning,
            )
            async with self._state_lock:
                job = self._job(job_id)
                job.status = "completed"
                job.result = message
                job.completion_order = self.research_completion_order
                self.research_completion_order += 1
                self.planner_pending = True
            await self._progress()

        except asyncio.CancelledError:
            async with self._state_lock:
                job = self._job(job_id)
                if job.status == "running":
                    job.status = "queued"
            if not self._closing:
                await self._progress()
            raise
        except Exception as exc:
            async with self._state_lock:
                job = self._job(job_id)
                job.status = "failed"
                job.error = str(exc)
                job.completion_order = self.research_completion_order
                self.research_completion_order += 1
                self.planner_pending = True
            await self._progress()
        finally:
            if not self._closing:
                self._planner_wakeup.set()

    async def _record_sources(self, job_id: str, domains: list[str]) -> None:
        async with self._state_lock:
            job = self._job(job_id)
            if job.status != "running":
                return
            job.domains.extend(domains)
        await self._progress()

    async def _progress(self) -> None:
        """Tell observers that the set of active researchers changed."""
        await self._outputs.put(SystemOutput(kind="progress"))

    def _job(self, job_id: str) -> ResearchJob:
        return next(job for job in self.research_jobs if job.id == job_id)

    def _render_research(self, job: ResearchJob) -> str:
        if job.status == "completed":
            return job.result or ""
        return self.prompts.render(
            "research_failed.md.j2",
            research_target=job.target,
            error=job.error,
        )

    @property
    def _stopped(self) -> bool:
        return self._closing or self.is_done or self.failure is not None

    async def _complete(self, result: OutputT) -> None:
        reconciliation = None
        async with self._state_lock:
            self.result = result
            self.is_done = True
            self.done_note = self.prompts.render("completion.md.j2")
            reconciliation = self._finish_reconciliation_locked("completed")
        self._record_reconciliation_outcome(reconciliation)
        self._done.set()
        self._planner_wakeup.set()
        self._speaker_wakeup.set()
        await self._outputs.put(
            SystemOutput(kind="done", message=self.done_note, result=result)
        )

    async def _fail(self, source: str, exc: BaseException) -> None:
        error = f"{source} failed: {exc}"
        reconciliation = None
        async with self._state_lock:
            self.failure = error
            reconciliation = self._finish_reconciliation_locked("failed")
        self._record_reconciliation_outcome(reconciliation)
        self._done.set()
        self._planner_wakeup.set()
        self._speaker_wakeup.set()
        await self._outputs.put(SystemOutput(kind="error", error=error))
