import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, TypeVar
from uuid import uuid4

import dotenv
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from openinference.semconv.trace import SpanAttributes
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Link
from pydantic_ai import (
    Agent,
    AgentCapability,
    ModelMessage,
    RunContext,
    ToolOutput,
)
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from tools.buffer import BufferToolset
from tools.exa import ExaWithoutCitation

# Setup env

dotenv.load_dotenv()

exa_api_key = os.getenv("EXA_API_KEY")
assert exa_api_key is not None

otel_endpoint = os.getenv("OTEL_ENDPOINT")
assert otel_endpoint is not None

# Setup OTEL reporting

tracer_provider = TracerProvider()
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer("lead-form")

exporter = OTLPSpanExporter(endpoint=otel_endpoint)
tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

exa_without_citation = ExaWithoutCitation(exa_api_key)

# Setup models

smart_model = GoogleModel("gemini-3.1-flash-lite-preview")
research_model = smart_model
auditor_model = smart_model


@dataclass
class buffer_deps:
    buffer: str = ""
    research_out: asyncio.Queue | None = None
    planner_trigger: asyncio.Queue | None = None
    tasks: set = field(default_factory=set)
    session_id: str = ""


@dataclass
class ResearchOutput:
    answer: Any
    reasoning: str


@dataclass
class NextMessage:
    speak: bool
    expects_reply: bool = False
    message: str = ""


@dataclass
class PlanStatus:
    done: bool
    note: str = ""


@dataclass
class AuditResult:
    approved: bool
    issues: str = ""


# Initialization helpers


def load_instructions(agent: str):
    f = open(
        os.path.join(
            os.path.dirname(__file__),
            f"instructions/{agent}.md",
        ),
        "r",
    )
    instructions = f.read()
    f.close()
    return instructions


def instrumented_capabilities(extra: list[AgentCapability]) -> list[AgentCapability]:
    return [Instrumentation(InstrumentationSettings(version=2)), *extra]


OutputT = TypeVar("OutputT")


def tool_output(t: type[OutputT]) -> ToolOutput[OutputT]:
    return ToolOutput(t, name=t.__name__)


# Agent definitions

planning_agent = Agent(
    smart_model,
    output_type=tool_output(PlanStatus),
    instructions=load_instructions("planner"),
    deps_type=buffer_deps,
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
    deps_type=buffer_deps,
    output_type=ToolOutput(NextMessage, name="NextMessage"),
    instructions=load_instructions("user"),
    capabilities=instrumented_capabilities([]),
    retries=3,
)


@user_agent.instructions
def current_plan(ctx: RunContext[buffer_deps]) -> str:
    if not ctx.deps.buffer:
        return "No plan yet. Warmly greet the user and ask which company they work for. Ask only that."
    return f"Current plan:\n{ctx.deps.buffer}"


@planning_agent.tool
async def spawn_research_agent(ctx: RunContext[buffer_deps], target: str) -> str:
    deps = ctx.deps
    dispatch_ctx = trace.get_current_span().get_span_context()

    async def _run() -> None:
        try:
            with tracer.start_as_current_span(
                "researcher",
                context=otel_context.Context(),
                links=[Link(dispatch_ctx)],
                attributes={
                    SpanAttributes.SESSION_ID: deps.session_id,
                    "research.target": target,
                },
            ):
                run = await research_agent.run(
                    f"Please find the following information: {target}",
                    conversation_id=deps.session_id,
                )
            out = run.output
            msg = (
                f"Research result for '{target}':\n"
                f"answer: {out.answer}\nreasoning: {out.reasoning}"
            )
        except Exception as exc:
            msg = f"Research for '{target}' failed: {exc}"
        if deps.research_out is not None:
            deps.research_out.put_nowait(msg)
        if deps.planner_trigger is not None:
            deps.planner_trigger.put_nowait(None)

    task = asyncio.create_task(_run())
    deps.tasks.add(task)
    task.add_done_callback(deps.tasks.discard)
    return f"Research dispatched for '{target}'; result will arrive in a later cycle."


# Agent loops

MAX_AUDIT_RETRIES = 2

AUDIT_FEEDBACK = load_instructions("audit_feedback")


async def run_audit(deps: buffer_deps, dialogue: list[str], session_id: str):
    transcript = "\n".join(dialogue)
    with tracer.start_as_current_span(
        "auditor", attributes={SpanAttributes.SESSION_ID: session_id}
    ):
        return await auditor_agent.run(
            f"Conversation:\n{transcript}\n\nPlan:\n{deps.buffer}",
            conversation_id=session_id,
        )


async def run_planner(
    deps: buffer_deps,
    dialogue: list[str],
    trigger: "asyncio.Queue[None]",
    research_out: "asyncio.Queue[str]",
    notify: "asyncio.Queue[None]",
    done: asyncio.Event,
    session_id: str,
) -> None:
    history: list[ModelMessage] = []
    sent = 0
    while True:
        await trigger.get()
        while not trigger.empty():
            trigger.get_nowait()

        findings = []
        while not research_out.empty():
            findings.append(research_out.get_nowait())

        new_lines = dialogue[sent:]
        sent = len(dialogue)

        parts = []
        if new_lines:
            parts.append("New in the conversation:\n" + "\n".join(new_lines))
        if findings:
            parts.append("New research results:\n\n" + "\n\n".join(findings))
        if not parts:
            parts.append(
                "Nothing new arrived. Re-check the plan and decide the next step."
            )

        with tracer.start_as_current_span(
            "planner", attributes={SpanAttributes.SESSION_ID: session_id}
        ):
            result = await planning_agent.run(
                "\n\n".join(parts),
                message_history=history,
                deps=deps,
                conversation_id=session_id,
            )
        history = result.all_messages()
        notify.put_nowait(None)

        audit_retries = 0
        while result.output.done:
            audit = await run_audit(deps, dialogue, session_id)
            if audit.output.approved:
                print(f"\nPlanner finished: {result.output.note}")
                done.set()
                return
            audit_retries += 1
            if audit_retries > MAX_AUDIT_RETRIES:
                print(
                    "\nPlanner finishing with unresolved audit issues: "
                    f"{audit.output.issues}"
                )
                done.set()
                return
            with tracer.start_as_current_span(
                "planner", attributes={SpanAttributes.SESSION_ID: session_id}
            ):
                result = await planning_agent.run(
                    AUDIT_FEEDBACK.format(issues=audit.output.issues),
                    message_history=history,
                    deps=deps,
                    conversation_id=session_id,
                )
            history = result.all_messages()
            notify.put_nowait(None)


async def run_speaker(
    deps: buffer_deps,
    notify: asyncio.Queue[None],
    dialogue: list[str],
    agent_floor: asyncio.Event,
    user_turn: asyncio.Queue[None],
    session_id: str,
) -> None:
    history: list[ModelMessage] = []
    sent = 0
    while True:
        await notify.get()
        while not notify.empty():
            notify.get_nowait()

        if not agent_floor.is_set():
            continue

        new_lines = dialogue[sent:]
        sent = len(dialogue)
        prompt = (
            "The plan changed. Decide the single next thing to say to the user, "
            "or stay silent."
        )
        if new_lines:
            prompt += "\n\nNew since your last check:\n" + "\n".join(new_lines)

        with tracer.start_as_current_span(
            "chat", attributes={SpanAttributes.SESSION_ID: session_id}
        ):
            result = await user_agent.run(
                prompt,
                message_history=history,
                deps=deps,
                conversation_id=session_id,
            )
        history = result.all_messages()
        out = result.output
        if out.speak and out.message.strip():
            print(f"\n{out.message}")
            dialogue.append(f"Assistant: {out.message}")
            if out.expects_reply:
                agent_floor.clear()
                user_turn.put_nowait(None)


async def main() -> None:
    dialogue: list[str] = []
    trigger = asyncio.Queue[None]()
    research_out = asyncio.Queue[str]()
    notify = asyncio.Queue[None]()
    user_turn = asyncio.Queue[None]()
    agent_floor = asyncio.Event()
    done = asyncio.Event()
    session_id = str(uuid4())

    deps = buffer_deps(
        research_out=research_out, planner_trigger=trigger, session_id=session_id
    )

    planner = asyncio.create_task(
        run_planner(deps, dialogue, trigger, research_out, notify, done, session_id)
    )
    speaker = asyncio.create_task(
        run_speaker(deps, notify, dialogue, agent_floor, user_turn, session_id)
    )

    with tracer.start_as_current_span(
        "chat", attributes={SpanAttributes.SESSION_ID: session_id}
    ):
        opener = await user_agent.run(
            "Greet the user warmly in one line and ask which company they work for. "
            "Ask only that, and set speak=true.",
            deps=deps,
            conversation_id=session_id,
        )
    print(opener.output.message)
    dialogue.append(f"Assistant: {opener.output.message}")

    async def conversation_loop() -> None:
        while True:
            answer = await asyncio.to_thread(input, "> ")
            dialogue.append(f"User: {answer}")
            agent_floor.set()
            trigger.put_nowait(None)
            await user_turn.get()

    conversation_task = asyncio.create_task(conversation_loop())
    done_task = asyncio.create_task(done.wait())

    try:
        await asyncio.wait(
            {conversation_task, done_task, planner, speaker},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        for t in (planner, speaker, conversation_task, done_task):
            t.cancel()
        for t in (planner, speaker, conversation_task, done_task):
            try:
                await t
            except (asyncio.CancelledError, EOFError):
                pass

    if done.is_set():
        print("\nFinal plan:\n")
        print(deps.buffer)


if __name__ == "__main__":
    asyncio.run(main())
