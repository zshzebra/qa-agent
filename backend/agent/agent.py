import asyncio
import os
from dataclasses import dataclass, field
from typing import Any
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
from pydantic_ai import Agent, ModelMessage, RunContext, ToolDefinition, ToolOutput
from pydantic_ai.capabilities import Hooks, Instrumentation
from pydantic_ai.common_tools.exa import ExaToolset
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from tools.buffer import BufferToolset

dotenv.load_dotenv()

exa_api_key = os.getenv("EXA_API_KEY")
assert exa_api_key is not None

tracer_provider = TracerProvider()
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer("lead-form")

endpoint = "http://127.0.0.1:6006/v1/traces"
exporter = OTLPSpanExporter(endpoint=endpoint)
tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

exa_toolset = ExaToolset(
    exa_api_key,
    num_results=5,
    max_characters=1000,
    include_search=False,
    include_find_similar=True,
    include_get_contents=False,
    include_answer=True,
)

hooks = Hooks()


@hooks.on.after_tool_execute
async def process_tool_result(
    ctx, *, call, tool_def: ToolDefinition, args, result: Any
):
    """Tool hook to drop citations from `exa_answer` results."""
    if tool_def.name == "exa_answer":
        return result["answer"]
    return result


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


planning_agent = Agent(
    smart_model,
    output_type=ToolOutput(PlanStatus, name="PlanStatus"),
    instructions=(
        "You run in the background and never speak to the user directly — a "
        "separate conversational agent does that. You maintain a single plan in a "
        "text buffer: read it first, then edit it, do not blindly overwrite.\n\n"
        "Your job is to assemble the company's details and get them confirmed by "
        "the user: name, website, industry, abn, address, phone, email, "
        "date_first_active, founder.\n\n"
        "What makes this feel fast or slow is research latency: a researcher takes "
        "many seconds to return, and its result only reaches you on a later cycle. "
        "So when you start research decides how long the user waits. If you hold "
        "off researching a field until the user brings it up, the user then has to "
        "sit through that researcher's full runtime as dead time. If you start it "
        "the moment you have anything to search on, in the background, the result "
        "is already waiting by the time it's relevant — so it appears instant to "
        "them. Treat any moment a researcher could be running but isn't as latency "
        "you are personally adding to the user's experience.\n\n"
        "For the same reason, fan out: dispatch several focused researchers at "
        "once rather than one broad query. They run concurrently, so several small "
        "searches finish in roughly the time of the slowest one, whereas a single "
        "query covering everything runs those lookups back-to-back. Call "
        "spawn_research_agent to dispatch each; it runs in the background and "
        "returns nothing immediately.\n\n"
        "The user is for confirming and disambiguating what research finds — not "
        "for looking things up. Asking them for a fact research could fetch trades "
        "fast machine work for a slow human round-trip.\n\n"
        "As results arrive, move them into the buffer for confirmation and write "
        "the '## Next message to user' directive to confirm them a few at a time, "
        "grouped naturally — core identity first (website, industry), then "
        "contacts, then registration and founders — because a wall of every field "
        "at once is hard to confirm and slows the user down.\n\n"
        "Each cycle: read the buffer, update it, and write a single 'Next message "
        "to user' directive — the one thing the conversational agent should say or "
        "ask next, or 'nothing yet' when the right move is to wait for pending "
        "research. The conversational agent reads this and decides whether to "
        "actually speak.\n\n"
        "Keep the buffer in this shape:\n"
        "## Confirmed by user\n"
        "## Found via research (awaiting confirmation)\n"
        "## Researching (dispatched — do not re-dispatch)\n"
        "## Not yet started\n"
        "## Next message to user\n"
        "<one short directive, or 'nothing yet'>\n"
        "List a field under 'Researching' only once you have actually dispatched a "
        "researcher for it, so you don't fire the same search twice; anything not "
        "confirmed, found, or dispatched stays under 'Not yet started' so you can "
        "see there is still work to kick off.\n\n"
        "End every turn by returning PlanStatus. Set done=true only when every "
        "field is assembled and confirmed and nothing remains to research or ask; "
        "otherwise done=false, with a short note on what you're waiting on."
    ),
    deps_type=buffer_deps,
    toolsets=[BufferToolset()],
    capabilities=[Instrumentation(InstrumentationSettings(version=2))],
    retries=3,
)


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
        except Exception as exc:  # noqa: BLE001
            msg = f"Research for '{target}' failed: {exc}"
        if deps.research_out is not None:
            deps.research_out.put_nowait(msg)
        if deps.planner_trigger is not None:
            deps.planner_trigger.put_nowait(None)

    task = asyncio.create_task(_run())
    deps.tasks.add(task)
    task.add_done_callback(deps.tasks.discard)
    return f"Research dispatched for '{target}'; result will arrive in a later cycle."


research_agent = Agent(
    research_model,
    instructions=(
        "You are a research sub-agent; your answer is consumed by another agent, "
        "not shown to a user, so be concise and factual. Use exa_answer as your "
        "primary tool. Search thoroughly until you are confident in the answer, or "
        "can clearly explain why it could not be found. Return the specific "
        "fact(s) found — or a clear statement that you could not find them — plus "
        "brief reasoning on how you found it and why it is trustworthy. Never "
        "fabricate — a wrong fact gets surfaced to the user as if true, so an "
        "honest 'not found' is more useful than a guess."
    ),
    output_type=ToolOutput(ResearchOutput, name="ResearchOutput"),
    toolsets=[exa_toolset],
    capabilities=[hooks, Instrumentation(InstrumentationSettings(version=2))],
    retries=3,
)

auditor_agent = Agent(
    auditor_model,
    output_type=ToolOutput(AuditResult, name="AuditResult"),
    instructions=(
        "You are a strict auditor deciding whether a data-collection task is "
        "genuinely complete. You are given the full user conversation and the "
        "planner's plan. Do not trust the plan's '## Confirmed by user' section: "
        "verify it against the conversation. For every field the plan treats as "
        "confirmed, find the user message that actually confirms that specific "
        "field; if there is no such message, it is not confirmed. Also check that "
        "nothing is still awaiting confirmation or being researched, that the last "
        "assistant message does not expect a reply, and that the user has no open "
        "objection or question. Set approved=true only if every confirmed field is "
        "backed by an explicit user confirmation and nothing is pending; otherwise "
        "approved=false and list precisely which fields are unconfirmed and what is "
        "pending in issues, so the planner can fix it."
    ),
    capabilities=[Instrumentation(InstrumentationSettings(version=2))],
    retries=3,
)

user_agent = Agent(
    smart_model,
    deps_type=buffer_deps,
    output_type=ToolOutput(NextMessage, name="NextMessage"),
    instructions=(
        "You are the friendly voice talking to the user. A background planner "
        "keeps a plan in the buffer, including a 'Next message to user' directive.\n"
        "Set three fields:\n"
        "- speak: true if there is something worth saying right now, else false.\n"
        "- message: the one short, warm line to send (empty when speak is false).\n"
        "- expects_reply: true only when your message needs the user to answer "
        "(a question, or a fact to confirm). false for a status update or FYI "
        "such as 'looking that up now', where you are informing them but will "
        "continue on your own.\n\n"
        "Guidance:\n"
        "- Present anything under 'Found via research (awaiting confirmation)' for "
        "the user to confirm ('I found X, does that look right?') with "
        "expects_reply=true. Never ask the user to supply it — the planner is "
        "already fetching it, and asking would waste the user's time on something "
        "arriving anyway.\n"
        "- If the user asks about progress ('have you found them?'), answer "
        "honestly from the current plan — share what is already found and awaiting "
        "confirmation rather than a generic 'still looking', which reads as evasive "
        "when you actually have results.\n"
        "- When presenting found facts for the user to confirm, keep it to a short "
        'paragraph at most, a couple of items at a time, phrased warmly ("I found '
        "your site is example.com and you're in venture capital, is that right?\"). "
        "A long list of every field at once is hard to check, so if the directive "
        "hands you many facts, surface only the first natural group and leave the "
        "rest for the next turn.\n"
        "- Send a status update (speak=true, expects_reply=false) sparingly, only "
        "when a real wait is starting and the user would otherwise be left hanging.\n"
        "- If the directive is 'nothing yet' and there is nothing new worth saying, "
        "set speak=false.\n"
        "- One message at a time. Natural and concise, never a form."
    ),
    capabilities=[Instrumentation(InstrumentationSettings(version=2))],
    retries=3,
)


@user_agent.instructions
def current_plan(ctx: RunContext[buffer_deps]) -> str:
    if not ctx.deps.buffer:
        return "No plan yet. Warmly greet the user and ask which company they work for. Ask only that."
    return f"Current plan:\n{ctx.deps.buffer}"


MAX_AUDIT_RETRIES = 2

AUDIT_FEEDBACK = (
    "A background check found fields marked confirmed that the user never actually "
    "confirmed: {issues}. Do not finish. For each one, move it from '## Confirmed "
    "by user' back to '## Found via research (awaiting confirmation)'. Then write "
    "the '## Next message to user' directive so the conversational agent asks the "
    "user to confirm those specific facts as the natural next step, e.g. 'I also "
    "have your ABN as <value> and your address as <value>, is that correct?'."
)


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
