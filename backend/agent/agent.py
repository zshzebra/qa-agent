import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import dotenv
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic_ai import Agent, ModelSettings, RunContext, ToolDefinition, ToolOutput
from pydantic_ai.capabilities import Hooks, Instrumentation
from pydantic_ai.common_tools.exa import ExaToolset
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from tools.buffer import BufferToolset

dotenv.load_dotenv()

exa_api_key = os.getenv("EXA_API_KEY")
assert exa_api_key is not None

tracer_provider = TracerProvider()
trace.set_tracer_provider(tracer_provider)

endpoint = "http://127.0.0.1:6006/v1/traces"
exporter = OTLPSpanExporter(endpoint=endpoint)
tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

exa_toolset = ExaToolset(
    exa_api_key,
    num_results=5,
    max_characters=1000,
    include_search=True,
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


smart_model = OpenAIChatModel(
    "google/diffusiongemma-26b-a4b-it",
    provider=OpenAIProvider(base_url="http://localhost:8002/v1"),
)

research_model = smart_model


@dataclass
class buffer_deps:
    buffer: str = ""
    research_out: asyncio.Queue | None = None
    planner_trigger: asyncio.Queue | None = None
    tasks: set = field(default_factory=set)


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


planning_agent = Agent(
    smart_model,
    output_type=ToolOutput(PlanStatus, name="PlanStatus"),
    instructions=(
        "You run in the background. A separate conversational agent talks to the "
        "user; you never talk to the user directly. You maintain one plan in the "
        "buffer: read it first, then edit it, do not blindly overwrite.\n\n"
        "Goal: assemble name, abn, website, email, phone, address, industry, "
        "date_first_active, founder, via research and the user's guidance.\n\n"
        "Research: call spawn_research_agent to dispatch a researcher. It runs in "
        "the BACKGROUND and returns nothing immediately; its result comes back to "
        "you as input in a later cycle. Record what you have dispatched under "
        "'Unknown / researching' so you do not re-dispatch it. Research can find "
        "most fields; the user is best for the company name, industry, "
        "disambiguation, and confirming found facts. You MAY still ask the user "
        "about a field research is working on when genuinely useful, but avoid "
        "firing a question research is about to answer anyway.\n\n"
        "Each cycle, update the buffer and write a single 'Next message to user' "
        "directive: the one thing the conversational agent should say or ask next, "
        "or 'nothing yet' if the right move is to wait for pending research."
        "conversational agent reads this and decides whether to actually speak.\n\n"
        "When surfacing found facts for confirmation, do it a few at a time, "
        "grouped naturally (say the core identity first: website and industry; "
        "then contact details; then registration and founders). Do not ask the "
        "user to confirm everything in one directive.\n\n"
        "The conversational agent reads this and decides whether to actually speak.\n\n"
        "Keep the buffer in this shape, current:\n"
        "## Confirmed by user\n"
        "## Found via research (awaiting confirmation)\n"
        "## Unknown / researching\n"
        "## Next message to user\n"
        "<one short directive, or 'nothing yet'>"
        "\n\nEnd every turn by returning PlanStatus. Set done=true only when every "
        "field is assembled and confirmed and nothing remains to research or ask; "
        "otherwise done=false. Put a brief closing summary in note when done, or a "
        "short status of what you are waiting on when not."
    ),
    deps_type=buffer_deps,
    toolsets=[BufferToolset()],
    capabilities=[Instrumentation(InstrumentationSettings(version=2))],
    retries=3,
)


@planning_agent.tool
async def spawn_research_agent(ctx: RunContext[buffer_deps], target: str) -> str:
    deps = ctx.deps

    async def _run() -> None:
        try:
            run = await research_agent.run(
                f"Please find the following information: {target}"
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
    instructions="Using exa answer primarily, you need to research the given question until you are confident, or explain why you could not find the answer. You must include reasoning as to why your answer was good or not.",
    output_type=ToolOutput(ResearchOutput, name="ResearchOutput"),
    toolsets=[exa_toolset],
    capabilities=[hooks, Instrumentation(InstrumentationSettings(version=2))],
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
        "expects_reply=true. Never ask the user to supply it.\n"
        "- When presenting found facts for the user to confirm, keep it to a short "
        'paragraph at most, a couple of items at a time, phrased warmly ("I found '
        "your site is example.com and you're in venture capital, is that right?\"). "
        "Never recite a long list of every field in one message. If the directive "
        "hands you many facts at once, surface only the first natural group and "
        "leave the rest for the next turn.\n"
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


async def run_planner(
    deps: buffer_deps,
    dialogue: list[str],
    trigger: "asyncio.Queue[None]",
    research_out: "asyncio.Queue[str]",
    notify: "asyncio.Queue[None]",
    done: asyncio.Event,
) -> None:
    while True:
        await trigger.get()
        while not trigger.empty():
            trigger.get_nowait()

        findings = []
        while not research_out.empty():
            findings.append(research_out.get_nowait())

        conversation = "\n".join(dialogue) or "(no conversation yet)"
        parts = [f"Current conversation:\n{conversation}"]
        if findings:
            parts.append("New research results:\n\n" + "\n\n".join(findings))

        result = await planning_agent.run("\n\n".join(parts), deps=deps)
        notify.put_nowait(None)
        if result.output.done:
            print(f"\nPlanner finished: {result.output.note}")
            done.set()
            return


async def run_speaker(
    deps: buffer_deps,
    notify: asyncio.Queue[None],
    dialogue: list[str],
    agent_floor: asyncio.Event,
    user_turn: asyncio.Queue[None],
) -> None:
    while True:
        await notify.get()
        while not notify.empty():
            notify.get_nowait()

        if not agent_floor.is_set():
            continue

        conversation = "\n".join(dialogue) or "(conversation not started)"
        result = await user_agent.run(
            "The plan changed. Decide the single next thing to say to the user, "
            f"or stay silent.\n\nConversation so far:\n{conversation}",
            deps=deps,
        )
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

    deps = buffer_deps(research_out=research_out, planner_trigger=trigger)

    planner = asyncio.create_task(
        run_planner(deps, dialogue, trigger, research_out, notify, done)
    )
    speaker = asyncio.create_task(
        run_speaker(deps, notify, dialogue, agent_floor, user_turn)
    )

    opener = await user_agent.run(
        "Greet the user warmly in one line and ask which company they work for. "
        "Ask only that, and set speak=true.",
        deps=deps,
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
