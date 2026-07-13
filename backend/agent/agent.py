import os
from typing import Any

import dotenv
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic_ai import Agent, ToolDefinition
from pydantic_ai.capabilities import Hooks, Instrumentation
from pydantic_ai.common_tools.exa import ExaToolset
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile

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
    """
    Tool hook to drop citations from `exa_answer` results.
    """
    if tool_def.name == "exa_answer":
        return result["answer"]
    return result


model = OpenAIChatModel(
    "zai-glm-4.7",
    provider="cerebras",
    profile=OpenAIModelProfile(openai_supports_strict_tool_definition=False),
)

agent = Agent(
    model,
    instructions="Be precise and concise. return a single sentence. Use answer tool when looking for an answer, and search to find urls, which is uncommon.",
    toolsets=[exa_toolset],
    capabilities=[hooks, Instrumentation(InstrumentationSettings(version=2))],
)

result = agent.run_sync(input(">"))
print(result.output)
