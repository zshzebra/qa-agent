import os

from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

_configured = False


def configure_telemetry(endpoint: str | None = None) -> None:
    global _configured

    if _configured:
        return

    endpoint = endpoint or os.getenv("OTEL_ENDPOINT")
    if not endpoint:
        return

    provider = TracerProvider()
    provider.add_span_processor(OpenInferenceSpanProcessor())
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    )
    trace.set_tracer_provider(provider)

    _configured = True
