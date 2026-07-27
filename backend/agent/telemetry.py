import os

from opentelemetry import metrics
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

_configured = False
_meter_provider: MeterProvider | None = None


def configure_telemetry(endpoint: str | None = None) -> None:
    global _configured, _meter_provider

    if _configured:
        return

    trace_endpoint = (
        endpoint
        or os.getenv("OTEL_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )
    metrics_endpoint = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    if not trace_endpoint and not metrics_endpoint:
        return

    if trace_endpoint:
        provider = TracerProvider()
        provider.add_span_processor(OpenInferenceSpanProcessor())
        provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(endpoint=trace_endpoint))
        )
        trace.set_tracer_provider(provider)

    if metrics_endpoint:
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=metrics_endpoint)
        )
        _meter_provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(_meter_provider)

    _configured = True
