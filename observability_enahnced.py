"""
Enhanced OpenTelemetry Implementation for incident-manager (new_flow)

Features:
- Log ↔ Trace correlation (trace_id in log records)
- Span events from log records
- Better span hierarchy (agent_run → phase → tool)
- Content truncation with metadata
- Enhanced error tracking
"""

import os
import logging
import traceback
import contextvars
from typing import Any, Tuple

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_NAMESPACE, DEPLOYMENT_ENVIRONMENT
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace import SpanKind

logger = logging.getLogger(__name__)

_tracer = None
_tracing_enabled = False
_provider = None
_log_handler_installed = False
_current_span: contextvars.ContextVar[Any] = contextvars.ContextVar("current_otel_span", default=None)

_CAPTURE_CONTENT = os.getenv("OTEL_CAPTURE_CONTENT", "true").lower() != "false"
_OUTPUT_MAX_CHARS = int(os.getenv("OTEL_OUTPUT_MAX_CHARS", "2000"))
_INPUT_MAX_CHARS = int(os.getenv("OTEL_INPUT_MAX_CHARS", "500"))


def _truncate_content(content: Any, max_chars: int) -> Tuple[Any, bool]:
    if content is None:
        return None, False
    content_str = str(content)
    if len(content_str) > max_chars:
        return content_str[:max_chars], True
    return content, False


def init_telemetry():
    global _tracer, _tracing_enabled, _provider
    try:
        if _provider is not None:
            logger.info("Telemetry already initialized, skipping...")
            return _provider

        set_global_textmap(TraceContextTextMapPropagator())
        service_name = os.getenv("TELEMETRY_SERVICE_NAME", "genai-de-incident-manager")
        telemetry_endpoint = os.getenv("TELEMETRY_ENDPOINT")
        telemetry_port = os.getenv("TELEMETRY_PORT", "4318")
        environment = os.getenv("ENVIRONMENT", "uat")

        if not telemetry_endpoint:
            logger.warning("TELEMETRY_ENDPOINT not set — telemetry disabled")
            return None

        resource = Resource.create({
            SERVICE_NAME: service_name,
            SERVICE_NAMESPACE: "ai-platform",
            DEPLOYMENT_ENVIRONMENT: environment,
            "service.version": os.getenv("APP_VERSION", "unknown"),
            "host.name": os.getenv("HOSTNAME", "unknown"),
        })

        endpoint_url = f"http://{telemetry_endpoint}:{telemetry_port}/v1/traces"
        otlp_exporter = OTLPSpanExporter(
            endpoint=endpoint_url,
            timeout=10,
            headers={"Authorization": f"Bearer {os.getenv('OTEL_AUTH_TOKEN', '')}"} if os.getenv('OTEL_AUTH_TOKEN') else {}
        )

        _provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(_provider)
        _provider.add_span_processor(BatchSpanProcessor(otlp_exporter, max_queue_size=2048, max_export_batch_size=512, export_timeout_millis=30000))
        _tracer = trace.get_tracer(service_name)
        _tracing_enabled = True

        _instrument_libraries()
        _install_log_correlation()

        logger.info(f"OpenTelemetry initialized: service={service_name}, env={environment}, endpoint={endpoint_url}")
        return _provider

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to initialize telemetry: {e}")
        return None


def get_tracer(name: str = 'genai-de-incident-agent'):
    global _tracer
    if _tracer:
        return _tracer
    return trace.get_tracer(name)


def get_provider():
    global _provider
    if _provider:
        return _provider
    return trace.get_tracer_provider()


def is_tracing_enabled() -> bool:
    return _tracing_enabled


def shutdown_telemetry():
    global _provider
    if _provider:
        try:
            _provider.shutdown()
            logger.info("Telemetry shutdown complete")
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")


def _instrument_libraries():
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPX auto-instrumented")
    except ImportError:
        pass
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        RequestsInstrumentor().instrument()
        logger.info("Requests auto-instrumented")
    except ImportError:
        pass
    try:
        from opentelemetry.instrumentation.aiohttp import AioHttpClientInstrumentor
        AioHttpClientInstrumentor().instrument()
        logger.info("aiohttp auto-instrumented")
    except ImportError:
        pass


# Log Correlation
def _install_log_correlation():
    global _log_handler_installed
    if _log_handler_installed:
        return
    try:
        _setup_trace_context_filter()
        _log_handler_installed = True
        logger.info("Log↔trace correlation installed")
    except Exception as e:
        logger.warning(f"Failed to install log correlation: {e}")


def _setup_trace_context_filter():
    class TraceContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            trace_id = "-"
            span_id = "-"
            try:
                span = _current_span.get()
                if span is not None:
                    ctx = span.get_span_context()
                    trace_id = format(ctx.trace_id, "032x")
                    span_id = format(ctx.span_id, "016x")
                else:
                    ctx = trace.get_current_span()
                    if ctx:
                        ctx_sanitized = ctx.get_span_context()
                        trace_id = format(ctx_sanitized.trace_id, "032x")
                        span_id = format(ctx_sanitized.span_id, "016x")
            except Exception:
                pass
            record.trace_id = trace_id
            record.span_id = span_id
            return True
    
    root_logger = logging.getLogger("new_flow")
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s span_id=%(span_id)s] %(message)s'))
        root_logger.addHandler(handler)
    for handler in root_logger.handlers:
        if not any(isinstance(f, TraceContextFilter) for f in handler.filters):
            handler.addFilter(TraceContextFilter())


def active_span_scope(span: Any):
    token = _current_span.set(span)
    try:
        yield
    finally:
        _current_span.reset(token)


def get_current_span() -> Any:
    span = _current_span.get()
    if span is not None:
        return span
    return trace.get_current_span()


# Enhanced Span Helpers
def set_span_attribute(span: Any, key: str, value: Any, max_length: int = None):
    if span is None:
        return
    try:
        if max_length and value:
            value_str = str(value)
            if len(value_str) > max_length:
                span.set_attribute(f"{key}.truncated", True)
                span.set_attribute(f"{key}.length", len(value_str))
                value = value_str[:max_length]
        span.set_attribute(key, value)
    except Exception:
        pass


def record_span_event(span: Any, event_name: str, metadata: dict = None):
    if span is None:
        return
    try:
        span.add_event(event_name, metadata or {})
    except Exception:
        pass


def record_span_exception(span: Any, exception: Exception, message: str = None):
    if span is None:
        return
    try:
        span.set_attribute("error", True)
        span.set_attribute("error.message", str(message or exception))
        span.record_exception(exception)
    except Exception:
        pass


# Context Manager Wrappers
def create_span(name: str, incident_id: str = None, incident_no: str = None, event_type: str = None, as_type: str = "span", kind: str = "internal", input_data: Any = None, metadata: dict = None):
    tracer = get_tracer(__name__)
    kind_map = {"internal": SpanKind.INTERNAL, "client": SpanKind.CLIENT, "server": SpanKind.SERVER, "producer": SpanKind.PRODUCER, "consumer": SpanKind.CONSUMER}
    otel_kind = kind_map.get(kind.lower(), SpanKind.INTERNAL)
    span = tracer.start_span(name, kind=otel_kind, start_time=None)
    if incident_id:
        span.set_attribute("incident.id", incident_id)
    if incident_no:
        span.set_attribute("incident.no", incident_no)
    if event_type:
        span.set_attribute("event.type", event_type)
    span.set_attribute("span.type", as_type)
    if input_data is not None and _CAPTURE_CONTENT:
        input_val, truncated = _truncate_content(input_data, _INPUT_MAX_CHARS)
        span.set_attribute("input", input_val)
        if truncated:
            span.set_attribute("input.truncated", True)
    if metadata:
        for key, value in metadata.items():
            span.set_attribute(f"meta.{key}", value)
    return tracer, span


def start_incident_span(incident_id: str, incident_no: str = None, event_type: str = "new_incident", phase: str = None):
    tracer = get_tracer(__name__)
    root_span = tracer.start_span(f"incident:{incident_id}", kind=SpanKind.INTERNAL)
    root_span.set_attribute("incident.id", incident_id)
    if incident_no:
        root_span.set_attribute("incident.no", incident_no)
    root_span.set_attribute("event.type", event_type)
    root_span.set_attribute("span.type", "agent")
    token = _current_span.set(root_span)
    if phase:
        child_span = tracer.start_span(f"phase:{phase}", kind=SpanKind.INTERNAL, context=trace.set_span_in_context(root_span))
        child_span.set_attribute("incident.id", incident_id)
        child_span.set_attribute("phase", phase)
        child_span.set_attribute("span.type", "phase")
        return tracer, root_span, child_span, token
    return tracer, root_span, token


def end_span(span: Any, output: Any = None, error: Exception = None, metadata: dict = None):
    if span is None:
        return
    try:
        if output is not None and _CAPTURE_CONTENT:
            output_val, truncated = _truncate_content(output, _OUTPUT_MAX_CHARS)
            span.set_attribute("output", output_val)
            if truncated:
                span.set_attribute("output.truncated", True)
                span.set_attribute("output.length", len(str(output)))
        if error:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(error))
            span.record_exception(error)
        if metadata:
            for key, value in metadata.items():
                span.set_attribute(f"meta.{key}", value)
        span.end()
    except Exception:
        pass


__all__ = [
    "init_telemetry", "get_tracer", "get_provider", "is_tracing_enabled", "shutdown_telemetry",
    "active_span_scope", "get_current_span", "set_span_attribute", "record_span_event",
    "record_span_exception", "create_span", "start_incident_span", "end_span",
]
