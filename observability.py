import os
import logging
import traceback

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_NAMESPACE, DEPLOYMENT_ENVIRONMENT
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)

_tracer = None
_tracing_enabled = False
_provider = None


def init_telemetry():
    try:
        global _tracer, _tracing_enabled, _provider

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
        })

        endpoint_url = f"http://{telemetry_endpoint}:{telemetry_port}/v1/traces"
        otlp_exporter = OTLPSpanExporter(
            endpoint=endpoint_url,
            timeout=10
        )

        _provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(_provider)

        _provider.add_span_processor(
            BatchSpanProcessor(
                otlp_exporter,
                max_queue_size=2048,
                max_export_batch_size=512,
                export_timeout_millis=30000,
            )
        )

        _tracer = trace.get_tracer(service_name)
        _tracing_enabled = True

        _instrument_libraries()

        logger.info(f"OpenTelemetry initialized successfully")
        logger.info(f"  Service Name : {service_name}")
        logger.info(f"  Environment  : {environment}")
        logger.info(f"  Endpoint     : {endpoint_url}")

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
