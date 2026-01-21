import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.propagate import set_global_textmap
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Module-level state
_tracer_initialized = False
_logger = logging.getLogger(__name__)


def is_tracing_enabled() -> bool:
    """Check if telemetry is enabled via environment variable."""
    return os.environ.get('TELEMETRY_SERVICE_NAME') is not None


def is_local_mode() -> bool:
    """Check if running in local/dev mode (console export)."""
    return os.environ.get('TELEMETRY_LOCAL', 'false').lower() == 'true'


def init_tracer():
    """Initialize OpenTelemetry tracer. Call BEFORE creating FastAPI app."""
    global _tracer_initialized

    if _tracer_initialized:
        _logger.warning("Tracer already initialized, skipping...")
        return

    set_global_textmap(TraceContextTextMapPropagator())
    _logger.info("W3C Trace Context propagation enabled")

    service_name = os.environ.get('TELEMETRY_SERVICE_NAME', 'sop-agent')
    environment = os.environ.get('ENVIRONMENT', 'local')

    resource = Resource.create({
        SERVICE_NAME: service_name,
        "service.namespace": "ai-platform",
        "deployment.environment": environment,
    })

    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    if is_local_mode():
        _init_local_trace_exporter(tracer_provider, service_name)
    else:
        _init_otlp_trace_exporter(tracer_provider, service_name)

    _instrument_libraries()
    _tracer_initialized = True


def _init_local_trace_exporter(tracer_provider: TracerProvider, service_name: str):
    """Initialize console exporter for local development (traces)."""
    console_exporter = ConsoleSpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(console_exporter))
    _logger.info("=" * 60)
    _logger.info("OpenTelemetry TRACES initialized in LOCAL mode")
    _logger.info(f"Service: {service_name}")
    _logger.info("Traces will be printed to console")
    _logger.info("=" * 60)


def _init_otlp_trace_exporter(tracer_provider: TracerProvider, service_name: str):
    """Initialize OTLP HTTP exporter for production (traces)."""
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        telemetry_endpoint = os.environ.get('TELEMETRY_ENDPOINT', 'localhost')
        telemetry_port = os.environ.get('TELEMETRY_PORT', '4318')
        endpoint_url = f"http://{telemetry_endpoint}:{telemetry_port}/v1/traces"

        otlp_exporter = OTLPSpanExporter(endpoint=endpoint_url, timeout=10)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                otlp_exporter,
                max_queue_size=2048,
                max_export_batch_size=512,
                export_timeout_millis=30000,
            )
        )

        _logger.info("=" * 60)
        _logger.info("OpenTelemetry TRACES initialized (OTLP/HTTP)")
        _logger.info(f"Service: {service_name}")
        _logger.info(f"Exporting to: {endpoint_url}")
        _logger.info("=" * 60)

    except Exception as e:
        _logger.error(f"Failed to initialize OTLP trace exporter: {e}")
        _logger.info("Falling back to console exporter for traces")
        _init_local_trace_exporter(tracer_provider, service_name)


def _instrument_libraries():
    """Auto-instrument common libraries."""
    instrumented = []
    
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
        instrumented.append("HTTPX")
    except ImportError:
        _logger.debug("HTTPX instrumentation not available")
    except Exception as e:
        _logger.warning(f"Failed to instrument HTTPX: {e}")
    
    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        LoggingInstrumentor().instrument(set_logging_format=False)
        instrumented.append("Logging")
    except ImportError:
        _logger.debug("Logging instrumentation not available")
    except Exception as e:
        _logger.warning(f"Failed to instrument Logging: {e}")
    
    if instrumented:
        _logger.info(f"✓ Auto-instrumented: {', '.join(instrumented)}")


def attach_opentelemetry(app):
    """
    Attach OpenTelemetry to FastAPI app.
    Call AFTER creating FastAPI app.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        
        # Instrument the app
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="/health,/static/*",  # Don't trace health checks or static files
        )
        _logger.info("✓ FastAPI instrumented with OpenTelemetry")
        
    except ImportError as e:
        _logger.error(f"✗ FastAPI instrumentation not available: {e}")
        _logger.error("  Install with: pip install opentelemetry-instrumentation-fastapi")
    except Exception as e:
        _logger.error(f"✗ Failed to instrument FastAPI: {e}")


def get_tracer(name: str = __name__):
    """Get a tracer instance for manual span creation."""
    return trace.get_tracer(name)


def shutdown_telemetry():
    """Gracefully shutdown telemetry."""
    try:
        provider = trace.get_tracer_provider()
        if hasattr(provider, 'shutdown'):
            provider.shutdown()
            _logger.info("✓ OpenTelemetry shutdown complete")
    except Exception as e:
        _logger.warning(f"✗ Error during OpenTelemetry shutdown: {e}")


# Backwards compatibility aliases
init_telemetry = init_tracer
instrument_fastapi = attach_opentelemetry
