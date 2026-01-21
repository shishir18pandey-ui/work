import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.propagate import set_global_textmap
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Module-level state
_tracer_initialized = False
_logger = logging.getLogger(__name__)


def is_tracing_enabled() -> bool:
    """Check if telemetry is enabled via environment variable."""
    return os.environ.get('TELEMETRY_SERVICE_NAME') is not None


def init_tracer():
    """Initialize OpenTelemetry tracer. Call BEFORE creating FastAPI app."""
    global _tracer_initialized
    
    if _tracer_initialized:
        _logger.warning("Tracer already initialized, skipping...")
        return
    
    # Set W3C Trace Context propagation for distributed tracing
    set_global_textmap(TraceContextTextMapPropagator())
    
    # Get configuration from environment
    service_name = os.environ.get('TELEMETRY_SERVICE_NAME', 'sop-agent')
    environment = os.environ.get('ENVIRONMENT', 'local')
    telemetry_endpoint = os.environ.get('TELEMETRY_ENDPOINT', 'localhost')
    telemetry_port = os.environ.get('TELEMETRY_PORT', '4318')
    use_console_exporter = os.environ.get('TELEMETRY_CONSOLE', 'false').lower() == 'true'
    
    # Create resource (metadata about your service)
    resource = Resource.create({
        SERVICE_NAME: service_name,
        "service.namespace": "ai-platform",
        "deployment.environment": environment,
    })
    print(resource,"testing 1")
    
    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    
    # Try OTLP exporter first, fall back to console
    exporter_configured = False
    
    if use_console_exporter:
        # Use console exporter for debugging
        console_exporter = ConsoleSpanExporter()
        tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
        _logger.info("=" * 60)
        _logger.info("✓ OpenTelemetry TRACES initialized (CONSOLE MODE)")
        _logger.info(f"  Service: {service_name}")
        _logger.info(f"  Environment: {environment}")
        _logger.info("  Traces will be printed to console")
        _logger.info("=" * 60)
        exporter_configured = True
    else:
        # Try OTLP exporter
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            
            endpoint_url = f"http://{telemetry_endpoint}:{telemetry_port}/v1/traces"
            
            _logger.info(f"Attempting to connect to OTLP endpoint: {endpoint_url}")
            
            otlp_exporter = OTLPSpanExporter(
                endpoint=endpoint_url,
                timeout=10,
            )
            
            tracer_provider.add_span_processor(
                BatchSpanProcessor(
                    otlp_exporter,
                    max_queue_size=2048,
                    max_export_batch_size=512,
                    export_timeout_millis=30000,
                )
            )
            
            _logger.info("=" * 60)
            _logger.info("✓ OpenTelemetry TRACES initialized (OTLP MODE)")
            _logger.info(f"  Service: {service_name}")
            _logger.info(f"  Endpoint: {endpoint_url}")
            _logger.info(f"  Environment: {environment}")
            _logger.info("=" * 60)
            exporter_configured = True
            
        except ImportError as e:
            _logger.error(f"✗ OTLP exporter not installed: {e}")
            _logger.error("  Install with: pip install opentelemetry-exporter-otlp-proto-http")
            _logger.info("  Falling back to console exporter...")
            
            # Fallback to console exporter
            console_exporter = ConsoleSpanExporter()
            tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
            _logger.info("✓ Using console exporter as fallback")
            exporter_configured = True
            
        except Exception as e:
            _logger.error(f"✗ Failed to initialize OTLP trace exporter: {e}")
            _logger.info("  Falling back to console exporter...")
            
            # Fallback to console exporter
            try:
                console_exporter = ConsoleSpanExporter()
                tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
                _logger.info("✓ Using console exporter as fallback")
                exporter_configured = True
            except Exception as console_error:
                _logger.error(f"✗ Failed to initialize console exporter: {console_error}")
                return
    
    if not exporter_configured:
        _logger.error("✗ No trace exporter could be configured!")
        return
    
    # Auto-instrument libraries
    _instrument_libraries()
    _tracer_initialized = True


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
