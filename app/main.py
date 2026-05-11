from contextlib import asynccontextmanager
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.redis import close_redis_pool, get_redis_pool
from app.db.session import engine
from app.routers import health, reports, admin, stream, auth, authority
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import os

logger=get_logger(__name__)

def setup_telemetry(settings) -> None:
    if not settings.otel_exporter_otlp_endpoint:
        return

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    provider=TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint))
    )
    trace.set_tracer_provider(provider)
    SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    settings = get_settings()
    setup_logging()
    setup_telemetry(settings)

    get_redis_pool()
    logger.info("application started", env=settings.app_env)

    yield

    await close_redis_pool()
    await engine.dispose()
    logger.info("application stopped")

def create_app() -> FastAPI:
    settings=get_settings()
    app=FastAPI(
        title="Civic Reporting Agent",
        version="0.1.0",
        docs_url="/docs" if settings.app_env=="development" else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    FastAPIInstrumentor.instrument_app(app)

    app.include_router(health.router)
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(reports.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")
    app.include_router(stream.router, prefix="/api/v1")
    app.include_router(authority.router, prefix="/api/v1")

    # Serve static files and root HTML
    static_path = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=static_path), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def read_root():
        return FileResponse(os.path.join(static_path, "civic_guardian.html"))

    return app

app=create_app()