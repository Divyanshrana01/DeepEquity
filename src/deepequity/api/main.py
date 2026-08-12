from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from deepequity.api.logging_middleware import RequestLoggingMiddleware
from deepequity.api.rate_limit import RateLimitMiddleware
from deepequity.api.routes.health import router as health_router
from deepequity.api.routes.ingest import router as ingest_router
from deepequity.api.routes.research import router as research_router
from deepequity.api.routes.search import router as search_router
from deepequity.api.routes.stats import router as stats_router
from deepequity.core.config import get_settings
from deepequity.core.db import close_pool, run_migrations
from deepequity.core.logging import configure_logging
from deepequity.core.redis_client import close_redis_pool
from deepequity.ingestion.events import ensure_group


#runs once at startup (logging, db tables, consumer group) and once at shutdown (close
#the redis and postgres pools cleanly)
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    await run_migrations()
    #the api creates the consumer group too, not just the worker, so publishing works
    #even if the api happens to boot first
    await ensure_group()
    yield
    await close_redis_pool()
    await close_pool()


#builds the app. no research endpoint yet, that shows up in phase 3 once the agents exist,
#right now this is health/readiness, auth, rate limiting, structured logging, and the
#ingestion endpoints that feed documents into the pipeline
def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    #middleware runs outside-in on the way in, so logging wraps rate limiting: we want
    #a request id bound before the limiter logs anything
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(search_router)
    app.include_router(research_router)
    app.include_router(stats_router)

    return app


app = create_app()
