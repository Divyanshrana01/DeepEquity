"""FastAPI app entrypoint.

This is intentionally quiet right now: health/readiness checks, auth, rate
limiting, structured logging. No research endpoint yet, that shows up in
Phase 3 once the agents exist. The point of Phase 1 is that everything
around the eventual endpoint (auth, limits, observability) already works.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from deepequity.api.logging_middleware import RequestLoggingMiddleware
from deepequity.api.rate_limit import RateLimitMiddleware
from deepequity.api.routes.health import router as health_router
from deepequity.core.config import get_settings
from deepequity.core.logging import configure_logging
from deepequity.core.redis_client import close_redis_pool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    yield
    await close_redis_pool()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    # Middleware runs outside-in on the way in, so logging wraps rate
    # limiting: we want a request ID bound before the limiter logs anything.
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(health_router)

    return app


app = create_app()
