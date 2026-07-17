"""Sliding-window rate limiter backed by Redis.

Why sliding window instead of a simple fixed window counter: a fixed window
lets someone burst 2x the limit right across a window boundary (e.g. 60
requests in the last second of one minute, then another 60 in the first
second of the next). A sliding window, storing a timestamp per request in a
sorted set and counting how many fall inside the trailing window, doesn't
have that gap.

Identifies callers by their JWT subject when they're authenticated, falls
back to IP address for anonymous requests (health checks etc, though those
are excluded from limiting entirely, see main.py).
"""

import time
import uuid

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from deepequity.core.config import get_settings
from deepequity.core.redis_client import get_redis


def _identify_caller(request: Request) -> str:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        # Hash-free identifier is fine here, the token itself never gets
        # logged or stored anywhere, just used as a Redis key suffix.
        return f"token:{auth_header[7:]}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


EXEMPT_PATHS = {"/health", "/ready"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        settings = get_settings()
        identifier = _identify_caller(request)
        redis = get_redis()
        key = f"ratelimit:{identifier}"

        now = time.time()
        window_start = now - settings.rate_limit_window_seconds

        async with redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {str(uuid.uuid4()): now})
            pipe.zcard(key)
            pipe.expire(key, settings.rate_limit_window_seconds)
            _, _, request_count, _ = await pipe.execute()

        if request_count > settings.rate_limit_requests:
            return Response(
                content="Rate limit exceeded, slow down.",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(settings.rate_limit_window_seconds)},
            )

        return await call_next(request)
