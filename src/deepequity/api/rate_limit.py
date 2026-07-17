import time
import uuid

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from deepequity.core.config import get_settings
from deepequity.core.redis_client import get_redis


#figures out who's calling so we can rate limit per-caller instead of globally. logged-in
#callers get limited by their token, anonymous ones by ip. fine to use the raw token here
#since it never gets logged or stored anywhere, just used as a redis key suffix
def _identify_caller(request: Request) -> str:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return f"token:{auth_header[7:]}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


EXEMPT_PATHS = {"/health", "/ready"}


#sliding-window limiter backed by redis. a plain fixed-window counter lets someone burst
#2x the limit right across a window boundary (60 requests in the last second of one minute,
#then another 60 in the first second of the next). this stores a timestamp per request in a
#sorted set and counts how many fall inside the trailing window, so there's no gap to abuse
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
