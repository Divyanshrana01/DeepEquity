import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger("deepequity.request")


#tags every request with an id and binds it to structlog's context, so every log line
#for this request carries it. matters once things break in prod: a user reports "my
#request failed", gives you the x-request-id from their error, and you grep for that
#exact string instead of guessing which of a thousand log lines belongs to them
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        start = time.perf_counter()

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id, path=request.url.path, method=request.method
        )

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        await logger.ainfo(
            "request_completed",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        return response
