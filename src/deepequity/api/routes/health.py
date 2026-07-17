import psycopg
from fastapi import APIRouter, Response, status

from deepequity.core.config import get_settings
from deepequity.core.redis_client import get_redis

router = APIRouter(tags=["health"])


#just says the process is alive, docker/kubernetes use this to decide whether to restart
#the container. doesn't check anything downstream on purpose
@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


#actually checks the things the app depends on (redis, postgres). traffic shouldn't get
#routed here until this returns 200, that's the difference from /health above
@router.get("/ready")
async def ready(response: Response) -> dict[str, str]:
    checks: dict[str, str] = {}

    try:
        redis = get_redis()
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001 - we want to report any failure, not just specific ones
        checks["redis"] = f"error: {exc}"

    try:
        async with await psycopg.AsyncConnection.connect(
            get_settings().postgres_dsn, connect_timeout=2
        ):
            checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001 - same reasoning as the redis check above
        checks["postgres"] = f"error: {exc}"

    if any(v != "ok" for v in checks.values()):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return checks
