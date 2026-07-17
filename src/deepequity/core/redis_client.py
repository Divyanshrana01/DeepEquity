"""One shared async Redis connection pool for the whole app.

FastAPI creates this once at startup and reuses it for every request, rather
than opening a new connection every time someone hits an endpoint.
"""

from redis.asyncio import ConnectionPool, Redis

from deepequity.core.config import get_settings

_pool: ConnectionPool | None = None


def get_redis_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool.from_url(get_settings().redis_url, decode_responses=True)
    return _pool


def get_redis() -> Redis:
    return Redis(connection_pool=get_redis_pool())


async def close_redis_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.disconnect()
        _pool = None
