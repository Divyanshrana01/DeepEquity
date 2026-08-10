from redis.asyncio import ConnectionPool, Redis

from deepequity.core.config import get_settings

_pool: ConnectionPool | None = None


#builds the one shared connection pool the first time anything asks for it, then just
#hands back the same pool after that. avoids opening a fresh connection per request
def get_redis_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        settings = get_settings()
        #socket_timeout is set on purpose, not left to the library default. see the
        #comment on the setting: the default is short enough to kill the worker's
        #blocking stream read.
        _pool = ConnectionPool.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=settings.redis_socket_timeout,
        )
    return _pool


#gives you a redis client backed by the shared pool, this is what the rest of the app calls
def get_redis() -> Redis:
    return Redis(connection_pool=get_redis_pool())


#closes the pool on shutdown so we don't leave connections hanging around
async def close_redis_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.disconnect()
        _pool = None
