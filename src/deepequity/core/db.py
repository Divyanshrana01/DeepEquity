from __future__ import annotations

from pathlib import Path

from psycopg_pool import AsyncConnectionPool

from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger

logger = get_logger("deepequity.db")

_pool: AsyncConnectionPool | None = None

#The schema file sits next to this package, we read it at startup and run it.
_SCHEMA_PATH = Path(__file__).parent.parent / "db" / "schema.sql"


#Builds the shared Postgres pool the first time anything asks for it, same idea as the
#redis pool. open=False then explicit open() because psycopg warns about opening a pool
#in the constructor, it wants you to do it deliberately.
async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(get_settings().postgres_dsn, open=False)
        await _pool.open()
    return _pool


#Runs schema.sql. It's all CREATE ... IF NOT EXISTS so calling this on every startup is
#safe, and it means a fresh database sets itself up without a separate migration step.
async def run_migrations() -> None:
    pool = await get_pool()
    schema = _SCHEMA_PATH.read_text()
    async with pool.connection() as conn:
        await conn.execute(schema)
    logger.info("migrations_applied")


#Closes the pool on shutdown so we don't leave connections open.
async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
