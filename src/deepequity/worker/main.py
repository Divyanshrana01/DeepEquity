"""Worker process entrypoint.

Empty on purpose for now, this just proves the worker container starts,
can reach Redis, and logs properly. Phase 2 replaces the loop body with
real event consumption for the ingestion pipeline (idempotency check,
parse, chunk, embed, store).
"""

import asyncio

from deepequity.core.config import get_settings
from deepequity.core.logging import configure_logging, get_logger
from deepequity.core.redis_client import close_redis_pool, get_redis

HEARTBEAT_SECONDS = 30


async def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("deepequity.worker")

    logger.info("worker_starting")
    redis = get_redis()

    try:
        while True:
            await redis.ping()
            logger.info("worker_heartbeat")
            await asyncio.sleep(HEARTBEAT_SECONDS)
    finally:
        await close_redis_pool()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
