import asyncio
import os
import socket

from deepequity.core.config import get_settings
from deepequity.core.db import close_pool, run_migrations
from deepequity.core.logging import configure_logging, get_logger
from deepequity.core.redis_client import close_redis_pool
from deepequity.ingestion import events, pipeline

#how long to wait before retrying after the stream read itself blows up
_READ_ERROR_BACKOFF_SECONDS = 2.0


#each worker needs a name redis can tell apart, otherwise two containers in the same
#consumer group look like one member. hostname plus pid is unique enough and, unlike a
#random uuid, stays the same across a restart so redis can hand back that worker's
#unfinished messages
def _consumer_name() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


#the main loop: pull a batch of ingestion events off the stream and process each one.
#read_batch blocks for a few seconds when the stream is empty, so an idle worker sits
#quietly rather than spinning
async def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("deepequity.worker")

    consumer = _consumer_name()
    logger.info("worker_starting", consumer=consumer)

    #the worker sets up its own tables and consumer group, so it works whether or not
    #the api container came up first
    await run_migrations()
    await events.ensure_group()

    try:
        while True:
            try:
                #before waiting for new work, pick up anything a dead worker left
                #half-done. this is what stops a crash mid-document losing that document.
                batch = await events.reclaim_stale(consumer)
                reclaimed = bool(batch)
                if not batch:
                    batch = await events.read_batch(consumer)
            except Exception:  # noqa: BLE001 - a blip talking to redis must not kill us
                #if redis hiccups we log it and try again in a moment. crashing here
                #would take the whole worker down over something that usually recovers
                #on its own.
                logger.exception("stream_read_failed")
                await asyncio.sleep(_READ_ERROR_BACKOFF_SECONDS)
                continue

            if not batch:
                continue

            logger.info("batch_received", count=len(batch), reclaimed=reclaimed)
            for message_id, event in batch:
                await pipeline.consume_event(message_id, event, reclaimed=reclaimed)
    finally:
        await close_redis_pool()
        await close_pool()


#sync entrypoint docker's CMD calls, just kicks off the async loop above
def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
