from __future__ import annotations

from typing import Any, cast

from redis.exceptions import ResponseError

from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger
from deepequity.core.redis_client import get_redis
from deepequity.ingestion.models import IngestionEvent

logger = get_logger("deepequity.ingestion.events")

#redis-py's type hints say every value could be bytes or str, because whether you get
#one or the other depends on the decode_responses flag it can't see at type-check time.
#we always create clients with decode_responses=True, so the casts below just tell mypy
#what we already know: these are strings.

#The main work queue and the graveyard for messages we gave up on.
STREAM_KEY = "ingestion:events"
DLQ_KEY = "ingestion:dlq"
#A consumer group lets several workers share one stream without doubling up, redis hands
#each message to exactly one member and tracks what hasn't been acked yet.
GROUP_NAME = "ingestion-workers"


#Creates the consumer group if it isn't there. mkstream builds the stream too, so a
#brand new redis works without anyone having published first. The BUSYGROUP error just
#means it already exists, which is the normal case on every restart after the first.
async def ensure_group() -> None:
    redis = get_redis()
    try:
        await redis.xgroup_create(STREAM_KEY, GROUP_NAME, id="0", mkstream=True)
        logger.info("consumer_group_created", stream=STREAM_KEY, group=GROUP_NAME)
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


#Drops an event on the stream for a worker to pick up. Returns the redis message id.
async def publish(event: IngestionEvent) -> str:
    redis = get_redis()
    message_id = cast(str, await redis.xadd(STREAM_KEY, cast(Any, _encode(event))))
    logger.info("event_published", document_id=event.document_id, message_id=message_id)
    return message_id


#Redis stream fields have to be flat strings, so ints get stringified on the way out and
#parsed back on the way in.
def _encode(event: IngestionEvent) -> dict[str, str]:
    return {
        "document_id": str(event.document_id),
        "ticker": event.ticker,
        "doc_type": event.doc_type,
        "source_ref": event.source_ref,
    }


def _decode(fields: dict[str, Any]) -> IngestionEvent:
    return IngestionEvent(
        document_id=int(fields["document_id"]),
        ticker=fields["ticker"],
        doc_type=fields["doc_type"],
        source_ref=fields["source_ref"],
    )


#Waits for the next batch of events for this worker. block_ms controls how long redis
#holds the connection open when there's nothing to do, which beats us hammering it in a
#tight polling loop. Returns pairs of (message_id, event).
async def read_batch(
    consumer_name: str, count: int = 10, block_ms: int | None = None
) -> list[tuple[str, IngestionEvent]]:
    #default comes from settings so the block duration and the redis socket timeout are
    #configured next to each other and can't quietly drift into conflict
    if block_ms is None:
        block_ms = get_settings().ingestion_block_ms
    redis = get_redis()
    #Shape is [(stream_key, [(message_id, fields), ...])]. The stub types this as a union
    #covering RESP2 and RESP3, so we pin it to the one shape we actually get.
    response = cast(
        list[tuple[str, list[tuple[str, dict[str, Any]]]]],
        await redis.xreadgroup(
            GROUP_NAME, consumer_name, {STREAM_KEY: ">"}, count=count, block=block_ms
        ),
    )
    if not response:
        return []

    #We only ever subscribe to one stream, so take that first entry's message list.
    _stream_key, messages = response[0]

    #Decode one message at a time. If we did the whole batch in one go, a single garbled
    #message would blow up the entire read and take its healthy neighbours down with it,
    #they'd be marked delivered but never processed and never dead lettered, so they'd
    #just quietly vanish. Instead a bad message gets quarantined on its own and the rest
    #of the batch carries on.
    decoded: list[tuple[str, IngestionEvent]] = []
    for message_id, fields in messages:
        try:
            decoded.append((message_id, _decode(fields)))
        except Exception as exc:  # noqa: BLE001 - anything unparseable is a poison message
            await send_raw_to_dlq(message_id, fields, f"undecodable event: {exc}")
    return decoded


#Picks up messages that were handed to some worker and never acked, which is what you're
#left with when a worker dies mid-document. Redis keeps them in a pending list rather
#than handing them to anyone else, so without this step that work is stranded forever.
#We only take messages that have been sitting untouched for min_idle_ms, so we don't
#snatch something another worker is busy with right now.
async def reclaim_stale(
    consumer_name: str, min_idle_ms: int | None = None, count: int = 10
) -> list[tuple[str, IngestionEvent]]:
    if min_idle_ms is None:
        min_idle_ms = get_settings().ingestion_reclaim_idle_ms
    redis = get_redis()

    #Returns (next_cursor, messages, deleted_ids). We start from 0 each time, so we
    #always look at the oldest stuck messages first.
    result = cast(
        tuple[str, list[tuple[str, dict[str, Any]]], list[str]],
        await redis.xautoclaim(
            STREAM_KEY, GROUP_NAME, consumer_name, min_idle_time=min_idle_ms, start_id="0-0",
            count=count,
        ),
    )
    _cursor, messages, _deleted = result

    reclaimed: list[tuple[str, IngestionEvent]] = []
    for message_id, fields in messages:
        try:
            reclaimed.append((message_id, _decode(fields)))
        except Exception as exc:  # noqa: BLE001 - same poison message handling as above
            await send_raw_to_dlq(message_id, fields, f"undecodable event: {exc}")

    if reclaimed:
        logger.info("messages_reclaimed", count=len(reclaimed), consumer=consumer_name)
    return reclaimed


#Tells redis this message is handled so it stops tracking it as outstanding. Without the
#ack, a message sits in the pending list forever and gets re-delivered.
async def ack(message_id: str) -> None:
    redis = get_redis()
    await redis.xack(STREAM_KEY, GROUP_NAME, message_id)


#Parks a message we've given up on in the dead letter queue along with why it died, then
#acks the original so it stops being retried. The DLQ is a plain stream you can read to
#see what broke, nothing consumes it automatically, that's a human's call.
async def send_to_dlq(message_id: str, event: IngestionEvent, error: str) -> None:
    await send_raw_to_dlq(message_id, _encode(event), error)


#Same thing for a message we couldn't even parse into an event. We keep whatever fields
#were on it so there's something to look at when working out where the bad publish came
#from, then ack it so it stops clogging the queue.
async def send_raw_to_dlq(message_id: str, fields: dict[str, Any], error: str) -> None:
    redis = get_redis()
    payload = {str(key): str(value) for key, value in fields.items()}
    payload["error"] = error[:2000]
    payload["original_message_id"] = message_id
    await redis.xadd(DLQ_KEY, cast(Any, payload))
    await ack(message_id)
    logger.warning("event_dead_lettered", message_id=message_id, error=error)
