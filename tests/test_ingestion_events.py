from __future__ import annotations

import fakeredis.aioredis
import pytest

import deepequity.ingestion.events as events
from deepequity.core.config import get_settings
from deepequity.ingestion.models import IngestionEvent


@pytest.fixture
def redis(monkeypatch: pytest.MonkeyPatch) -> fakeredis.aioredis.FakeRedis:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(events, "get_redis", lambda: fake)
    return fake


def _event(document_id: int = 1) -> IngestionEvent:
    return IngestionEvent(
        document_id=document_id, ticker="AAPL", doc_type="10-K", source_ref="0000-24-1"
    )


async def test_publish_then_read_round_trips(redis: fakeredis.aioredis.FakeRedis) -> None:
    await events.ensure_group()
    await events.publish(_event(42))

    batch = await events.read_batch("worker-1", block_ms=10)

    assert len(batch) == 1
    _message_id, received = batch[0]
    assert received.document_id == 42
    assert received.ticker == "AAPL"


async def test_ensure_group_is_safe_to_call_twice(redis: fakeredis.aioredis.FakeRedis) -> None:
    # Every worker/api restart calls this, so a second call must not blow up on BUSYGROUP.
    await events.ensure_group()
    await events.ensure_group()


async def test_acked_message_is_not_redelivered(redis: fakeredis.aioredis.FakeRedis) -> None:
    await events.ensure_group()
    await events.publish(_event())

    batch = await events.read_batch("worker-1", block_ms=10)
    await events.ack(batch[0][0])

    # ">" only returns messages never handed out before, so a second read sees nothing.
    assert await events.read_batch("worker-1", block_ms=10) == []


async def test_bad_message_is_quarantined_without_losing_the_batch(
    redis: fakeredis.aioredis.FakeRedis,
) -> None:
    # Regression guard. A malformed event used to blow up the whole read, which left the
    # healthy messages next to it marked delivered but never processed and never dead
    # lettered, so they silently disappeared. The bad one must be quarantined alone.
    await events.ensure_group()
    await redis.xadd(events.STREAM_KEY, {"document_id": "not-a-number", "ticker": "X"})
    await events.publish(_event(123))

    batch = await events.read_batch("worker-1", block_ms=10)

    # The good message still comes through.
    assert [event.document_id for _mid, event in batch] == [123]

    # The bad one is in the DLQ with the reason attached.
    dlq = await redis.xrange(events.DLQ_KEY)
    assert len(dlq) == 1
    assert "undecodable event" in dlq[0][1]["error"]

    # And it's acked, so it isn't left hanging in the pending list.
    pending = await redis.xpending(events.STREAM_KEY, events.GROUP_NAME)
    assert pending["pending"] == 1  # only the good message, which we haven't acked yet


async def test_reclaim_picks_up_a_dead_workers_message(
    redis: fakeredis.aioredis.FakeRedis,
) -> None:
    # Simulates a worker taking a message and then dying before acking it. Another
    # worker must be able to take it over, otherwise that document is stranded forever.
    await events.ensure_group()
    await events.publish(_event(55))
    delivered = await events.read_batch("worker-that-dies", block_ms=10)
    assert len(delivered) == 1  # it's now pending against the dead worker

    # min_idle_ms=0 so we don't have to wait out the real idle window in a test.
    reclaimed = await events.reclaim_stale("worker-that-lives", min_idle_ms=0)

    assert [event.document_id for _mid, event in reclaimed] == [55]


async def test_reclaim_returns_nothing_when_all_work_is_acked(
    redis: fakeredis.aioredis.FakeRedis,
) -> None:
    await events.ensure_group()
    await events.publish(_event(56))
    message_id, _event_obj = (await events.read_batch("worker-1", block_ms=10))[0]
    await events.ack(message_id)

    assert await events.reclaim_stale("worker-2", min_idle_ms=0) == []


def test_block_duration_stays_under_the_socket_timeout() -> None:
    # Regression guard. redis-py gives each connection a 5s socket read timeout by
    # default, and our blocking stream read was also 5s, so the socket timed out at the
    # exact moment the read was still legitimately waiting and the worker died on its
    # first idle loop. These two values must never meet again.
    settings = get_settings()

    assert settings.ingestion_block_ms / 1000 < settings.redis_socket_timeout


async def test_dlq_stores_error_and_acks_original(redis: fakeredis.aioredis.FakeRedis) -> None:
    await events.ensure_group()
    await events.publish(_event(7))
    message_id, event = (await events.read_batch("worker-1", block_ms=10))[0]

    await events.send_to_dlq(message_id, event, "boom")

    dlq = await redis.xrange(events.DLQ_KEY)
    assert len(dlq) == 1
    _dlq_id, fields = dlq[0]
    assert fields["document_id"] == "7"
    assert fields["error"] == "boom"
    assert fields["original_message_id"] == message_id

    # The original must be acked, otherwise it sits pending and gets retried forever.
    pending = await redis.xpending(events.STREAM_KEY, events.GROUP_NAME)
    assert pending["pending"] == 0
