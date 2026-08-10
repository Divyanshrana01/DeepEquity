from __future__ import annotations

from typing import Any

import pytest

import deepequity.ingestion.pipeline as pipeline
from deepequity.core.config import get_settings
from deepequity.ingestion.errors import PermanentIngestionError, TransientIngestionError
from deepequity.ingestion.models import Document, DocumentStatus, IngestionEvent


# A stand-in for the postgres-backed repository. Keeps one document in memory and records
# which status transitions happened, so tests can assert on the flow without a database.
class FakeRepository:
    def __init__(
        self, document: Document, claimable: bool = True, stuck: bool = False
    ) -> None:
        self.document = document
        self.claimable = claimable
        #stuck means the row is still marked processing, ie a worker died holding it
        self.stuck = stuck
        self.completed: list[int] = []
        self.failed: list[tuple[int, str]] = []
        self.claim_calls = 0
        self.last_allow_stuck = False

    async def mark_processing(self, document_id: int, allow_stuck: bool = False) -> bool:
        self.claim_calls += 1
        self.last_allow_stuck = allow_stuck
        #a stuck (still "processing") row is only claimable when we've reclaimed it
        if self.stuck:
            return allow_stuck
        return self.claimable

    async def get(self, document_id: int) -> Document | None:
        return self.document if self.document.id == document_id else None

    async def mark_complete(self, document_id: int) -> None:
        self.completed.append(document_id)

    async def mark_failed(self, document_id: int, error: str) -> None:
        self.failed.append((document_id, error))
        self.document = self.document.model_copy(update={"last_error": error})


# Collects anything sent to the dead letter queue so tests can check what got parked.
class FakeEvents:
    def __init__(self) -> None:
        self.acked: list[str] = []
        self.dead_lettered: list[tuple[str, int, str]] = []

    async def ack(self, message_id: str) -> None:
        self.acked.append(message_id)

    async def send_to_dlq(self, message_id: str, event: IngestionEvent, error: str) -> None:
        self.dead_lettered.append((message_id, event.document_id, error))


def _document(**overrides: Any) -> Document:
    base: dict[str, Any] = {
        "id": 1,
        "ticker": "AAPL",
        "doc_type": "10-K",
        "source_ref": "0000-24-1",
        "raw_text": "some filing text",
        "status": DocumentStatus.PENDING,
    }
    base.update(overrides)
    return Document(**base)


def _event(document_id: int = 1) -> IngestionEvent:
    return IngestionEvent(
        document_id=document_id, ticker="AAPL", doc_type="10-K", source_ref="0000-24-1"
    )


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real backoff sleeps up to 16 seconds, which would make this suite crawl. We keep the
    # retry counting real and only skip the waiting.
    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(pipeline.asyncio, "sleep", _no_sleep)
    get_settings.cache_clear()


# These tests are about the event plumbing (claiming, retrying, dead lettering), not
# about what processing does. The real process_document downloads a filing and runs an
# embedding model, so tests that don't specifically care about that swap it for a no-op.
@pytest.fixture
def stub_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(_document: Document) -> None:
        return None

    monkeypatch.setattr(pipeline, "process_document", _noop)


async def test_happy_path_marks_complete_and_acks(
    monkeypatch: pytest.MonkeyPatch, stub_processing: None
) -> None:
    repo = FakeRepository(_document())
    evts = FakeEvents()
    monkeypatch.setattr(pipeline, "repository", repo)
    monkeypatch.setattr(pipeline, "events", evts)

    await pipeline.consume_event("msg-1", _event())

    assert repo.completed == [1]
    assert evts.acked == ["msg-1"]
    assert evts.dead_lettered == []


async def test_duplicate_event_is_skipped(
    monkeypatch: pytest.MonkeyPatch, stub_processing: None
) -> None:
    # claimable=False simulates another worker already having taken this document, the
    # second idempotency check should stop us doing the work twice.
    repo = FakeRepository(_document(), claimable=False)
    evts = FakeEvents()
    monkeypatch.setattr(pipeline, "repository", repo)
    monkeypatch.setattr(pipeline, "events", evts)

    await pipeline.consume_event("msg-1", _event())

    assert repo.completed == []
    # Still acked: the message is handled, just not by us, leaving it pending would mean
    # redis redelivers it forever.
    assert evts.acked == ["msg-1"]


async def test_transient_failure_retries_then_dead_letters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = FakeRepository(_document())
    evts = FakeEvents()
    attempts = 0

    async def always_fails(_document: Document) -> None:
        nonlocal attempts
        attempts += 1
        raise TransientIngestionError("network wobble")

    monkeypatch.setattr(pipeline, "repository", repo)
    monkeypatch.setattr(pipeline, "events", evts)
    monkeypatch.setattr(pipeline, "process_document", always_fails)

    await pipeline.consume_event("msg-1", _event())

    assert attempts == get_settings().ingestion_max_attempts
    assert repo.completed == []
    assert len(evts.dead_lettered) == 1
    assert "transient" in evts.dead_lettered[0][2]


async def test_transient_failure_that_recovers_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = FakeRepository(_document())
    evts = FakeEvents()
    attempts = 0

    async def fails_once(_document: Document) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TransientIngestionError("first try fails")

    monkeypatch.setattr(pipeline, "repository", repo)
    monkeypatch.setattr(pipeline, "events", evts)
    monkeypatch.setattr(pipeline, "process_document", fails_once)

    await pipeline.consume_event("msg-1", _event())

    assert attempts == 2
    assert repo.completed == [1]
    assert evts.dead_lettered == []


async def test_permanent_failure_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = FakeRepository(_document())
    evts = FakeEvents()
    attempts = 0

    async def bad_data(_document: Document) -> None:
        nonlocal attempts
        attempts += 1
        raise PermanentIngestionError("malformed filing")

    monkeypatch.setattr(pipeline, "repository", repo)
    monkeypatch.setattr(pipeline, "events", evts)
    monkeypatch.setattr(pipeline, "process_document", bad_data)

    await pipeline.consume_event("msg-1", _event())

    # The whole point of the permanent/transient split: one attempt, not three.
    assert attempts == 1
    assert len(evts.dead_lettered) == 1
    assert "permanent" in evts.dead_lettered[0][2]


async def test_reclaimed_message_can_take_over_a_stuck_document(
    monkeypatch: pytest.MonkeyPatch, stub_processing: None
) -> None:
    # A worker died holding this document, so the row is still marked processing. A
    # normal delivery must not touch it, but a reclaimed one must be able to finish it,
    # otherwise the crash loses that document for good.
    repo = FakeRepository(_document(status=DocumentStatus.PROCESSING), stuck=True)
    evts = FakeEvents()
    monkeypatch.setattr(pipeline, "repository", repo)
    monkeypatch.setattr(pipeline, "events", evts)

    await pipeline.consume_event("msg-1", _event(), reclaimed=True)

    assert repo.last_allow_stuck is True
    assert repo.completed == [1]


async def test_stuck_document_is_left_alone_on_a_normal_delivery(
    monkeypatch: pytest.MonkeyPatch, stub_processing: None
) -> None:
    repo = FakeRepository(_document(status=DocumentStatus.PROCESSING), stuck=True)
    evts = FakeEvents()
    monkeypatch.setattr(pipeline, "repository", repo)
    monkeypatch.setattr(pipeline, "events", evts)

    await pipeline.consume_event("msg-1", _event())

    # Someone may genuinely be working on it, so we skip rather than double process.
    assert repo.completed == []


async def test_document_with_no_source_url_is_permanent_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nothing to download and no way to get a url, so retrying would just fail the same
    # way three times. Straight to the dead letter queue.
    repo = FakeRepository(_document(source_url=None))
    evts = FakeEvents()
    monkeypatch.setattr(pipeline, "repository", repo)
    monkeypatch.setattr(pipeline, "events", evts)

    await pipeline.consume_event("msg-1", _event())

    assert repo.completed == []
    assert len(evts.dead_lettered) == 1
    assert "permanent" in evts.dead_lettered[0][2]
