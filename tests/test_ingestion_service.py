from __future__ import annotations

from typing import Any

import pytest

import deepequity.ingestion.service as service
from deepequity.ingestion.errors import PermanentIngestionError
from deepequity.ingestion.models import Document, DocumentStatus, IngestionEvent

_FILING_OK: dict[str, Any] = {
    "status": "ok",
    "accession_number": "0000320193-24-000123",
    "document_url": "https://sec.gov/doc.htm",
    "text": "Item 1A. Risk Factors ...",
}


# Minimal repository stand-in: remembers whether a document already exists and what got
# inserted, which is all the service layer touches.
class FakeRepository:
    def __init__(self, existing: Document | None = None) -> None:
        self.existing = existing
        self.created: list[dict[str, Any]] = []
        self.create_returns_new = True

    async def find_by_source(
        self, ticker: str, doc_type: str, source_ref: str
    ) -> Document | None:
        return self.existing

    async def create_pending(self, **kwargs: Any) -> tuple[Document, bool]:
        self.created.append(kwargs)
        doc = Document(
            id=99,
            ticker=kwargs["ticker"],
            doc_type=kwargs["doc_type"],
            source_ref=kwargs["source_ref"],
            source_url=kwargs["source_url"],
            raw_text=kwargs["raw_text"],
            status=DocumentStatus.PENDING,
        )
        return doc, self.create_returns_new


class FakeEvents:
    def __init__(self) -> None:
        self.published: list[IngestionEvent] = []

    async def publish(self, event: IngestionEvent) -> str:
        self.published.append(event)
        return "msg-1"


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    repo: FakeRepository,
    evts: FakeEvents,
    filing: dict[str, Any],
) -> None:
    async def fake_fetch(ticker: str, form_type: str) -> dict[str, Any]:
        return filing

    monkeypatch.setattr(service, "repository", repo)
    monkeypatch.setattr(service, "events", evts)
    monkeypatch.setattr(service.mcp_client, "fetch_filing", fake_fetch)


async def test_new_filing_is_stored_and_published(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, evts = FakeRepository(), FakeEvents()
    _patch(monkeypatch, repo, evts, _FILING_OK)

    result = await service.request_ingestion("aapl", "10-k")

    assert result.document_id == 99
    assert result.already_ingested is False
    assert result.ticker == "AAPL"  # normalised to uppercase
    assert result.doc_type == "10-K"
    assert len(evts.published) == 1
    assert evts.published[0].document_id == 99


async def test_duplicate_filing_is_not_republished(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = Document(
        id=5,
        ticker="AAPL",
        doc_type="10-K",
        source_ref="0000320193-24-000123",
        status=DocumentStatus.COMPLETE,
    )
    repo, evts = FakeRepository(existing=existing), FakeEvents()
    _patch(monkeypatch, repo, evts, _FILING_OK)

    result = await service.request_ingestion("AAPL", "10-K")

    assert result.document_id == 5
    assert result.already_ingested is True
    # This is the first idempotency check doing its job: nothing new queued.
    assert evts.published == []
    assert repo.created == []


async def test_insert_race_does_not_double_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    # find_by_source says "new", but the insert hits the unique index because another
    # request got there first. We must not publish a second event for the same document.
    repo, evts = FakeRepository(), FakeEvents()
    repo.create_returns_new = False
    _patch(monkeypatch, repo, evts, _FILING_OK)

    result = await service.request_ingestion("AAPL", "10-K")

    assert result.already_ingested is True
    assert evts.published == []


async def test_failed_fetch_raises_permanent(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, evts = FakeRepository(), FakeEvents()
    _patch(monkeypatch, repo, evts, {"status": "error", "error": "Unknown ticker: NOPE"})

    with pytest.raises(PermanentIngestionError, match="Unknown ticker"):
        await service.request_ingestion("NOPE", "10-K")

    assert evts.published == []
