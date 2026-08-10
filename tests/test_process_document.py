from __future__ import annotations

from typing import Any

import pytest

import deepequity.ingestion.pipeline as pipeline
from deepequity.ingestion.errors import PermanentIngestionError
from deepequity.ingestion.models import Document, DocumentStatus

_HTML = """
<html><body>
<p>Item 1A. Risk Factors. The Company faces intense competition.</p>
<p>Item 7. Management's discussion. Total net sales increased year over year.</p>
</body></html>
"""


def _document(**overrides: Any) -> Document:
    base: dict[str, Any] = {
        "id": 1,
        "ticker": "AAPL",
        "doc_type": "10-K",
        "source_ref": "0000-24-1",
        "source_url": "https://sec.gov/doc.htm",
        "raw_text": "only the first 20k chars, not what we process",
        "status": DocumentStatus.PROCESSING,
    }
    base.update(overrides)
    return Document(**base)


# Captures what would have been written, so we can assert on the shape of the result
# without needing a live postgres with pgvector.
class FakeChunkStore:
    def __init__(self) -> None:
        self.calls: list[tuple[int, Any, Any]] = []

    async def __call__(
        self, document_id: int, parents: Any, vectors: Any
    ) -> tuple[int, int]:
        self.calls.append((document_id, parents, vectors))
        children = sum(len(parent.children) for parent in parents)
        return len(parents), children


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> FakeChunkStore:
    async def fake_fetch(url: str) -> str:
        return _HTML

    async def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    store = FakeChunkStore()
    monkeypatch.setattr(pipeline, "fetch_full_document", fake_fetch)
    monkeypatch.setattr(pipeline, "embed_texts", fake_embed)
    monkeypatch.setattr(pipeline, "replace_document_chunks", store)
    return store


async def test_processing_fetches_the_full_document_not_the_stored_excerpt(
    monkeypatch: pytest.MonkeyPatch, wired: FakeChunkStore
) -> None:
    # The stored raw_text is only the first 20k characters the MCP tool returned. If we
    # chunked that instead of refetching, we'd index the cover page and miss the risk
    # factors and MD&A, which is the entire point of the corpus.
    requested: list[str] = []

    async def recording_fetch(url: str) -> str:
        requested.append(url)
        return _HTML

    monkeypatch.setattr(pipeline, "fetch_full_document", recording_fetch)

    await pipeline.process_document(_document())

    assert requested == ["https://sec.gov/doc.htm"]
    stored_text = " ".join(
        child.text for _doc_id, parents, _v in wired.calls
        for parent in parents for child in parent.children
    )
    assert "Risk Factors" in stored_text
    assert "only the first 20k chars" not in stored_text


async def test_one_vector_per_child_chunk(wired: FakeChunkStore) -> None:
    # If these ever drift apart, chunks silently get the wrong neighbour's embedding and
    # search returns confidently wrong passages.
    await pipeline.process_document(_document())

    _doc_id, parents, vectors = wired.calls[0]
    expected = sum(len(parent.children) for parent in parents)
    assert len(vectors) == expected


async def test_missing_source_url_is_permanent(wired: FakeChunkStore) -> None:
    with pytest.raises(PermanentIngestionError, match="no source url"):
        await pipeline.process_document(_document(source_url=None))


async def test_html_that_cleans_to_nothing_is_permanent(
    monkeypatch: pytest.MonkeyPatch, wired: FakeChunkStore
) -> None:
    # A filing that reduces to no text is broken, not a temporary blip, so retrying it
    # three times would just waste time before failing anyway.
    async def empty_html(url: str) -> str:
        return "<html><body><script>var x=1;</script></body></html>"

    monkeypatch.setattr(pipeline, "fetch_full_document", empty_html)

    with pytest.raises(PermanentIngestionError, match="no text"):
        await pipeline.process_document(_document())
