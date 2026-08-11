from __future__ import annotations

import pytest

import deepequity.retrieval.search as search_module
from deepequity.core.config import get_settings
from deepequity.retrieval.models import RetrievedChunk


def _chunk(chunk_id: int, score: float, method: str) -> RetrievedChunk:
    return RetrievedChunk(
        child_chunk_id=chunk_id,
        document_id=1,
        ticker="AAPL",
        doc_type="10-K",
        child_text=f"passage {chunk_id}",
        score=score,
        retrieval_method=method,
    )


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Swaps out postgres and the reranker model so the orchestration can be tested
    without a database or a 80MB download."""
    calls: dict[str, list] = {"dense": [], "keyword": [], "rerank": []}

    async def fake_dense(query: str, limit: int, ticker: str | None = None):
        calls["dense"].append((query, limit, ticker))
        return [_chunk(1, 0.9, "dense"), _chunk(2, 0.8, "dense")]

    async def fake_keyword(query: str, limit: int, ticker: str | None = None):
        calls["keyword"].append((query, limit, ticker))
        return [_chunk(3, 0.4, "keyword"), _chunk(2, 0.3, "keyword")]

    async def fake_rerank(query: str, candidates: list[RetrievedChunk], top_k: int):
        calls["rerank"].append((query, len(candidates), top_k))
        return candidates[:top_k]

    async def fake_attach(chunks: list[RetrievedChunk]):
        return [
            chunk.model_copy(update={"parent_text": f"parent of {chunk.child_chunk_id}"})
            for chunk in chunks
        ]

    monkeypatch.setattr(search_module, "search_dense", fake_dense)
    monkeypatch.setattr(search_module, "search_keyword", fake_keyword)
    monkeypatch.setattr(search_module, "rerank", fake_rerank)
    monkeypatch.setattr(search_module, "_attach_parents", fake_attach)
    get_settings.cache_clear()
    return calls


async def test_both_retrievers_run_and_results_are_fused(wired: dict[str, list]) -> None:
    result = await search_module.hybrid_search("legal risks", top_k=3)

    assert len(wired["dense"]) == 1
    assert len(wired["keyword"]) == 1
    # chunks 1, 2, 3 across both lists, deduplicated to 3
    assert result.fused_candidates == 3
    assert result.dense_candidates == 2
    assert result.keyword_candidates == 2


async def test_chunk_found_by_both_ranks_first(wired: dict[str, list]) -> None:
    # Chunk 2 appears in both lists, so fusion should lift it above chunk 1 which only
    # dense found. This is the behaviour that makes hybrid worth the extra query.
    result = await search_module.hybrid_search("legal risks", top_k=3, use_reranker=False)

    assert result.chunks[0].child_chunk_id == 2
    assert result.chunks[0].retrieval_method == "dense+keyword"


async def test_reranker_sees_all_candidates_not_just_the_final_few(
    wired: dict[str, list],
) -> None:
    # If we truncated to top_k before reranking, the reranker could only reorder what
    # fusion already picked and could never rescue a good passage ranked 10th.
    await search_module.hybrid_search("legal risks", top_k=2)

    _query, candidate_count, top_k = wired["rerank"][0]
    assert candidate_count == 3  # all fused candidates
    assert top_k == 2


async def test_reranker_can_be_turned_off(wired: dict[str, list]) -> None:
    result = await search_module.hybrid_search("legal risks", use_reranker=False)

    assert wired["rerank"] == []
    assert result.reranked is False


async def test_ticker_filter_reaches_both_retrievers(wired: dict[str, list]) -> None:
    await search_module.hybrid_search("legal risks", ticker="AAPL")

    assert wired["dense"][0][2] == "AAPL"
    assert wired["keyword"][0][2] == "AAPL"


async def test_candidate_pool_is_wider_than_the_final_result(wired: dict[str, list]) -> None:
    # The reranker needs more to choose from than we intend to return, otherwise there's
    # nothing for it to improve on.
    settings = get_settings()
    await search_module.hybrid_search("legal risks", top_k=3)

    requested_limit = wired["dense"][0][1]
    assert requested_limit == settings.retrieval_candidates_per_method
    assert requested_limit > 3


async def test_parent_text_is_attached_to_results(wired: dict[str, list]) -> None:
    result = await search_module.hybrid_search("legal risks", top_k=2)

    assert all(chunk.parent_text for chunk in result.chunks)


async def test_top_k_is_respected(wired: dict[str, list]) -> None:
    result = await search_module.hybrid_search("legal risks", top_k=1)

    assert len(result.chunks) == 1
