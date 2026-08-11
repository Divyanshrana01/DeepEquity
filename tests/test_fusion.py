from __future__ import annotations

from deepequity.retrieval.fusion import reciprocal_rank_fusion
from deepequity.retrieval.models import RetrievedChunk


def _chunk(chunk_id: int, score: float, method: str) -> RetrievedChunk:
    return RetrievedChunk(
        child_chunk_id=chunk_id,
        document_id=1,
        ticker="AAPL",
        doc_type="10-K",
        child_text=f"chunk {chunk_id}",
        score=score,
        retrieval_method=method,
    )


def test_chunk_found_by_both_methods_beats_one_found_by_only_one() -> None:
    # The whole point of hybrid search. Chunk 2 is ranked second by both methods, chunk
    # 1 is ranked first by dense alone. Agreement across methods should win.
    dense = [_chunk(1, 0.9, "dense"), _chunk(2, 0.8, "dense")]
    keyword = [_chunk(3, 0.5, "keyword"), _chunk(2, 0.4, "keyword")]

    fused = reciprocal_rank_fusion([dense, keyword], k=60)

    assert fused[0].child_chunk_id == 2


def test_raw_scores_are_ignored_entirely() -> None:
    # Dense scores are cosine similarities (0.6-0.9), keyword scores are ts_rank values
    # (often under 0.1). If we summed the raw numbers, dense would win everything just
    # by having a bigger scale. Only rank position may matter.
    dense = [_chunk(1, 0.99, "dense")]
    keyword = [_chunk(2, 0.00001, "keyword")]

    fused = reciprocal_rank_fusion([dense, keyword], k=60)

    # Both are rank 1 in their own list, so they must tie despite wildly different scores
    assert fused[0].score == fused[1].score


def test_rrf_score_matches_the_formula() -> None:
    dense = [_chunk(1, 0.9, "dense"), _chunk(2, 0.8, "dense")]

    fused = reciprocal_rank_fusion([dense], k=60)

    # rank 1 -> 1/(60+1), rank 2 -> 1/(60+2)
    assert fused[0].score == 1 / 61
    assert fused[1].score == 1 / 62


def test_scores_from_both_lists_are_added() -> None:
    dense = [_chunk(7, 0.9, "dense")]
    keyword = [_chunk(7, 0.2, "keyword")]

    fused = reciprocal_rank_fusion([dense, keyword], k=60)

    assert len(fused) == 1
    assert fused[0].score == (1 / 61) * 2


def test_method_label_records_who_found_it() -> None:
    # Makes it obvious at a glance whether both halves contributed, which is the first
    # thing you check when hybrid results look no better than dense alone.
    dense = [_chunk(1, 0.9, "dense")]
    keyword = [_chunk(1, 0.2, "keyword"), _chunk(2, 0.1, "keyword")]

    fused = reciprocal_rank_fusion([dense, keyword], k=60)
    by_id = {chunk.child_chunk_id: chunk for chunk in fused}

    assert by_id[1].retrieval_method == "dense+keyword"
    assert by_id[2].retrieval_method == "keyword"


def test_results_come_back_sorted() -> None:
    dense = [_chunk(i, 0.5, "dense") for i in range(1, 6)]

    fused = reciprocal_rank_fusion([dense], k=60)

    scores = [chunk.score for chunk in fused]
    assert scores == sorted(scores, reverse=True)


def test_empty_and_partial_inputs_are_handled() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []

    # one method finding nothing must not break the other
    only_keyword = reciprocal_rank_fusion([[], [_chunk(1, 0.3, "keyword")]], k=60)
    assert len(only_keyword) == 1


def test_larger_k_flattens_the_gap_between_ranks() -> None:
    # k controls how much rank 1 outweighs rank 2. A bigger k narrows that gap, which
    # stops one over-confident method running away with the fused ranking.
    dense = [_chunk(1, 0.9, "dense"), _chunk(2, 0.8, "dense")]

    small_k = reciprocal_rank_fusion([dense], k=1)
    large_k = reciprocal_rank_fusion([dense], k=1000)

    small_gap = small_k[0].score - small_k[1].score
    large_gap = large_k[0].score - large_k[1].score
    assert large_gap < small_gap
