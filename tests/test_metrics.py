from __future__ import annotations

import math

from deepequity.evaluation.metrics import (
    aggregate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
)


def test_precision_counts_only_what_was_returned() -> None:
    # 2 correct out of the 4 shown
    assert precision_at_k([True, False, True, False], k=4) == 0.5


def test_precision_respects_k() -> None:
    # only the first 2 are considered, both correct
    assert precision_at_k([True, True, False, False], k=2) == 1.0


def test_precision_of_empty_results_is_zero_not_an_error() -> None:
    assert precision_at_k([], k=5) == 0.0


def test_recall_is_against_all_relevant_not_just_returned() -> None:
    # found 2, but 10 exist in the corpus
    assert recall_at_k([True, True, False], k=3, total_relevant=10) == 0.2


def test_recall_is_capped_by_k() -> None:
    # This is the trap in reading recall@k. 20 passages are relevant but we only return
    # 5, so even a perfect search tops out at 0.25. The number is meaningless without
    # knowing the total, which is why we report it alongside.
    perfect_top_5 = [True] * 5
    assert recall_at_k(perfect_top_5, k=5, total_relevant=20) == 0.25


def test_recall_with_no_relevant_documents_is_zero() -> None:
    # A query nothing in the corpus answers must not divide by zero.
    assert recall_at_k([False], k=1, total_relevant=0) == 0.0


def test_reciprocal_rank_rewards_an_early_hit() -> None:
    assert reciprocal_rank([True, False, False]) == 1.0
    assert reciprocal_rank([False, True, False]) == 0.5
    assert reciprocal_rank([False, False, True]) == 1 / 3


def test_reciprocal_rank_is_zero_when_nothing_was_found() -> None:
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg_is_one_for_perfect_ordering() -> None:
    # every relevant doc first, and exactly as many exist as we returned
    assert ndcg_at_k([True, True, True], k=3, total_relevant=3) == 1.0


def test_ndcg_punishes_burying_the_right_answer() -> None:
    # Same number of correct results, different positions. This is the difference
    # between nDCG and precision, precision would call these identical.
    early = ndcg_at_k([True, False, False], k=3, total_relevant=1)
    late = ndcg_at_k([False, False, True], k=3, total_relevant=1)

    assert early > late
    assert early == 1.0


def test_ndcg_matches_hand_computed_value() -> None:
    # one hit at position 2: dcg = 1/log2(3), ideal = 1/log2(2) = 1
    expected = (1 / math.log2(3)) / 1.0
    assert ndcg_at_k([False, True], k=2, total_relevant=1) == expected


def test_score_query_bundles_everything() -> None:
    metrics = score_query("q1", [False, True, True], k=3, total_relevant=4)

    assert metrics.query_id == "q1"
    assert metrics.precision_at_k == 2 / 3
    assert metrics.recall_at_k == 0.5
    assert metrics.mrr == 0.5
    assert metrics.total_relevant == 4
    assert metrics.returned == 3


def test_aggregate_averages_each_query_equally() -> None:
    # A plain mean, so a query with a hundred right answers doesn't outweigh one with
    # two. Every question counts the same.
    perfect = score_query("a", [True], k=1, total_relevant=1)
    useless = score_query("b", [False], k=1, total_relevant=1)

    averaged = aggregate([perfect, useless])

    assert averaged["precision"] == 0.5
    assert averaged["mrr"] == 0.5


def test_aggregate_of_nothing_returns_zeros() -> None:
    assert aggregate([]) == {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
