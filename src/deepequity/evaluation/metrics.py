from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class QueryMetrics:
    query_id: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    #how many passages in the whole corpus actually answer this question. worth keeping
    #because a recall of 0.2 means something very different when there are 5 right
    #answers versus 100.
    total_relevant: int
    returned: int


#of the results we showed, what fraction were actually right. this is the "did we waste
#the reader's time" number.
def precision_at_k(relevance: list[bool], k: int) -> float:
    top = relevance[:k]
    if not top:
        return 0.0
    return sum(top) / len(top)


#of all the right answers that exist, what fraction did we surface. this is the "what
#did we miss" number.
#
#note it's capped by k: if 100 passages are relevant and we only return 5, recall can
#never exceed 0.05 no matter how good the search is. that's why recall@k is read
#alongside the total, not on its own.
def recall_at_k(relevance: list[bool], k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(relevance[:k]) / total_relevant


#how far down the list the first correct answer sat. 1.0 means the very first result was
#right, 0.5 means the second, and so on. this is the one that best matches how it feels
#to use a search box, because nobody reads past the first couple of hits.
def reciprocal_rank(relevance: list[bool]) -> float:
    for position, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return 1.0 / position
    return 0.0


#like precision but it cares where the right answers sat, not just how many there were.
#a correct result at position 1 counts for more than the same result at position 5.
#normalised against the best possible ordering, so 1.0 means perfectly ranked.
def ndcg_at_k(relevance: list[bool], k: int, total_relevant: int) -> float:
    top = relevance[:k]
    if not top:
        return 0.0

    #discounted cumulative gain: each hit is worth 1/log2(position+1)
    dcg = sum(1.0 / math.log2(position + 1) for position, hit in enumerate(top, 1) if hit)

    #the ideal ordering puts every relevant result first, limited by how many exist
    ideal_hits = min(total_relevant, k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))

    return dcg / idcg if idcg else 0.0


#scores one query's results. relevance is the judgement for each returned passage, in
#the order they were returned.
def score_query(
    query_id: str, relevance: list[bool], k: int, total_relevant: int
) -> QueryMetrics:
    return QueryMetrics(
        query_id=query_id,
        precision_at_k=precision_at_k(relevance, k),
        recall_at_k=recall_at_k(relevance, k, total_relevant),
        mrr=reciprocal_rank(relevance),
        ndcg_at_k=ndcg_at_k(relevance, k, total_relevant),
        total_relevant=total_relevant,
        returned=len(relevance),
    )


#the best recall@k anyone could possibly score on this query set.
#
#worth computing because raw recall@k reads as a failure when it isn't. if a question has
#24 relevant passages in the corpus and we only return 5, then even a flawless search
#tops out at about 0.21. comparing against this ceiling says how much of the achievable
#result we actually got, which is the number that means something.
def recall_ceiling(results: list[QueryMetrics], k: int = 5) -> float:
    if not results:
        return 0.0
    per_query = [
        min(k, r.total_relevant) / r.total_relevant if r.total_relevant else 0.0
        for r in results
    ]
    return sum(per_query) / len(per_query)


#averages each metric across the whole query set. plain mean, every question counts the
#same, so one query with a hundred right answers can't drown out the rest.
def aggregate(results: list[QueryMetrics]) -> dict[str, float]:
    if not results:
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    count = len(results)
    return {
        "precision": sum(r.precision_at_k for r in results) / count,
        "recall": sum(r.recall_at_k for r in results) / count,
        "mrr": sum(r.mrr for r in results) / count,
        "ndcg": sum(r.ndcg_at_k for r in results) / count,
    }
