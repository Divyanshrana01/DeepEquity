from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from psycopg.rows import dict_row

from deepequity.core.db import get_pool
from deepequity.core.logging import get_logger
from deepequity.evaluation.golden_queries import GOLDEN_QUERIES, GoldenQuery
from deepequity.evaluation.metrics import (
    QueryMetrics,
    aggregate,
    recall_ceiling,
    score_query,
)
from deepequity.retrieval.dense import search_dense
from deepequity.retrieval.keyword import search_keyword
from deepequity.retrieval.search import hybrid_search

logger = get_logger("deepequity.evaluation")


@dataclass
class ModeResult:
    mode: str
    per_query: list[QueryMetrics] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        scores = aggregate(self.per_query)
        ordered = sorted(self.latencies_ms)
        #p50 rather than mean, one slow first call shouldn't define the number
        scores["p50_ms"] = ordered[len(ordered) // 2] if ordered else 0.0
        #recall@5 looks terrible on its own, and it shouldn't. most of these questions
        #have far more than five right answers in the corpus, one has 128, so returning
        #five perfect results still scores about 0.25. reporting what fraction of the
        #achievable maximum we actually got turns a misleading number into a useful one.
        ceiling = recall_ceiling(self.per_query)
        scores["recall_ceiling"] = ceiling
        scores["recall_vs_ceiling"] = scores["recall"] / ceiling if ceiling else 0.0
        return scores

    #scores split by question type. this matters for reading the results honestly: the
    #factual and keyword questions name terms the filing really uses, so keyword search
    #has a natural advantage there, while the semantic questions are worded nothing like
    #the document and only meaning-based search can do well. one blended average would
    #hide both effects and make the comparison look more even than it is.
    def by_category(self) -> dict[str, dict[str, float]]:
        groups: dict[str, list[QueryMetrics]] = {}
        for metrics in self.per_query:
            category = metrics.query_id.split("-")[0]
            groups.setdefault(category, []).append(metrics)
        return {category: aggregate(items) for category, items in sorted(groups.items())}


#loads every chunk so relevance can be judged across the whole corpus rather than only
#the passages some search happened to return.
#
#this is the part that makes recall a real number. the usual shortcut is to pool the top
#results from each method and only judge those, which quietly defines "all the right
#answers" as "the ones something already found", and no method can ever look like it
#missed anything. judging all 853 chunks costs nothing at this size and gives an honest
#denominator.
async def load_corpus() -> list[dict[str, Any]]:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("SELECT id, text FROM child_chunks ORDER BY id")
            return list(await cur.fetchall())


#works out, for one question, which chunks in the entire corpus are correct answers
def find_relevant_ids(query: GoldenQuery, corpus: list[dict[str, Any]]) -> set[int]:
    return {
        int(row["id"]) for row in corpus if query.is_relevant(str(row["text"]))
    }


#runs one retrieval mode over the whole query set and scores it
async def evaluate_mode(
    mode: str, corpus: list[dict[str, Any]], k: int = 5
) -> ModeResult:
    result = ModeResult(mode=mode)

    for golden in GOLDEN_QUERIES:
        relevant_ids = find_relevant_ids(golden, corpus)

        start = time.perf_counter()
        if mode == "dense":
            chunks = await search_dense(golden.query, k)
        elif mode == "keyword":
            chunks = await search_keyword(golden.query, k)
        elif mode == "hybrid":
            chunks = (await hybrid_search(golden.query, top_k=k, use_reranker=False)).chunks
        elif mode == "hybrid+rerank":
            chunks = (await hybrid_search(golden.query, top_k=k, use_reranker=True)).chunks
        else:
            raise ValueError(f"unknown mode: {mode}")
        result.latencies_ms.append((time.perf_counter() - start) * 1000)

        relevance = [chunk.child_chunk_id in relevant_ids for chunk in chunks]
        result.per_query.append(
            score_query(golden.query_id, relevance, k, len(relevant_ids))
        )

    return result


#the whole comparison: every mode over every question, so the numbers sit side by side
#and the effect of each stage is visible rather than assumed
async def run_evaluation(k: int = 5) -> dict[str, ModeResult]:
    corpus = await load_corpus()
    logger.info("evaluation_corpus_loaded", chunks=len(corpus), queries=len(GOLDEN_QUERIES))

    #warm the models first so the first mode measured isn't unfairly penalised by a
    #one-off model load
    await hybrid_search("warmup query", top_k=1, use_reranker=True)

    results: dict[str, ModeResult] = {}
    for mode in ("keyword", "dense", "hybrid", "hybrid+rerank"):
        results[mode] = await evaluate_mode(mode, corpus, k)
        logger.info("evaluation_mode_complete", mode=mode, **results[mode].summary())

    return results
