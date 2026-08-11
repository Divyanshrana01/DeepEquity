from __future__ import annotations

import anyio
from psycopg.rows import dict_row

from deepequity.core.config import get_settings
from deepequity.core.db import get_pool
from deepequity.core.logging import get_logger
from deepequity.retrieval.dense import search_dense
from deepequity.retrieval.fusion import reciprocal_rank_fusion
from deepequity.retrieval.keyword import search_keyword
from deepequity.retrieval.models import RetrievedChunk, SearchResult
from deepequity.retrieval.rerank import rerank

logger = get_logger("deepequity.retrieval.search")


#the full retrieval pipeline:
#
#  dense search  ─┐
#                 ├─> RRF fusion ─> cross-encoder rerank ─> attach parent text
#  keyword search ┘
#
#the two searches run at the same time rather than one after the other, since neither
#depends on the other and both are waiting on postgres anyway.
async def hybrid_search(
    query: str,
    top_k: int | None = None,
    ticker: str | None = None,
    use_reranker: bool | None = None,
) -> SearchResult:
    settings = get_settings()
    top_k = top_k or settings.retrieval_final_top_k
    use_reranker = settings.reranker_enabled if use_reranker is None else use_reranker
    candidates_per_method = settings.retrieval_candidates_per_method

    dense_results: list[RetrievedChunk] = []
    keyword_results: list[RetrievedChunk] = []

    async def run_dense() -> None:
        nonlocal dense_results
        dense_results = await search_dense(query, candidates_per_method, ticker)

    async def run_keyword() -> None:
        nonlocal keyword_results
        keyword_results = await search_keyword(query, candidates_per_method, ticker)

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_dense)
        tg.start_soon(run_keyword)

    fused = reciprocal_rank_fusion([dense_results, keyword_results])

    if use_reranker and fused:
        final = await rerank(query, fused, top_k)
    else:
        final = fused[:top_k]

    #only now do we fetch parent text, and only for the handful we're actually
    #returning. doing it earlier would mean pulling ~2000 characters for every one of
    #the 60 candidates, most of which get thrown away.
    final = await _attach_parents(final)

    logger.info(
        "hybrid_search",
        query=query[:100],
        dense=len(dense_results),
        keyword=len(keyword_results),
        fused=len(fused),
        returned=len(final),
        reranked=use_reranker,
    )

    return SearchResult(
        query=query,
        chunks=final,
        dense_candidates=len(dense_results),
        keyword_candidates=len(keyword_results),
        fused_candidates=len(fused),
        reranked=bool(use_reranker),
    )


#swaps in the wider parent text for each result. the child is what matched, the parent
#is what an llm needs to make sense of it, a sentence about a lawsuit is nearly useless
#without the paragraph it sits in.
async def _attach_parents(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks:
        return []

    child_ids = [chunk.child_chunk_id for chunk in chunks]

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT c.id AS child_id, p.text AS parent_text
                FROM child_chunks c
                JOIN parent_chunks p ON p.id = c.parent_chunk_id
                WHERE c.id = ANY(%s)
                """,
                (child_ids,),
            )
            rows = await cur.fetchall()

    parent_by_child = {row["child_id"]: row["parent_text"] for row in rows}
    return [
        chunk.model_copy(update={"parent_text": parent_by_child.get(chunk.child_chunk_id)})
        for chunk in chunks
    ]
