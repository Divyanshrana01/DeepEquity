from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row

from deepequity.core.db import get_pool
from deepequity.retrieval.models import RetrievedChunk


#searches by the actual words, which is the half that vector search is bad at. exact
#terms matter enormously in filings: a query for "Item 1A" or a specific product name or
#a dollar figure should hit the passage containing that exact string, and embeddings are
#happy to return something merely similar instead. this is why hybrid beats either alone.
#
#a note on naming: this is postgres full-text ranking (ts_rank_cd), not textbook BM25.
#postgres has no built-in BM25, getting the real thing means an extension like ParadeDB
#or a separate index. ts_rank_cd is a close cousin, it weights by term frequency and
#density, and it's good enough to be the keyword half of the hybrid without adding
#another moving part to the stack.
async def search_keyword(
    query: str, limit: int, ticker: str | None = None
) -> list[RetrievedChunk]:
    #websearch_to_tsquery accepts what a person would actually type, including quoted
    #phrases and OR, and it doesn't throw on punctuation the way plainto_tsquery can
    params: dict[str, Any] = {"query": query, "limit": limit}

    ticker_filter = ""
    if ticker:
        ticker_filter = "AND d.ticker = %(ticker)s"
        params["ticker"] = ticker.upper()

    sql = f"""
        SELECT c.id, c.document_id, c.text AS child_text,
               d.ticker, d.doc_type,
               ts_rank_cd(
                   to_tsvector('english', c.text),
                   websearch_to_tsquery('english', %(query)s)
               ) AS score
        FROM child_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE to_tsvector('english', c.text) @@ websearch_to_tsquery('english', %(query)s)
        {ticker_filter}
        ORDER BY score DESC
        LIMIT %(limit)s
    """

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()

    return [
        RetrievedChunk(
            child_chunk_id=row["id"],
            document_id=row["document_id"],
            ticker=row["ticker"],
            doc_type=row["doc_type"],
            child_text=row["child_text"],
            score=float(row["score"]),
            retrieval_method="keyword",
        )
        for row in rows
    ]
