from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row

from deepequity.core.db import get_pool
from deepequity.ingestion.embeddings import embed_texts
from deepequity.retrieval.models import RetrievedChunk


#pgvector needs a literal like '[0.1,0.2]', a python list won't cast to a vector
def _to_vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"


#searches by meaning rather than words. the query gets embedded with the same model the
#chunks were, then postgres finds the nearest vectors. this is what catches a passage
#about "litigation exposure" when the question asked about "legal risks", where no
#keyword overlaps at all.
#
#the <=> operator is cosine distance, so smaller is closer. we return 1 - distance as
#the score so bigger always means better, matching the keyword search and saving the
#fusion step from having to track which direction each method counts in.
async def search_dense(
    query: str, limit: int, ticker: str | None = None
) -> list[RetrievedChunk]:
    vectors = await embed_texts([query])
    if not vectors:
        return []

    #named parameters, because the vector is used twice (scoring and ordering) and
    #positional placeholders would make the order easy to get subtly wrong
    params: dict[str, Any] = {
        "vector": _to_vector_literal(vectors[0]),
        "limit": limit,
    }

    #filtering by ticker lets an agent research one company without another company's
    #filings bleeding into the results
    ticker_filter = ""
    if ticker:
        ticker_filter = "AND d.ticker = %(ticker)s"
        params["ticker"] = ticker.upper()

    sql = f"""
        SELECT c.id, c.document_id, c.text AS child_text,
               d.ticker, d.doc_type,
               1 - (c.embedding <=> %(vector)s::vector) AS score
        FROM child_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
        {ticker_filter}
        ORDER BY c.embedding <=> %(vector)s::vector
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
            retrieval_method="dense",
        )
        for row in rows
    ]
