from __future__ import annotations

from deepequity.core.db import get_pool
from deepequity.core.logging import get_logger
from deepequity.ingestion.chunking import ParentChunk

logger = get_logger("deepequity.ingestion.chunk_repository")


#pgvector wants a vector literal like '[0.1,0.2]', not a python list, so we format it
#ourselves. psycopg would otherwise send an array and postgres would reject the type.
def _to_vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"


#writes a document's chunks and their vectors in one transaction.
#
#reprocessing a document deletes its old chunks first. without that, a re-run would
#either collide with the unique key or silently leave stale chunks behind alongside the
#new ones, and searches would return text that no longer matches the document.
#
#the whole thing is one transaction on purpose: a half-written set of chunks is worse
#than none, because the document would look processed while only part of it is
#searchable. either it all lands or nothing does.
async def replace_document_chunks(
    document_id: int, parents: list[ParentChunk], child_vectors: list[list[float]]
) -> tuple[int, int]:
    total_children = sum(len(parent.children) for parent in parents)
    if total_children != len(child_vectors):
        raise ValueError(
            f"got {len(child_vectors)} vectors for {total_children} children, "
            "these must line up or chunks get the wrong embeddings"
        )

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.transaction():
            #child rows go too, the cascade on parent_chunks takes care of them
            await conn.execute(
                "DELETE FROM parent_chunks WHERE document_id = %s", (document_id,)
            )
            await conn.execute(
                "DELETE FROM child_chunks WHERE document_id = %s", (document_id,)
            )

            vector_position = 0
            for parent in parents:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO parent_chunks (document_id, chunk_index, text)
                        VALUES (%s, %s, %s)
                        RETURNING id
                        """,
                        (document_id, parent.index, parent.text),
                    )
                    row = await cur.fetchone()
                    parent_id = row[0] if row else None

                if parent_id is None:
                    raise RuntimeError(f"failed to insert parent chunk for doc {document_id}")

                #executemany keeps this to one round trip per parent instead of one per
                #child, which matters when a filing produces thousands of children
                child_rows = [
                    (
                        document_id,
                        parent_id,
                        child.index,
                        child.text,
                        _to_vector_literal(child_vectors[vector_position + offset]),
                    )
                    for offset, child in enumerate(parent.children)
                ]
                vector_position += len(parent.children)

                async with conn.cursor() as cur:
                    await cur.executemany(
                        """
                        INSERT INTO child_chunks
                            (document_id, parent_chunk_id, chunk_index, text, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        child_rows,
                    )

    logger.info(
        "chunks_stored",
        document_id=document_id,
        parents=len(parents),
        children=total_children,
    )
    return len(parents), total_children


#how many child chunks a document has. used by tests and by anything that wants to know
#whether a document actually produced searchable content.
async def count_children(document_id: int) -> int:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT count(*) FROM child_chunks WHERE document_id = %s", (document_id,)
            )
            row = await cur.fetchone()
    return int(row[0]) if row else 0
