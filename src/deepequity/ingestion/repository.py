from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row

from deepequity.core.db import get_pool
from deepequity.core.logging import get_logger
from deepequity.ingestion.models import Document, DocumentStatus

logger = get_logger("deepequity.ingestion.repository")

_COLUMNS = "id, ticker, doc_type, source_ref, source_url, raw_text, status, attempts, last_error"


#Turns a database row into a Document. Small helper so every query below doesn't repeat
#the same unpacking.
def _to_document(row: dict[str, Any]) -> Document:
    return Document(**row)


#Looks up a document by its natural key (ticker + type + source ref). This is the first
#idempotency check, the api calls it before fetching anything so a filing we already
#have never gets downloaded twice.
async def find_by_source(ticker: str, doc_type: str, source_ref: str) -> Document | None:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                f"SELECT {_COLUMNS} FROM documents "
                "WHERE ticker = %s AND doc_type = %s AND source_ref = %s",
                (ticker, doc_type, source_ref),
            )
            row = await cur.fetchone()
    return _to_document(row) if row else None


#Fetches one document by id. The worker uses this to load the row an event points at.
async def get(document_id: int) -> Document | None:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(f"SELECT {_COLUMNS} FROM documents WHERE id = %s", (document_id,))
            row = await cur.fetchone()
    return _to_document(row) if row else None


#Inserts a new pending document. The ON CONFLICT clause is the real safety net: if two
#requests for the same filing race past the find_by_source check at the same moment, the
#unique index rejects the second one and we return the existing row instead of blowing up
#or creating a duplicate.
async def create_pending(
    ticker: str, doc_type: str, source_ref: str, source_url: str | None, raw_text: str | None
) -> tuple[Document, bool]:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                f"""
                INSERT INTO documents (ticker, doc_type, source_ref, source_url, raw_text, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, doc_type, source_ref) DO NOTHING
                RETURNING {_COLUMNS}
                """,
                (ticker, doc_type, source_ref, source_url, raw_text, DocumentStatus.PENDING),
            )
            row = await cur.fetchone()

    if row is not None:
        return _to_document(row), True

    #DO NOTHING means someone beat us to it, so load and return their row.
    existing = await find_by_source(ticker, doc_type, source_ref)
    if existing is None:
        raise RuntimeError(f"insert conflicted but no row found for {ticker}/{source_ref}")
    return existing, False


#Moves a document to processing, but only from pending or failed. The status guard in the
#WHERE clause is what makes this safe against duplicate events: if two workers grab the
#same message, only the first UPDATE matches a row, the second gets nothing back and
#knows to skip. This is the second idempotency check.
#
#allow_stuck widens that guard to include rows already marked processing. Only pass it
#for a message reclaimed from a dead worker: a row sitting in processing normally means
#someone is working on it right now, but a reclaimed message means redis has handed us
#exclusive ownership, so the previous owner is gone and the row is ours to finish.
async def mark_processing(document_id: int, allow_stuck: bool = False) -> bool:
    claimable = [DocumentStatus.PENDING, DocumentStatus.FAILED]
    if allow_stuck:
        claimable.append(DocumentStatus.PROCESSING)

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE documents
                SET status = %s, attempts = attempts + 1, updated_at = now()
                WHERE id = %s AND status = ANY(%s)
                """,
                (DocumentStatus.PROCESSING, document_id, claimable),
            )
            return cur.rowcount > 0


#Marks a document done and clears any error left over from an earlier failed attempt.
async def mark_complete(document_id: int) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE documents SET status = %s, last_error = NULL, updated_at = now() "
            "WHERE id = %s",
            (DocumentStatus.COMPLETE, document_id),
        )


#Marks a document failed and records why. Called when retries are exhausted or the
#failure is permanent, the error text is what you read when debugging a DLQ entry.
async def mark_failed(document_id: int, error: str) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE documents SET status = %s, last_error = %s, updated_at = now() WHERE id = %s",
            (DocumentStatus.FAILED, error[:2000], document_id),
        )


#Puts a document back to pending so it can be picked up again on the next attempt.
async def reset_to_pending(document_id: int, error: str) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE documents SET status = %s, last_error = %s, updated_at = now() WHERE id = %s",
            (DocumentStatus.PENDING, error[:2000], document_id),
        )


#How many searchable chunks a document actually has.
#
#Used to tell a document that finished from one that finished with nothing, which the
#status column on its own cannot do.
async def searchable_chunk_count(document_id: int) -> int:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT count(*) FROM child_chunks "
                "WHERE document_id = %s AND embedding IS NOT NULL",
                (document_id,),
            )
            row = await cur.fetchone()
    return int(row[0]) if row else 0


#Puts a document back in the queue's path after it finished badly.
#
#Separate from reset_to_pending, which exists for a retry that's already in flight. This
#one is for a deliberate re-ingest of something that ended up unusable, so the attempt
#count goes back to zero and the old error is cleared rather than kept.
async def reset_for_reingest(document_id: int) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE documents SET status = 'pending', attempts = 0, last_error = NULL, "
            "updated_at = now() WHERE id = %s",
            (document_id,),
        )
    logger.info("document_reset_for_reingest", document_id=document_id)
