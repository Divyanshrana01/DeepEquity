from __future__ import annotations

import asyncio

from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger
from deepequity.ingestion import events, repository
from deepequity.ingestion.chunk_repository import replace_document_chunks
from deepequity.ingestion.chunking import chunk_document
from deepequity.ingestion.embeddings import embed_texts
from deepequity.ingestion.errors import PermanentIngestionError, TransientIngestionError
from deepequity.ingestion.full_document import fetch_full_document
from deepequity.ingestion.html_text import html_to_text
from deepequity.ingestion.models import Document, IngestionEvent

logger = get_logger("deepequity.ingestion.pipeline")


#Turns one stored document into searchable chunks: fetch the whole filing, strip the
#html, split it parent-child, embed the children, write it all to postgres.
#
#We go back to the source for the full text rather than using document.raw_text, which
#only holds the first 20k characters the mcp tool returned. 20k of a 10-K is the cover
#page and contents, the risk factors and md&a that make the document worth searching are
#much further in.
async def process_document(document: Document) -> None:
    if not document.source_url:
        #nothing to fetch and no way to get one, so retrying is pointless
        raise PermanentIngestionError(f"document {document.id} has no source url")

    html = await fetch_full_document(document.source_url)
    text = html_to_text(html)

    if not text.strip():
        #a filing that cleans down to nothing is a broken document, not a blip
        raise PermanentIngestionError(
            f"document {document.id} produced no text after cleaning html"
        )

    parents = chunk_document(text)
    if not parents:
        raise PermanentIngestionError(f"document {document.id} produced no chunks")

    children = [child.text for parent in parents for child in parent.children]
    vectors = await embed_texts(children)

    parent_count, child_count = await replace_document_chunks(document.id, parents, vectors)

    #children are the searchable unit, parents are only the context we hand back once a
    #child matches. a document that produced parents but no children is not a partial
    #success, it's invisible to every search while being marked complete, which is worse
    #than a visible failure because nothing ever prompts anyone to look at it. observed
    #live: one filing sat as complete with zero chunks and no re-ingest could dislodge it.
    if child_count == 0:
        raise PermanentIngestionError(
            f"document {document.id} produced {parent_count} parents but no searchable "
            "chunks, so nothing about it can ever be found"
        )

    logger.info(
        "document_processed",
        document_id=document.id,
        ticker=document.ticker,
        chars=len(text),
        parents=parent_count,
        children=child_count,
    )


#Handles one event end to end, with retries. Returns True if the document ended up
#complete, False if it's headed for the dead letter queue.
#
#Backoff is a plain sleep between attempts inside this call, so a retrying document
#holds up this worker for a few seconds. That's fine at our volume and much easier to
#reason about than re-queueing with a delay, if throughput ever matters we'd move to
#scheduled re-publish instead.
async def handle_event(event: IngestionEvent, reclaimed: bool = False) -> bool:
    settings = get_settings()

    #Second idempotency check. If another worker already grabbed this document, or it's
    #already complete, the status guard in mark_processing fails and we skip the work.
    #A reclaimed message is the exception: its previous owner died, so we're allowed to
    #take over a row that's still sitting in processing.
    claimed = await repository.mark_processing(event.document_id, allow_stuck=reclaimed)
    if not claimed:
        logger.info("event_skipped_not_claimable", document_id=event.document_id)
        return True

    document = await repository.get(event.document_id)
    if document is None:
        raise PermanentIngestionError(f"document {event.document_id} disappeared")

    last_error = ""
    for attempt in range(1, settings.ingestion_max_attempts + 1):
        try:
            await process_document(document)
            await repository.mark_complete(document.id)
            logger.info("ingestion_complete", document_id=document.id, attempt=attempt)
            return True

        except PermanentIngestionError as exc:
            #No point trying again, the input itself is wrong.
            last_error = f"permanent: {exc}"
            logger.warning(
                "ingestion_permanent_failure", document_id=document.id, error=str(exc)
            )
            break

        except (TransientIngestionError, TimeoutError, ConnectionError) as exc:
            last_error = f"transient: {exc}"
            logger.warning(
                "ingestion_transient_failure",
                document_id=document.id,
                attempt=attempt,
                error=str(exc),
            )
            if attempt < settings.ingestion_max_attempts:
                #1s, 4s, 16s with the default base of 4.
                delay = settings.ingestion_backoff_base_seconds ** (attempt - 1)
                await asyncio.sleep(delay)

    await repository.mark_failed(document.id, last_error)
    return False


#What the worker calls per message: run the event, ack it on success, park it in the DLQ
#on failure. Any unexpected exception is treated as a failure too, a crash here would
#otherwise leave the message unacked and stuck redelivering forever.
async def consume_event(message_id: str, event: IngestionEvent, reclaimed: bool = False) -> None:
    try:
        succeeded = await handle_event(event, reclaimed=reclaimed)
    except Exception as exc:  # noqa: BLE001 - last line of defence, nothing should escape
        logger.exception("ingestion_unexpected_error", document_id=event.document_id)
        await repository.mark_failed(event.document_id, f"unexpected: {exc}")
        await events.send_to_dlq(message_id, event, str(exc))
        return

    if succeeded:
        await events.ack(message_id)
    else:
        document = await repository.get(event.document_id)
        error = document.last_error if document and document.last_error else "unknown failure"
        await events.send_to_dlq(message_id, event, error)
