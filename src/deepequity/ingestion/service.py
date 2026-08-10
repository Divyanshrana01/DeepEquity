from __future__ import annotations

from deepequity.core.logging import get_logger
from deepequity.ingestion import events, mcp_client, repository
from deepequity.ingestion.errors import PermanentIngestionError
from deepequity.ingestion.models import IngestionEvent, IngestResponse

logger = get_logger("deepequity.ingestion.service")


#The api side of ingestion: work out which filing is being asked for, skip it if we've
#already got it, otherwise store it and queue it for the worker.
#
#Note the order here. We have to call the MCP tool before the idempotency check, because
#the thing that makes a filing unique is its accession number and we don't know that
#until SEC tells us. So the first check saves us the reprocessing work, not the fetch.
async def request_ingestion(ticker: str, form_type: str) -> IngestResponse:
    ticker = ticker.strip().upper()
    form_type = form_type.strip().upper()

    filing = await mcp_client.fetch_filing(ticker, form_type)
    if filing.get("status") != "ok":
        #An unknown ticker or a form type this company never filed won't fix itself.
        raise PermanentIngestionError(filing.get("error", "filing fetch failed"))

    source_ref = filing["accession_number"]

    #First idempotency check: already have this exact filing? Then don't queue it again.
    existing = await repository.find_by_source(ticker, form_type, source_ref)
    if existing is not None:
        logger.info("ingestion_skipped_duplicate", ticker=ticker, source_ref=source_ref)
        return IngestResponse(
            document_id=existing.id,
            status=existing.status,
            ticker=ticker,
            doc_type=form_type,
            source_ref=source_ref,
            already_ingested=True,
        )

    document, created = await repository.create_pending(
        ticker=ticker,
        doc_type=form_type,
        source_ref=source_ref,
        source_url=filing.get("document_url"),
        raw_text=filing.get("text"),
    )

    #created is False when two requests raced and the other one won. Their event is
    #already on the stream, so publishing a second one would just be a duplicate.
    if created:
        await events.publish(
            IngestionEvent(
                document_id=document.id,
                ticker=ticker,
                doc_type=form_type,
                source_ref=source_ref,
            )
        )

    return IngestResponse(
        document_id=document.id,
        status=document.status,
        ticker=ticker,
        doc_type=form_type,
        source_ref=source_ref,
        already_ingested=not created,
    )
