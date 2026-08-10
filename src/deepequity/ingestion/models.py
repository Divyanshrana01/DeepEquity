from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


#Where a document is in the pipeline. Kept as an enum so a typo like "complete " can't
#silently become a status nobody ever queries for.
class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


#What the caller sends to POST /ingest.
class IngestRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    form_type: str = Field(default="10-K", max_length=20)


#What POST /ingest sends back. Deliberately small: the work happens in the background,
#so all the caller gets is an id to poll and whether this was a fresh doc or one we
#already had.
class IngestResponse(BaseModel):
    document_id: int
    status: DocumentStatus
    ticker: str
    doc_type: str
    source_ref: str
    #True when the doc already existed, meaning we skipped the fetch entirely.
    already_ingested: bool


#One row of the documents table, as the rest of the code sees it.
class Document(BaseModel):
    id: int
    ticker: str
    doc_type: str
    source_ref: str
    source_url: str | None = None
    raw_text: str | None = None
    status: DocumentStatus
    attempts: int = 0
    last_error: str | None = None


#The event we drop on the Redis stream. Just the id plus enough context to make worker
#logs readable, the worker re-reads the full row from Postgres anyway, so the event
#never carries the document text.
class IngestionEvent(BaseModel):
    document_id: int
    ticker: str
    doc_type: str
    source_ref: str
