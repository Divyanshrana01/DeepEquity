from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from psycopg.rows import dict_row
from pydantic import BaseModel

from deepequity.api.auth import require_auth
from deepequity.core.db import get_pool
from deepequity.ingestion import repository, service
from deepequity.ingestion.errors import PermanentIngestionError
from deepequity.ingestion.models import Document, IngestRequest, IngestResponse

router = APIRouter(tags=["ingestion"])


#Kicks off ingestion for a filing and returns straight away with 202 Accepted. The
#actual processing happens in the worker, so the caller isn't left waiting on SEC and
#an embedding run, they get a document_id and poll the status endpoint below.
@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest(
    payload: IngestRequest,
    _claims: Annotated[dict[str, Any], Depends(require_auth)],
) -> IngestResponse:
    try:
        return await service.request_ingestion(payload.ticker, payload.form_type)
    except PermanentIngestionError as exc:
        #Bad ticker or a form type this company never filed, that's the caller's
        #problem to fix, so 400 rather than 500.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


#Polling endpoint so a caller can see whether their document made it through.
@router.get("/ingest/{document_id}")
async def ingest_status(
    document_id: int,
    _claims: Annotated[dict[str, Any], Depends(require_auth)],
) -> Document:
    document = await repository.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    #raw_text can be 20k characters and nobody polling for a status wants it echoed
    #back, so blank it out on the way to the client.
    return document.model_copy(update={"raw_text": None})


class CorpusEntry(BaseModel):
    ticker: str
    documents: int
    chunks: int


class CorpusResponse(BaseModel):
    tickers: list[CorpusEntry]


#Says which companies we can actually research right now.
#
#Worth an endpoint rather than leaving it to be discovered. Researching a ticker whose
#filings were never ingested doesn't fail loudly, it produces a note built on no evidence,
#which reads like a real answer and isn't one. Anything asking for a ticker should be able
#to check first.
@router.get("/corpus")
async def corpus(
    _claims: Annotated[dict[str, Any], Depends(require_auth)],
) -> CorpusResponse:
    #only complete documents with embedded chunks count. a filing sitting in pending or
    #failed is not something you can search, so listing it would be a lie.
    sql = """
        SELECT d.ticker,
               COUNT(DISTINCT d.id) AS documents,
               COUNT(c.id)          AS chunks
        FROM documents d
        JOIN child_chunks c ON c.document_id = d.id AND c.embedding IS NOT NULL
        WHERE d.status = 'complete'
        GROUP BY d.ticker
        ORDER BY d.ticker
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql)
            rows = await cur.fetchall()

    return CorpusResponse(tickers=[CorpusEntry(**row) for row in rows])
