from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from deepequity.api.auth import require_auth
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
