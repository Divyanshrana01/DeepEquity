from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from deepequity.api.auth import require_auth
from deepequity.retrieval.models import SearchResult
from deepequity.retrieval.search import hybrid_search

router = APIRouter(tags=["retrieval"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    #optional so an agent can scope a search to the company it's researching
    ticker: str | None = Field(default=None, max_length=10)
    top_k: int | None = Field(default=None, ge=1, le=50)
    #exposed so we can measure what the reranker is actually buying us, rather than
    #assuming. turning it off is the baseline the reranked numbers get compared against.
    use_reranker: bool | None = None


#runs the hybrid retrieval pipeline and returns the passages, each with the wider parent
#context attached. the agents in phase 3 will call this internally, it's an endpoint now
#so retrieval quality can be checked and measured on its own before anything is built
#on top of it.
@router.post("/search")
async def search(
    payload: SearchRequest,
    _claims: Annotated[dict[str, Any], Depends(require_auth)],
) -> SearchResult:
    return await hybrid_search(
        query=payload.query,
        top_k=payload.top_k,
        ticker=payload.ticker,
        use_reranker=payload.use_reranker,
    )
