from __future__ import annotations

from pydantic import BaseModel


#One retrieved passage. child_text is the small chunk that actually matched, parent_text
#is the wider context we'd hand an LLM. score means different things depending on which
#stage produced it, so retrieval_method records where it came from.
class RetrievedChunk(BaseModel):
    child_chunk_id: int
    document_id: int
    ticker: str
    doc_type: str
    child_text: str
    parent_text: str | None = None
    score: float
    retrieval_method: str


#What a search returns, plus enough detail to see how the result was produced. The
#counts make it obvious whether both halves of the hybrid actually contributed, which
#is the first thing you want to know when results look wrong.
class SearchResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    dense_candidates: int
    keyword_candidates: int
    fused_candidates: int
    reranked: bool
