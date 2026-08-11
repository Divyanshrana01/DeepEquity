from __future__ import annotations

import anyio
from fastembed.rerank.cross_encoder import TextCrossEncoder

from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger
from deepequity.retrieval.models import RetrievedChunk

logger = get_logger("deepequity.retrieval.rerank")

_model: TextCrossEncoder | None = None


#loads the cross-encoder once and keeps it, same pattern as the embedding model
def get_reranker() -> TextCrossEncoder:
    global _model
    if _model is None:
        settings = get_settings()
        logger.info("loading_reranker", model=settings.reranker_model)
        _model = TextCrossEncoder(model_name=settings.reranker_model)
        logger.info("reranker_loaded", model=settings.reranker_model)
    return _model


#rescores candidates by reading the query and the passage together.
#
#this is the difference from the embedding model. an embedding turns the query and the
#passage into vectors separately and compares them, which is fast (you can index it) but
#lossy, because neither one was written with the other in mind. a cross-encoder feeds
#both into the model at once, so it can judge whether this specific passage answers this
#specific question. much more accurate, far too slow to run over a whole corpus.
#
#so the shape of the pipeline is: cheap methods cast a wide net, this expensive one
#sorts out the shortlist.
async def rerank(
    query: str, candidates: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    if not candidates:
        return []

    scores = await anyio.to_thread.run_sync(
        _score_sync, query, [chunk.child_text for chunk in candidates]
    )

    rescored = [
        chunk.model_copy(update={"score": float(score), "retrieval_method": "reranked"})
        for chunk, score in zip(candidates, scores, strict=True)
    ]
    rescored.sort(key=lambda chunk: chunk.score, reverse=True)
    return rescored[:top_k]


#the model is synchronous and cpu-heavy, so it runs off the event loop
def _score_sync(query: str, passages: list[str]) -> list[float]:
    model = get_reranker()
    return list(model.rerank(query, passages))
