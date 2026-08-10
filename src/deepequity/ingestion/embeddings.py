from __future__ import annotations

import anyio
from fastembed import TextEmbedding

from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger

logger = get_logger("deepequity.ingestion.embeddings")

_model: TextEmbedding | None = None


#loads the embedding model once and keeps it. the first call downloads the weights and
#takes a few seconds, every call after that reuses the loaded model, so we never pay
#that cost per document
def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        settings = get_settings()
        logger.info("loading_embedding_model", model=settings.embedding_model)
        _model = TextEmbedding(model_name=settings.embedding_model)
        logger.info("embedding_model_loaded", model=settings.embedding_model)
    return _model


#turns text into vectors. fastembed is synchronous and cpu-bound, so this runs in a
#worker thread to keep the event loop free, otherwise embedding a big filing would
#freeze everything else the worker is doing.
async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return await anyio.to_thread.run_sync(_embed_sync, texts)


def _embed_sync(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    model = get_model()

    #fastembed returns numpy arrays, postgres wants plain lists of floats
    vectors = [
        vector.tolist()
        for vector in model.embed(texts, batch_size=settings.embedding_batch_size)
    ]

    #a dimension mismatch here means the model was changed without updating the schema,
    #which postgres would otherwise reject with a much less obvious error
    if vectors and len(vectors[0]) != settings.embedding_dim:
        raise ValueError(
            f"model {settings.embedding_model} produced {len(vectors[0])}-dim vectors "
            f"but embedding_dim is set to {settings.embedding_dim}"
        )

    return vectors
