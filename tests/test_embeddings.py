from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import deepequity.ingestion.embeddings as embeddings
from deepequity.core.config import get_settings


# Stands in for the real model. Loading BAAI/bge-small-en-v1.5 downloads ~130MB and
# takes seconds, which we don't want on every CI run. The live docker run is what
# proves the real model works, these tests cover our wrapper around it.
class FakeModel:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], batch_size: int = 32) -> Any:
        self.calls.append(list(texts))
        return (np.ones(self.dim, dtype=np.float32) for _ in texts)


@pytest.fixture(autouse=True)
def _clear_model_cache() -> None:
    embeddings._model = None
    get_settings.cache_clear()


async def test_empty_input_skips_the_model_entirely() -> None:
    # Guards against loading a 130MB model just to embed nothing.
    assert await embeddings.embed_texts([]) == []


async def test_returns_plain_lists_of_floats(monkeypatch: pytest.MonkeyPatch) -> None:
    # fastembed hands back numpy arrays, but postgres needs plain floats, so the
    # conversion has to happen here rather than blowing up at insert time.
    monkeypatch.setattr(embeddings, "get_model", lambda: FakeModel())

    vectors = await embeddings.embed_texts(["one", "two"])

    assert len(vectors) == 2
    assert all(isinstance(value, float) for value in vectors[0])
    assert len(vectors[0]) == 384


async def test_dimension_mismatch_is_caught_early(monkeypatch: pytest.MonkeyPatch) -> None:
    # Swapping the model without updating embedding_dim (and the vector column) would
    # otherwise surface as a confusing postgres type error thousands of rows later.
    monkeypatch.setattr(embeddings, "get_model", lambda: FakeModel(dim=768))

    with pytest.raises(ValueError, match="768-dim"):
        await embeddings.embed_texts(["text"])


async def test_model_is_loaded_once_and_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = 0

    def fake_ctor(model_name: str) -> FakeModel:
        nonlocal loads
        loads += 1
        return FakeModel()

    monkeypatch.setattr(embeddings, "TextEmbedding", fake_ctor)

    embeddings.get_model()
    embeddings.get_model()

    assert loads == 1
