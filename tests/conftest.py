"""Shared test fixtures.

We swap the real Redis client for fakeredis everywhere it's imported, so
tests don't need a live Redis instance and don't leak state between runs.
"""

from collections.abc import Iterator

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def fake_redis() -> fakeredis.aioredis.FakeRedis:
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def client(
    fake_redis: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    import deepequity.api.rate_limit as rate_limit
    import deepequity.api.routes.health as health

    monkeypatch.setattr(rate_limit, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(health, "get_redis", lambda: fake_redis)

    from deepequity.api.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
