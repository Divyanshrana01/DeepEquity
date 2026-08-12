from collections.abc import Iterator

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient


#the semantic cache is off for every test unless a test turns it on itself.
#
#this is not tidiness, it is a real bug that already bit. the cache lives in redis and is
#invisible to callers by design, so with it on, one test's answer gets served to another
#test that expected to call the model, and the assertion fails somewhere far away from the
#cause. the same thing happens against whatever a developer's local redis happens to be
#holding, which makes a green run mean nothing.
#
#tests that are actually about the cache exercise the cache module directly.
@pytest.fixture(autouse=True)
def semantic_cache_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from deepequity.core.config import get_settings

    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_redis() -> fakeredis.aioredis.FakeRedis:
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


#builds a TestClient with the outside world swapped out: fakeredis instead of a real
#redis, and the postgres migration step turned into a no-op. that keeps the api tests
#runnable in ci with no services running
@pytest.fixture
def client(
    fake_redis: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    import deepequity.api.main as api_main
    import deepequity.api.rate_limit as rate_limit
    import deepequity.api.routes.health as health
    import deepequity.ingestion.events as events

    monkeypatch.setattr(rate_limit, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(health, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(events, "get_redis", lambda: fake_redis)

    async def _no_migrations() -> None:
        return None

    async def _no_close() -> None:
        return None

    monkeypatch.setattr(api_main, "run_migrations", _no_migrations)
    monkeypatch.setattr(api_main, "close_pool", _no_close)

    from deepequity.api.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
