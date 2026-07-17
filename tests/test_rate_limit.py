import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import deepequity.api.rate_limit as rate_limit
from deepequity.api.rate_limit import RateLimitMiddleware
from deepequity.core.config import get_settings


@pytest.fixture
def limited_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(rate_limit, "get_redis", lambda: fake_redis)

    # Small limit so the test doesn't need to fire 60 requests to prove the point.
    get_settings.cache_clear()
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
    get_settings.cache_clear()

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"pong": "true"}

    return TestClient(app)


def test_requests_within_limit_succeed(limited_client: TestClient) -> None:
    for _ in range(3):
        response = limited_client.get("/ping")
        assert response.status_code == 200


def test_requests_over_limit_are_rejected(limited_client: TestClient) -> None:
    for _ in range(3):
        limited_client.get("/ping")

    response = limited_client.get("/ping")

    assert response.status_code == 429
    assert "Retry-After" in response.headers


def test_health_path_is_exempt_from_rate_limit(limited_client: TestClient) -> None:
    for _ in range(10):
        response = limited_client.get("/health")
        # No /health route registered on this bare app, exemption just means
        # the middleware doesn't 429 it, routing 404 is expected here.
        assert response.status_code == 404
