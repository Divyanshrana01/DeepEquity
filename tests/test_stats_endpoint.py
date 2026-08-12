from __future__ import annotations

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

import deepequity.agents.cache as cache_module
from deepequity.api.auth import create_access_token
from deepequity.core.config import get_settings


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token('analyst-1')}"}


@pytest.fixture
def stats_redis(monkeypatch: pytest.MonkeyPatch) -> fakeredis.aioredis.FakeRedis:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(cache_module, "get_redis", lambda: fake)
    return fake


def test_stats_needs_auth(client: TestClient) -> None:
    assert client.get("/stats").status_code == 401


#the cheapest possible bug in this system is a role quietly pointing at the expensive
#model. nothing breaks, the notes still read fine, and the bill is several times what it
#should be. this endpoint exists so that's visible at a glance, so it's worth a test.
def test_stats_shows_which_model_each_agent_is_on(
    client: TestClient, stats_redis: fakeredis.aioredis.FakeRedis
) -> None:
    settings = get_settings()

    body = client.get("/stats", headers=_auth()).json()
    by_role = {row["role"]: row for row in body["routing"]}

    assert by_role["synthesis"]["model"] == settings.llm_strong_model
    assert by_role["planner"]["model"] == settings.llm_fast_model
    assert by_role["bull"]["model"] == settings.llm_fast_model
    #and the price is reported next to it, so "which is the expensive one" needs no
    #outside knowledge
    assert by_role["synthesis"]["output_per_million_usd"] > 0


def test_hit_rate_is_reported_with_its_denominator(
    client: TestClient, stats_redis: fakeredis.aioredis.FakeRedis
) -> None:
    #a hit rate on its own is not a measurement. 100% off two lookups means nothing, and
    #the only way to see that is the count next to it.
    body = client.get("/stats", headers=_auth()).json()

    assert body["cache"]["lookups"] == 0
    assert body["cache"]["hit_rate"] == 0.0
    assert "misses" in body["cache"]
