from __future__ import annotations

from typing import Any

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

import deepequity.agents.run_store as run_store
import deepequity.api.routes.research as research_route
from deepequity.agents.run_store import RunStatus
from deepequity.agents.schemas import ConfidenceBreakdown, ResearchNote
from deepequity.api.auth import create_access_token


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token('analyst-1')}"}


def _note() -> ResearchNote:
    return ResearchNote(
        ticker="AAPL",
        summary="the note",
        confidence=ConfidenceBreakdown(
            evidence_strength=0.7,
            reasoning_consistency=0.8,
            source_diversity=0.6,
            data_freshness=0.9,
            overall=0.7,
        ),
    )


@pytest.fixture
def store_redis(monkeypatch: pytest.MonkeyPatch) -> fakeredis.aioredis.FakeRedis:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(run_store, "get_redis", lambda: fake)
    return fake


# --- the run store -------------------------------------------------------------------


async def test_a_run_round_trips_through_redis(store_redis: Any) -> None:
    await run_store.create("run-1", "AAPL")

    run = await run_store.get("run-1")

    assert run is not None
    assert run.ticker == "AAPL"
    assert run.status is RunStatus.QUEUED
    assert run.note is None


async def test_a_finished_note_survives_the_round_trip(store_redis: Any) -> None:
    # The note is nested pydantic going into a flat redis hash, so this is the bit most
    # likely to quietly lose data.
    await run_store.create("run-1", "AAPL")

    await run_store.update("run-1", status=RunStatus.COMPLETE, note=_note())
    run = await run_store.get("run-1")

    assert run is not None
    assert run.note is not None
    assert run.note.summary == "the note"
    assert run.note.confidence.overall == 0.7


async def test_empty_strings_come_back_as_none(store_redis: Any) -> None:
    # Redis has no null, so None goes in as "". Without converting back, a caller sees
    # error: "" and can't tell that apart from an actual empty error message.
    await run_store.create("run-1", "AAPL")

    run = await run_store.get("run-1")

    assert run is not None
    assert run.error is None
    assert run.stop_reason is None


async def test_updates_leave_other_fields_alone(store_redis: Any) -> None:
    await run_store.create("run-1", "AAPL")

    await run_store.update("run-1", stage="debating")
    run = await run_store.get("run-1")

    assert run is not None
    assert run.stage == "debating"
    assert run.ticker == "AAPL"  # untouched


async def test_updating_a_run_that_does_not_exist_returns_none(store_redis: Any) -> None:
    assert await run_store.update("nope", stage="x") is None


# --- the endpoint --------------------------------------------------------------------


@pytest.fixture
def wired(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_redis: Any
) -> dict[str, Any]:
    """The endpoint with the graph replaced, so the api contract is tested without
    spending four minutes and real tokens on a research run."""
    monkeypatch.setattr(run_store, "get_redis", lambda: fake_redis)
    captured: dict[str, Any] = {"ran": []}

    async def fake_run(ticker: str, run_id: str | None = None, on_stage: Any = None):
        captured["ran"].append((ticker, run_id))
        if on_stage:
            await on_stage("planning", {"round_count": 0, "tokens_used": 10})
        return {
            "ticker": ticker,
            "note": _note(),
            "round_count": 1,
            "tokens_used": 500,
            "stop_reason": "completed",
        }

    monkeypatch.setattr(research_route, "run_research", fake_run)
    return captured


def test_research_requires_auth(client: TestClient) -> None:
    assert client.post("/research", json={"ticker": "AAPL"}).status_code == 401


def test_starting_a_run_returns_202_and_a_poll_url(
    client: TestClient, wired: dict[str, Any]
) -> None:
    # 202 not 200: a run takes minutes, so the request accepts the work rather than
    # doing it. No client or proxy will hold a connection open that long.
    response = client.post("/research", json={"ticker": "aapl"}, headers=_auth())

    assert response.status_code == 202
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["status"] == "queued"
    assert body["poll_url"] == f"/research/{body['run_id']}"


def test_the_note_arrives_at_the_poll_url(
    client: TestClient, wired: dict[str, Any]
) -> None:
    # TestClient runs background tasks before returning, so by the time we poll the fake
    # run has finished.
    run_id = client.post("/research", json={"ticker": "AAPL"}, headers=_auth()).json()["run_id"]

    response = client.get(f"/research/{run_id}", headers=_auth())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "complete"
    assert body["note"]["summary"] == "the note"
    assert body["tokens_used"] == 500


def test_polling_an_unknown_run_404s(client: TestClient, wired: dict[str, Any]) -> None:
    assert client.get("/research/does-not-exist", headers=_auth()).status_code == 404


def test_a_crashed_run_is_recorded_rather_than_hanging(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_redis: Any
) -> None:
    # A background task has nobody to raise to, the request finished long ago. If the
    # failure isn't written to the run, the client polls forever against a run that
    # silently died.
    monkeypatch.setattr(run_store, "get_redis", lambda: fake_redis)

    async def exploding_run(ticker: str, run_id: str | None = None, on_stage: Any = None):
        raise RuntimeError("groq fell over")

    monkeypatch.setattr(research_route, "run_research", exploding_run)

    run_id = client.post("/research", json={"ticker": "AAPL"}, headers=_auth()).json()["run_id"]
    body = client.get(f"/research/{run_id}", headers=_auth()).json()

    assert body["status"] == "failed"
    assert "groq fell over" in body["error"]


def test_a_run_that_ends_without_a_note_is_a_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_redis: Any
) -> None:
    # Hitting the token budget before any thesis exists ends the graph with no note.
    # That's not a success with a missing field, and the status should say so.
    monkeypatch.setattr(run_store, "get_redis", lambda: fake_redis)

    async def noteless(ticker: str, run_id: str | None = None, on_stage: Any = None):
        return {"ticker": ticker, "note": None, "stop_reason": "token_budget_exceeded"}

    monkeypatch.setattr(research_route, "run_research", noteless)

    run_id = client.post("/research", json={"ticker": "AAPL"}, headers=_auth()).json()["run_id"]
    body = client.get(f"/research/{run_id}", headers=_auth()).json()

    assert body["status"] == "failed"
    assert body["stop_reason"] == "token_budget_exceeded"


def test_the_run_id_becomes_the_checkpoint_thread(
    client: TestClient, wired: dict[str, Any]
) -> None:
    # The run id is what ties a run to its checkpoints, so a run that died can be
    # re-invoked with the same id and pick up where it stopped.
    run_id = client.post("/research", json={"ticker": "AAPL"}, headers=_auth()).json()["run_id"]

    assert wired["ran"] == [("AAPL", run_id)]
