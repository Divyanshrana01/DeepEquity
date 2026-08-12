from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from deepequity.agents.schemas import AgentCost, ResearchNote
from deepequity.core.logging import get_logger
from deepequity.core.redis_client import aw, get_redis

logger = get_logger("deepequity.agents.run_store")

_KEY = "research:run:{run_id}"
#runs expire after a day. long enough to poll for a result and look at it afterwards,
#short enough that finished research notes don't accumulate in redis forever.
_TTL_SECONDS = 24 * 60 * 60


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


#what a caller sees when they poll. the note is only there once it's finished, everything
#else is available from the moment the run starts so progress is visible rather than a
#silent four minute wait.
class ResearchRun(BaseModel):
    run_id: str
    ticker: str
    status: RunStatus
    created_at: str
    updated_at: str
    #which stage it's on, so polling tells you something more useful than "still going"
    stage: str = "queued"
    rounds: int = 0
    tokens_used: int = 0
    #what the run cost, in dollars, and what the semantic cache saved it. exposed on the
    #run rather than buried in logs because "how much does one of these cost" is a
    #question anyone looking at this will ask within about a minute.
    cost_usd: float = 0.0
    saved_usd: float = 0.0
    cached_calls: int = 0
    #the same money broken down by agent, so the total is checkable rather than asserted
    costs: list[AgentCost] = []
    stop_reason: str | None = None
    note: ResearchNote | None = None
    error: str | None = None


#redis hashes hold flat strings, so anything nested goes in as json and comes back out
#again. the note and the per-agent cost breakdown are both nested, everything else is a
#scalar that str() handles fine.
_NESTED = ("note", "costs")


def _dump(run: ResearchRun) -> dict[str, str]:
    payload = run.model_dump(mode="json")
    nested = {field: payload.pop(field) for field in _NESTED}
    flat = {key: "" if value is None else str(value) for key, value in payload.items()}
    for field, value in nested.items():
        flat[field] = json.dumps(value) if value else ""
    return flat


def _load(raw: dict[str, Any]) -> ResearchRun:
    data: dict[str, Any] = dict(raw)
    note_json = data.pop("note", "") or ""
    costs_json = data.pop("costs", "") or ""
    #empty strings came from None on the way in, put them back rather than letting
    #"stop_reason: ''" reach the caller
    for nullable in ("stop_reason", "error"):
        if not data.get(nullable):
            data[nullable] = None
    data["note"] = ResearchNote.model_validate_json(note_json) if note_json else None
    data["costs"] = (
        [AgentCost.model_validate(item) for item in json.loads(costs_json)]
        if costs_json
        else []
    )
    return ResearchRun.model_validate(data)


async def create(run_id: str, ticker: str) -> ResearchRun:
    now = datetime.now(UTC).isoformat()
    run = ResearchRun(
        run_id=run_id, ticker=ticker, status=RunStatus.QUEUED, created_at=now, updated_at=now
    )
    await _save(run)
    return run


async def get(run_id: str) -> ResearchRun | None:
    redis = get_redis()
    raw = await aw(redis.hgetall(_KEY.format(run_id=run_id)))
    return _load(raw) if raw else None


#updates whichever fields changed and leaves the rest alone
async def update(run_id: str, **changes: Any) -> ResearchRun | None:
    run = await get(run_id)
    if run is None:
        return None
    updated = run.model_copy(update={**changes, "updated_at": datetime.now(UTC).isoformat()})
    await _save(updated)
    return updated


async def _save(run: ResearchRun) -> None:
    redis = get_redis()
    key = _KEY.format(run_id=run.run_id)
    await aw(redis.hset(key, mapping=_dump(run)))  # type: ignore[arg-type]
    await aw(redis.expire(key, _TTL_SECONDS))
