from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from deepequity.agents import run_store
from deepequity.agents.graph import run_research
from deepequity.agents.run_store import ResearchRun, RunStatus
from deepequity.agents.state import ResearchState
from deepequity.api.auth import require_auth
from deepequity.core.logging import get_logger

logger = get_logger("deepequity.api.research")

router = APIRouter(tags=["research"])


class ResearchRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)


class ResearchAccepted(BaseModel):
    run_id: str
    ticker: str
    status: RunStatus
    #told to the caller rather than left to be discovered, since a run takes minutes and
    #a client that polls every 200ms is just wasting both our time
    poll_url: str
    estimated_seconds: int


#does the actual work, outside the request. a run takes minutes, so it cannot happen while
#a client holds a connection open, most proxies give up somewhere around thirty seconds.
async def _execute(run_id: str, ticker: str) -> None:
    async def report(stage: str, state: ResearchState) -> None:
        await run_store.update(
            run_id,
            status=RunStatus.RUNNING,
            stage=stage,
            rounds=state.get("round_count", 0),
            tokens_used=state.get("tokens_used", 0),
        )

    try:
        final = await run_research(ticker, run_id=run_id, on_stage=report)
    except Exception as exc:  # noqa: BLE001 - nothing may escape a background task
        #an exception here has nobody to propagate to, the request finished long ago. if
        #it isn't recorded on the run the client polls forever against a run that silently
        #died.
        logger.exception("research_run_failed", run_id=run_id, ticker=ticker)
        await run_store.update(
            run_id, status=RunStatus.FAILED, stage="failed", error=str(exc)[:1000]
        )
        return

    note = final.get("note")
    await run_store.update(
        run_id,
        status=RunStatus.COMPLETE if note else RunStatus.FAILED,
        stage="complete" if note else "failed",
        rounds=final.get("round_count", 0),
        tokens_used=final.get("tokens_used", 0),
        stop_reason=final.get("stop_reason"),
        note=note,
        error=None if note else "the run finished without producing a note",
    )


#starts a research run and returns straight away.
#
#the plan describes this endpoint as returning the note. in practice a run takes about four
#minutes, which no client or reverse proxy will wait for, so it follows the same shape as
#/ingest: accept the work, hand back an id, and let the caller poll. the note arrives at
#the polling endpoint below.
@router.post("/research", status_code=status.HTTP_202_ACCEPTED)
async def start_research(
    payload: ResearchRequest,
    background: BackgroundTasks,
    _claims: Annotated[dict[str, Any], Depends(require_auth)],
) -> ResearchAccepted:
    ticker = payload.ticker.strip().upper()
    run_id = str(uuid.uuid4())

    await run_store.create(run_id, ticker)
    background.add_task(_execute, run_id, ticker)

    logger.info("research_accepted", run_id=run_id, ticker=ticker)
    return ResearchAccepted(
        run_id=run_id,
        ticker=ticker,
        status=RunStatus.QUEUED,
        poll_url=f"/research/{run_id}",
        estimated_seconds=240,
    )


#where the note turns up. reports the stage while it's still going so a caller can tell the
#difference between working and stuck.
@router.get("/research/{run_id}")
async def get_research(
    run_id: str,
    _claims: Annotated[dict[str, Any], Depends(require_auth)],
) -> ResearchRun:
    run = await run_store.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No such research run"
        )
    return run


#the synchronous version, for when you actually want to wait, mostly demos and testing.
#separate from the endpoint above rather than a flag on it, because holding a connection
#open for four minutes is a deliberate choice and should look like one at the call site.
@router.post("/research/sync")
async def research_sync(
    payload: ResearchRequest,
    _claims: Annotated[dict[str, Any], Depends(require_auth)],
) -> ResearchRun:
    ticker = payload.ticker.strip().upper()
    run_id = str(uuid.uuid4())
    await run_store.create(run_id, ticker)

    await asyncio.shield(_execute(run_id, ticker))

    run = await run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=500, detail="run vanished mid-flight")
    return run
