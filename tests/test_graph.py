from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

import deepequity.agents.graph as graph_module
from deepequity.agents.memory import MemoryEntry
from deepequity.agents.schemas import (
    AgentCost,
    ConfidenceBreakdown,
    ResearchNote,
    ResearchScope,
    Stance,
    Thesis,
)
from deepequity.core.config import get_settings
from deepequity.retrieval.models import RetrievedChunk


def _chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        child_chunk_id=chunk_id,
        document_id=1,
        ticker="AAPL",
        doc_type="10-K",
        child_text=f"text {chunk_id}",
        score=0.9,
        retrieval_method="dense",
    )


#a stand-in for what one agent's call cost. the token split is arbitrary, what matters is
#that the totals the graph reports are built from these and not made up somewhere else.
def _cost(agent: str, tokens: int) -> AgentCost:
    return AgentCost(
        agent=agent,
        model="fake-model",
        prompt_tokens=tokens,
        completion_tokens=0,
        cost_usd=tokens / 1000,
    )


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


# Fakes every agent so the graph's wiring can be tested on its own. What's being checked
# here is the orchestration: what runs, in what order, how many times, and whether the
# limits hold. Whether the agents write good arguments is a separate question.
@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {
        "order": [],
        "debate_rounds": 0,
        "gaps": [],
        "memories": [],
    }

    async def fake_plan(
        ticker: str, memories: Any = None
    ) -> tuple[ResearchScope, AgentCost]:
        calls["order"].append("plan")
        calls["memories_seen"] = memories or []
        return ResearchScope(
            focus="a focus", search_queries=["q1"], reasoning="because"
        ), _cost("planner", 100)

    async def fake_gather(scope: Any, ticker: str, already: Any = None) -> list[RetrievedChunk]:
        calls["order"].append("retrieve")
        return [_chunk(101), _chunk(102)]

    async def fake_debate(
        ticker: str, evidence: Any, focus: str, previous: Any = None
    ) -> tuple[list[Thesis], list[AgentCost]]:
        calls["order"].append("debate")
        calls["debate_rounds"] += 1
        calls["saw_previous"] = previous is not None
        gaps = calls["gaps"]
        return [
            Thesis(stance=Stance.BULL, summary="bull", claims=[], evidence_gaps=gaps),
            Thesis(stance=Stance.BEAR, summary="bear", claims=[], evidence_gaps=gaps),
        ], [_cost("bull", 120), _cost("bear", 80)]

    async def fake_synthesise(
        ticker: str, theses: Any, evidence: Any
    ) -> tuple[ResearchNote, AgentCost]:
        calls["order"].append("synthesise")
        calls["theses_seen"] = len(theses)
        return _note(), _cost("synthesis", 300)

    #long-term memory talks to postgres, which isn't running in ci. the graph tests are
    #about wiring, so both sides are stubbed and what got passed around is recorded.
    async def fake_recall(ticker: str, query: str, limit: Any = None) -> list[MemoryEntry]:
        calls["recalled_for"] = ticker
        return calls["memories"]

    async def fake_remember(
        ticker: str, run_id: str, focus: str, note: ResearchNote
    ) -> bool:
        calls["remembered"] = {"ticker": ticker, "run_id": run_id, "focus": focus}
        return True

    monkeypatch.setattr(graph_module, "recall", fake_recall)
    monkeypatch.setattr(graph_module, "remember", fake_remember)

    monkeypatch.setattr(graph_module, "plan_research", fake_plan)
    monkeypatch.setattr(graph_module, "gather_evidence", fake_gather)
    monkeypatch.setattr(graph_module, "run_debate_round", fake_debate)
    monkeypatch.setattr(graph_module, "synthesise", fake_synthesise)
    graph_module._compiled = None
    get_settings.cache_clear()
    return calls


async def test_the_whole_flow_runs_in_order(wired: dict[str, Any]) -> None:
    final = await graph_module.run_research("aapl")

    assert wired["order"] == ["plan", "retrieve", "debate", "synthesise"]
    assert final["note"] is not None
    assert final["ticker"] == "AAPL"


async def test_tokens_accumulate_across_every_node(wired: dict[str, Any]) -> None:
    # The budget can only be enforced if each node's spend actually reaches the total.
    final = await graph_module.run_research("AAPL")

    assert final["tokens_used"] == 600  # 100 plan + 200 debate + 300 synthesis


async def test_a_second_round_happens_when_both_sides_report_gaps(
    wired: dict[str, Any],
) -> None:
    wired["gaps"] = ["something missing"]

    await graph_module.run_research("AAPL")

    assert wired["debate_rounds"] == 2
    # and the second round is given the first round's theses to build on
    assert wired["saw_previous"] is True


async def test_the_round_ceiling_holds(
    wired: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both sides ask for another round every time. Without the ceiling this never stops.
    monkeypatch.setenv("MAX_DEBATE_ROUNDS", "2")
    get_settings.cache_clear()
    wired["gaps"] = ["always want more"]

    final = await graph_module.run_research("AAPL")

    assert wired["debate_rounds"] == 2
    assert final["note"] is not None


async def test_no_second_round_when_the_evidence_was_enough(wired: dict[str, Any]) -> None:
    wired["gaps"] = []

    await graph_module.run_research("AAPL")

    assert wired["debate_rounds"] == 1


async def test_the_token_budget_cuts_the_run_short(
    wired: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # 100 tokens is spent by planning alone, so the budget is already blown by the time
    # the supervisor next looks. It should still produce something rather than nothing.
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "50")
    get_settings.cache_clear()

    final = await graph_module.run_research("AAPL")

    assert "debate" not in wired["order"]
    assert final["stop_reason"] == "token_budget_exceeded"


async def test_evidence_accumulates_rather_than_being_replaced(
    wired: dict[str, Any],
) -> None:
    # A second round adds to the pool. If the reducer replaced instead of appending, the
    # later rounds would argue from less evidence than the earlier ones.
    wired["gaps"] = ["more please"]

    final = await graph_module.run_research("AAPL")

    # the fake returns the same two chunks each time, dedup keeps it at two
    assert len(final["evidence"]) == 2


async def test_two_runs_for_the_same_ticker_do_not_share_a_checkpoint(
    wired: dict[str, Any],
) -> None:
    # Regression guard. The thread id is what LangGraph resumes from, and it was
    # defaulting to something derived from the ticker. That meant a second run for the
    # same company silently picked up the first run's checkpoint and jumped straight to
    # synthesis, returning stale research while looking like it had done the work.
    await graph_module.run_research("AAPL")
    first_pass = list(wired["order"])
    wired["order"].clear()

    await graph_module.run_research("AAPL")

    assert wired["order"] == first_pass
    assert "debate" in wired["order"]


async def test_all_theses_from_every_round_reach_synthesis(
    wired: dict[str, Any],
) -> None:
    # Two rounds means four theses, and the synthesis should see the debate's history
    # rather than only its final state.
    wired["gaps"] = ["more please"]

    await graph_module.run_research("AAPL")

    assert wired["theses_seen"] == 4


# --- cost tracking and long-term memory ------------------------------------------------


async def test_every_call_shows_up_in_the_cost_breakdown(wired: dict[str, Any]) -> None:
    # One total tells you a run was expensive. The breakdown tells you which agent did it,
    # which is the question you actually have next.
    final = await graph_module.run_research("AAPL")

    assert [cost.agent for cost in final["costs"]] == [
        "planner",
        "bull",
        "bear",
        "synthesis",
    ]
    assert sum(cost.total_tokens for cost in final["costs"]) == final["tokens_used"]


async def test_the_cost_breakdown_grows_with_extra_rounds(wired: dict[str, Any]) -> None:
    # A second debate round is real money, so it has to appear in the breakdown rather
    # than being folded into whatever the first round reported.
    wired["gaps"] = ["more please"]

    final = await graph_module.run_research("AAPL")

    assert [cost.agent for cost in final["costs"]].count("bull") == 2
    assert [cost.agent for cost in final["costs"]].count("bear") == 2


async def test_past_notes_are_recalled_before_planning(wired: dict[str, Any]) -> None:
    # The point of long-term memory is that the planner sees it. Recalling notes and then
    # not passing them anywhere would look identical in the logs and do nothing at all.
    wired["memories"] = [
        MemoryEntry(
            run_id="older-run",
            ticker="AAPL",
            focus="services margin",
            summary="margin gains were slowing",
            created_at=datetime.now(UTC) - timedelta(days=30),
        )
    ]

    final = await graph_module.run_research("AAPL")

    assert wired["recalled_for"] == "AAPL"
    assert len(wired["memories_seen"]) == 1
    # and it stays in the state, so the trace shows what the plan was working from
    assert len(final["memories"]) == 1


async def test_the_finished_note_is_written_back_to_memory(wired: dict[str, Any]) -> None:
    # Without this the second run on a ticker has nothing to recall and the whole feature
    # is inert.
    await graph_module.run_research("AAPL", run_id="run-42")

    assert wired["remembered"] == {
        "ticker": "AAPL",
        "run_id": "run-42",
        "focus": "a focus",
    }
