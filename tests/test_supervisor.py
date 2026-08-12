from __future__ import annotations

from typing import Any

import pytest

from deepequity.agents.schemas import ResearchScope, Stance, Thesis
from deepequity.agents.state import ResearchState
from deepequity.agents.supervisor import Route, StopReason, decide_next, stop_reason_for
from deepequity.core.config import get_settings
from deepequity.retrieval.models import RetrievedChunk


@pytest.fixture(autouse=True)
def _clean_settings() -> None:
    get_settings.cache_clear()


def _scope() -> ResearchScope:
    return ResearchScope(
        focus="competitive position",
        search_queries=["competition risk"],
        reasoning="because the question is about rivals",
    )


def _chunk(chunk_id: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        child_chunk_id=chunk_id,
        document_id=1,
        ticker="AAPL",
        doc_type="10-K",
        child_text="some evidence",
        score=0.9,
        retrieval_method="dense",
    )


def _thesis(stance: Stance, gaps: list[str] | None = None) -> Thesis:
    return Thesis(
        stance=stance,
        summary="a case",
        claims=[],
        evidence_gaps=gaps or [],
    )


def _state(**overrides: Any) -> ResearchState:
    base: dict[str, Any] = {
        "ticker": "AAPL",
        "scope": None,
        "evidence": [],
        "theses": [],
        "round_count": 0,
        "tokens_used": 0,
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


def test_a_fresh_run_plans_first() -> None:
    # Scoping before gathering is the point of having a planner at all.
    assert decide_next(_state()) is Route.PLAN


def test_once_scoped_it_retrieves() -> None:
    assert decide_next(_state(scope=_scope())) is Route.RETRIEVE


def test_with_evidence_it_debates() -> None:
    assert decide_next(_state(scope=_scope(), evidence=[_chunk()])) is Route.DEBATE


def test_after_the_debate_it_synthesises() -> None:
    state = _state(
        scope=_scope(),
        evidence=[_chunk()],
        theses=[_thesis(Stance.BULL), _thesis(Stance.BEAR)],
        round_count=1,
    )

    assert decide_next(state) is Route.SYNTHESISE


def test_another_round_when_both_sides_report_gaps() -> None:
    # Both sides saying evidence is missing is a real signal. One side saying it usually
    # just means that side is losing the argument.
    state = _state(
        scope=_scope(),
        evidence=[_chunk()],
        theses=[
            _thesis(Stance.BULL, gaps=["no margin data"]),
            _thesis(Stance.BEAR, gaps=["no litigation detail"]),
        ],
        round_count=1,
    )

    assert decide_next(state) is Route.DEBATE


def test_no_extra_round_when_only_one_side_complains() -> None:
    state = _state(
        scope=_scope(),
        evidence=[_chunk()],
        theses=[_thesis(Stance.BULL, gaps=["I want more"]), _thesis(Stance.BEAR)],
        round_count=1,
    )

    assert decide_next(state) is Route.SYNTHESISE


def test_the_round_ceiling_stops_an_endless_debate(monkeypatch: pytest.MonkeyPatch) -> None:
    # The answer to "how do you stop an agent looping forever". Both sides want another
    # round and they are refused, because the ceiling outranks their opinion.
    monkeypatch.setenv("MAX_DEBATE_ROUNDS", "2")
    get_settings.cache_clear()

    state = _state(
        scope=_scope(),
        evidence=[_chunk()],
        theses=[
            _thesis(Stance.BULL, gaps=["still not enough"]),
            _thesis(Stance.BEAR, gaps=["still not enough"]),
        ],
        round_count=2,
    )

    assert decide_next(state) is Route.SYNTHESISE


def test_token_budget_stops_the_run_even_mid_debate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "1000")
    get_settings.cache_clear()

    state = _state(
        scope=_scope(),
        evidence=[_chunk()],
        theses=[_thesis(Stance.BULL, gaps=["more"]), _thesis(Stance.BEAR, gaps=["more"])],
        round_count=0,
        tokens_used=1500,
    )

    # There's something to work with, so write the note rather than throwing the run away.
    assert decide_next(state) is Route.SYNTHESISE


def test_token_budget_with_nothing_to_show_just_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "1000")
    get_settings.cache_clear()

    state = _state(scope=_scope(), evidence=[_chunk()], theses=[], tokens_used=1500)

    assert decide_next(state) is Route.END


def test_budget_is_checked_before_spending_more(monkeypatch: pytest.MonkeyPatch) -> None:
    # An over-budget run must not be sent off to plan or retrieve, that would spend more
    # money after the limit was already breached.
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "100")
    get_settings.cache_clear()

    unplanned_but_broke = _state(scope=None, tokens_used=500)

    assert decide_next(unplanned_but_broke) is not Route.PLAN


def test_retrieval_finding_nothing_ends_the_run() -> None:
    # Searching the same corpus the same way again would return the same nothing.
    state = _state(scope=_scope(), evidence=[], round_count=1)

    assert decide_next(state) is Route.END


def test_stop_reason_records_why_it_finished(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "1000")
    get_settings.cache_clear()
    assert stop_reason_for(_state(tokens_used=2000)) is StopReason.TOKEN_BUDGET

    get_settings.cache_clear()
    assert stop_reason_for(_state(evidence=[])) is StopReason.NO_EVIDENCE

    monkeypatch.setenv("MAX_DEBATE_ROUNDS", "2")
    get_settings.cache_clear()
    finished = _state(evidence=[_chunk()], round_count=2)
    assert stop_reason_for(finished) is StopReason.MAX_ROUNDS

    get_settings.cache_clear()
    clean = _state(evidence=[_chunk()], round_count=1)
    assert stop_reason_for(clean) is StopReason.COMPLETED
