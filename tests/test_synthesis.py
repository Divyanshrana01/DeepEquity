from __future__ import annotations

from typing import Any

import pytest

import deepequity.agents.synthesis as synthesis
from deepequity.agents.llm import LLMResponse, Usage
from deepequity.agents.routing import AgentRole, model_for
from deepequity.agents.schemas import (
    Citation,
    Claim,
    ConfidenceBreakdown,
    ResearchNote,
    Stance,
    Thesis,
)
from deepequity.core.config import get_settings
from deepequity.retrieval.models import RetrievedChunk


@pytest.fixture(autouse=True)
def _settings() -> None:
    get_settings.cache_clear()


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


def _claim(text: str, cited: list[int], confidence: float = 0.8) -> Claim:
    return Claim(
        statement=text,
        confidence=confidence,
        citations=[Citation(chunk_id=c, document_id=1, quote="q") for c in cited],
    )


def _confidence(**overrides: float) -> ConfidenceBreakdown:
    base = {
        "evidence_strength": 0.9,
        "reasoning_consistency": 0.9,
        "source_diversity": 0.9,
        "data_freshness": 0.9,
        "overall": 0.9,
    }
    base.update(overrides)
    return ConfidenceBreakdown(**base)  # type: ignore[arg-type]


def _note(claims: list[Claim], confidence: ConfidenceBreakdown | None = None) -> ResearchNote:
    return ResearchNote(
        ticker="AAPL",
        summary="a note",
        supporting_claims=claims,
        confidence=confidence or _confidence(),
    )


def _theses(cited: list[int]) -> list[Thesis]:
    return [
        Thesis(stance=Stance.BULL, summary="bull", claims=[_claim("bull point", cited)]),
        Thesis(stance=Stance.BEAR, summary="bear", claims=[_claim("bear point", cited)]),
    ]


def _fake_llm(monkeypatch: pytest.MonkeyPatch, note: ResearchNote) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_complete(
        system_prompt: str, user_prompt: str, schema: Any, **kwargs: Any
    ) -> LLMResponse[Any]:
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        captured["model"] = kwargs.get("model")
        captured["role"] = kwargs.get("role")
        return LLMResponse(
            parsed=note,
            usage=Usage(50, 100),
            model="fake",
            role=kwargs.get("role", AgentRole.SYNTHESIS),
        )

    monkeypatch.setattr(synthesis, "complete_structured", fake_complete)
    return captured


# --- measured confidence -------------------------------------------------------------


def test_unsupported_claims_drag_evidence_strength_down() -> None:
    # A model scoring its own work grades generously. Half the claims being unsupported
    # is a fact about the note, not a matter of opinion, so we measure it.
    claims = [_claim("supported", [101]), _claim("not supported", [])]

    result = synthesis._measure_confidence(
        _confidence(evidence_strength=0.9), claims, [_chunk(101)]
    )

    # halfway between the measured 0.5 and the model's claimed 0.9
    assert result.evidence_strength == 0.7


def test_source_diversity_is_counted_not_asked() -> None:
    # There's no judgement in this one, it's arithmetic, so asking a model for it is
    # strictly worse than working it out.
    one_source = [_claim("a", [101]), _claim("b", [101]), _claim("c", [101])]
    evidence = [_chunk(i) for i in range(101, 107)]

    result = synthesis._measure_confidence(_confidence(source_diversity=0.95), one_source, evidence)

    # one distinct passage out of six available, not the 0.95 the model claimed
    assert result.source_diversity < 0.2


def test_diversity_rises_when_the_note_spreads_across_passages() -> None:
    spread = [_claim("a", [101]), _claim("b", [102]), _claim("c", [103])]
    evidence = [_chunk(i) for i in range(101, 107)]

    result = synthesis._measure_confidence(_confidence(), spread, evidence)

    assert result.source_diversity == 0.5  # 3 of 6


def test_overall_cannot_outrun_the_evidence() -> None:
    # A tidy, confident-sounding note built on nothing is still built on nothing. This is
    # the guard against a high headline score the note hasn't earned.
    nothing_supported = [_claim("assertion", []), _claim("another", [])]

    result = synthesis._measure_confidence(
        _confidence(overall=0.95), nothing_supported, [_chunk(1)]
    )

    assert result.overall <= result.evidence_strength + 0.1
    assert result.overall < 0.6


def test_the_model_keeps_the_dimensions_that_need_judgement() -> None:
    # Whether the reasoning hangs together isn't something we can count, so that one we
    # take from the model rather than inventing a proxy for it.
    claims = [_claim("a", [101])]

    result = synthesis._measure_confidence(
        _confidence(reasoning_consistency=0.42, data_freshness=0.77), claims, [_chunk(101)]
    )

    assert result.reasoning_consistency == 0.42
    assert result.data_freshness == 0.77


def test_a_note_with_no_claims_scores_zero_evidence() -> None:
    result = synthesis._measure_confidence(_confidence(), [], [_chunk(1)])

    assert result.evidence_strength < 0.5


# --- synthesis behaviour -------------------------------------------------------------


async def test_citations_not_argued_by_either_side_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A chunk id appearing for the first time at the final step would be invented at the
    # point it's least likely to be noticed, since nobody re-reads the theses.
    _fake_llm(monkeypatch, _note([_claim("smuggled in", [999])]))

    note, _tokens = await synthesis.synthesise("AAPL", _theses([101]), [_chunk(101)])

    assert note.supporting_claims[0].citations == []


async def test_citations_that_were_argued_survive(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_llm(monkeypatch, _note([_claim("legitimate", [101])]))

    note, _tokens = await synthesis.synthesise("AAPL", _theses([101]), [_chunk(101)])

    assert [c.chunk_id for c in note.supporting_claims[0].citations] == [101]


async def test_synthesis_uses_the_strong_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # This is the reasoning-heavy step and the one anyone actually reads, so it's where
    # the expensive model is worth spending on. The agent asks for a role now rather than
    # naming a model, so the check goes through the routing table, which is the thing that
    # would actually be wrong if this broke.
    captured = _fake_llm(monkeypatch, _note([]))

    await synthesis.synthesise("AAPL", _theses([101]), [_chunk(101)])

    assert captured["role"] is AgentRole.SYNTHESIS
    assert model_for(captured["role"]) == get_settings().llm_strong_model


async def test_both_theses_reach_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _fake_llm(monkeypatch, _note([]))

    await synthesis.synthesise("AAPL", _theses([101]), [_chunk(101)])

    assert "BULL CASE" in captured["user"]
    assert "BEAR CASE" in captured["user"]


async def test_unsupported_claims_are_labelled_in_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The synthesis agent is told to weigh evidence quality, so it needs to see which
    # claims arrived with nothing behind them.
    captured = _fake_llm(monkeypatch, _note([]))
    theses = [
        Thesis(stance=Stance.BULL, summary="b", claims=[_claim("bare assertion", [])]),
        Thesis(stance=Stance.BEAR, summary="b", claims=[_claim("cited", [101])]),
    ]

    await synthesis.synthesise("AAPL", theses, [_chunk(101)])

    assert "UNSUPPORTED" in captured["user"]


async def test_the_ticker_is_ours_not_the_models(monkeypatch: pytest.MonkeyPatch) -> None:
    wrong = _note([])
    wrong = wrong.model_copy(update={"ticker": "WRONG"})
    _fake_llm(monkeypatch, wrong)

    note, _tokens = await synthesis.synthesise("AAPL", _theses([101]), [_chunk(101)])

    assert note.ticker == "AAPL"


async def test_the_disclaimer_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_llm(monkeypatch, _note([_claim("x", [101])]))

    note, _tokens = await synthesis.synthesise("AAPL", _theses([101]), [_chunk(101)])

    assert "not financial advice" in note.disclaimer.lower()
