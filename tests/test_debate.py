from __future__ import annotations

from typing import Any

import pytest

import deepequity.agents.debate as debate
from deepequity.agents.llm import LLMResponse, Usage
from deepequity.agents.schemas import Citation, Claim, Stance, Thesis
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
        child_text=f"evidence text {chunk_id}",
        parent_text=f"wider context around evidence {chunk_id}",
        score=0.9,
        retrieval_method="dense",
    )


def _claim(statement: str, cited_ids: list[int]) -> Claim:
    return Claim(
        statement=statement,
        confidence=0.8,
        citations=[
            Citation(chunk_id=cid, document_id=1, quote=f"quote from {cid}")
            for cid in cited_ids
        ],
    )


# --- citation validation, the hallucination guard ------------------------------------


def test_invented_citations_are_stripped() -> None:
    # A model under pressure to cite everything will sometimes produce a plausible id
    # that was never in its evidence. An invented citation is worse than a missing one
    # because it looks checkable and isn't.
    thesis = Thesis(
        stance=Stance.BULL,
        summary="a case",
        claims=[_claim("margins improved", [101, 999])],
    )

    cleaned, removed = debate.strip_invalid_citations(thesis, {101, 102})

    assert removed == 1
    assert [c.chunk_id for c in cleaned.claims[0].citations] == [101]


def test_a_claim_losing_every_citation_becomes_visibly_unsupported() -> None:
    # We keep the claim rather than deleting it. Silently dropping it would hide that the
    # agent asserted something it couldn't back, and is_supported is what makes the
    # unsupported ones countable.
    thesis = Thesis(
        stance=Stance.BEAR,
        summary="a case",
        claims=[_claim("something invented", [777, 888])],
    )

    cleaned, removed = debate.strip_invalid_citations(thesis, {101})

    assert removed == 2
    assert len(cleaned.claims) == 1
    assert cleaned.claims[0].is_supported is False


def test_the_same_passage_cited_twice_counts_once() -> None:
    # Observed live: the model padded a claim with [487, 487, 487]. One passage cited
    # three times is one piece of evidence, and leaving the repeats in makes a claim look
    # better supported than it is, which defeats the point of counting citations.
    thesis = Thesis(
        stance=Stance.BULL,
        summary="a case",
        claims=[_claim("margins rose", [487, 487, 487])],
    )

    cleaned, removed = debate.strip_invalid_citations(thesis, {487})

    assert len(cleaned.claims[0].citations) == 1
    # duplicates aren't hallucinations, so they don't count as removed
    assert removed == 0


def test_valid_citations_survive_untouched() -> None:
    thesis = Thesis(
        stance=Stance.BULL,
        summary="a case",
        claims=[_claim("real claim", [101, 102])],
    )

    cleaned, removed = debate.strip_invalid_citations(thesis, {101, 102, 103})

    assert removed == 0
    assert len(cleaned.claims[0].citations) == 2


# --- the debate ----------------------------------------------------------------------


def _fake_llm(monkeypatch: pytest.MonkeyPatch, thesis: Thesis) -> dict[str, Any]:
    """Captures what the agents were actually asked, so the prompts can be checked."""
    captured: dict[str, Any] = {"calls": []}

    async def fake_complete(
        system_prompt: str, user_prompt: str, schema: Any, **kwargs: Any
    ) -> LLMResponse[Any]:
        captured["calls"].append({"system": system_prompt, "user": user_prompt})
        return LLMResponse(parsed=thesis, usage=Usage(10, 20), model="fake")

    monkeypatch.setattr(debate, "complete_structured", fake_complete)
    return captured


async def test_both_sides_get_identical_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    # The core of the whole design. Bull and Bear must differ only in their mandate, so
    # any difference in their conclusions comes from how the evidence was read rather
    # than from one side being handed better material.
    captured = _fake_llm(monkeypatch, Thesis(stance=Stance.BULL, summary="x", claims=[]))
    evidence = [_chunk(101), _chunk(102)]

    await debate.run_debate_round("AAPL", evidence, "competitive position")

    assert len(captured["calls"]) == 2
    bull_user, bear_user = captured["calls"][0]["user"], captured["calls"][1]["user"]
    assert bull_user == bear_user  # same evidence, same focus, same everything


async def test_the_two_sides_get_opposite_mandates(monkeypatch: pytest.MonkeyPatch) -> None:
    # Checks the two agents really are given different instructions, without asserting on
    # exact wording, prompts get reworded constantly and a test that breaks on a line
    # wrap is noise rather than a safety net.
    captured = _fake_llm(monkeypatch, Thesis(stance=Stance.BULL, summary="x", claims=[]))

    await debate.run_debate_round("AAPL", [_chunk(101)], "focus")

    systems = [call["system"] for call in captured["calls"]]
    assert systems[0] != systems[1]
    assert any("bull-side" in s for s in systems)
    assert any("bear-side" in s for s in systems)


async def test_stance_is_forced_to_match_the_seat(monkeypatch: pytest.MonkeyPatch) -> None:
    # The bear agent returning stance=bull would corrupt everything downstream, so the
    # label comes from which seat the agent sat in, not from what it claimed.
    _fake_llm(monkeypatch, Thesis(stance=Stance.BULL, summary="wrong label", claims=[]))

    thesis, _tokens = await debate.build_thesis(
        Stance.BEAR, "AAPL", [_chunk(101)], "focus"
    )

    assert thesis.stance is Stance.BEAR


async def test_invented_citations_are_dropped_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_llm(
        monkeypatch,
        Thesis(
            stance=Stance.BULL,
            summary="x",
            claims=[_claim("cites a chunk that was never given", [404])],
        ),
    )

    thesis, _tokens = await debate.build_thesis(
        Stance.BULL, "AAPL", [_chunk(101)], "focus"
    )

    assert thesis.claims[0].citations == []


async def test_a_second_round_shows_the_agent_its_previous_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without this the agent rewrites the same thesis from scratch and the extra round
    # buys nothing but tokens.
    captured = _fake_llm(monkeypatch, Thesis(stance=Stance.BULL, summary="new", claims=[]))
    earlier = Thesis(stance=Stance.BULL, summary="my earlier argument", claims=[])

    await debate.build_thesis(
        Stance.BULL, "AAPL", [_chunk(101)], "focus", previous=earlier
    )

    user_prompt = captured["calls"][0]["user"]
    assert "my earlier argument" in user_prompt
    assert "do not simply repeat" in user_prompt


async def test_tokens_from_both_sides_are_summed(monkeypatch: pytest.MonkeyPatch) -> None:
    # The budget can only be enforced if every call's cost actually reaches the total.
    _fake_llm(monkeypatch, Thesis(stance=Stance.BULL, summary="x", claims=[]))

    _theses, tokens = await debate.run_debate_round("AAPL", [_chunk(101)], "focus")

    assert tokens == 60  # 30 per side


async def test_results_come_back_bull_then_bear(monkeypatch: pytest.MonkeyPatch) -> None:
    # The two run concurrently, so without a fixed order the output would depend on which
    # request happened to finish first.
    _fake_llm(monkeypatch, Thesis(stance=Stance.BULL, summary="x", claims=[]))

    theses, _tokens = await debate.run_debate_round("AAPL", [_chunk(101)], "focus")

    assert [t.stance for t in theses] == [Stance.BULL, Stance.BEAR]
