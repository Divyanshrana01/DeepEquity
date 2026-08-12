from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from deepequity.agents import memory
from deepequity.agents.memory import MemoryEntry, format_memory
from deepequity.agents.schemas import ConfidenceBreakdown, ResearchNote
from deepequity.core.config import Settings


def _entry(days_ago: int = 10, confidence: float | None = 0.62) -> MemoryEntry:
    return MemoryEntry(
        run_id="run-1",
        ticker="AAPL",
        focus="services margin durability",
        summary="Services revenue keeps growing but the margin gain is slowing.",
        key_risks=["China exposure", "regulatory pressure on the App Store"],
        confidence=confidence,
        created_at=datetime.now(UTC) - timedelta(days=days_ago),
        similarity=0.81,
    )


def _note() -> ResearchNote:
    return ResearchNote(
        ticker="AAPL",
        summary="A note.",
        key_risks=["China exposure"],
        confidence=ConfidenceBreakdown(
            evidence_strength=0.5,
            reasoning_consistency=0.6,
            source_diversity=0.4,
            data_freshness=0.7,
            overall=0.55,
        ),
    )


def test_no_memories_adds_nothing_to_the_prompt() -> None:
    assert format_memory([]) == ""


#the framing is the feature here, not the formatting. handed a past conclusion with no
#instruction a model agrees with it, and the system quietly stops researching and starts
#confirming itself. these assertions are about that, so a later prompt tidy-up can't
#delete the guard without a test going red.
def test_recalled_notes_are_framed_as_ours_and_not_as_evidence() -> None:
    text = format_memory([_entry()]).lower()

    assert "not source documents" in text
    assert "do not treat them as evidence" in text
    assert "changed" in text


#age and old confidence are printed because a stale, unsure conclusion should carry less
#weight than a recent confident one, and the model can only do that if it's told
def test_age_and_old_confidence_are_shown() -> None:
    text = format_memory([_entry(days_ago=400, confidence=0.31)])

    assert "400 days ago" in text
    assert "0.31" in text


def test_missing_confidence_is_labelled_rather_than_faked() -> None:
    text = format_memory([_entry(confidence=None)])
    assert "confidence unknown" in text


def test_age_never_goes_negative() -> None:
    future = _entry(days_ago=-5)
    assert future.age_days() == 0


async def test_recall_returns_nothing_when_memory_is_switched_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory, "get_settings", lambda: Settings(memory_enabled=False))
    assert await memory.recall("AAPL", "anything") == []


#memory is an optimisation on top of the research, never a precondition for it. a database
#that's down should cost us the recall and nothing else, so both sides swallow their own
#failures rather than letting them reach the graph.
async def test_a_broken_database_does_not_break_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(*_args, **_kwargs):
        raise RuntimeError("postgres is down")

    async def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    monkeypatch.setattr(memory, "embed_texts", fake_embed)
    monkeypatch.setattr(memory, "get_pool", boom)

    assert await memory.recall("AAPL", "margins") == []
    assert await memory.remember("AAPL", "run-1", "margins", _note()) is False


#recall matches on what a run was about, and the summary alone often describes the answer
#without ever naming the question. the focus has to be in the embedded text or a run about
#a lawsuit won't surface the previous lawsuit note.
def test_embedded_text_includes_the_focus_not_just_the_summary() -> None:
    text = memory._memory_text("ongoing antitrust case", _note())
    assert "ongoing antitrust case" in text
    assert "A note." in text
