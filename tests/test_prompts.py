from __future__ import annotations

import pytest

from deepequity.prompts.loader import list_prompts, load


def test_every_prompt_in_the_library_loads() -> None:
    # A prompt file that can't be read only fails when that agent runs, which might be
    # deep into a research run. Better to find out here.
    names = list_prompts()
    assert names, "the prompt library is empty"

    for stem in names:
        name, _, version = stem.rpartition("_")
        prompt = load(name, version)
        assert prompt.text.strip()


def test_the_agents_we_have_all_have_prompts() -> None:
    for name in ("planner", "bull", "bear"):
        assert load(name).text


def test_version_travels_with_the_text() -> None:
    # Phase 6 changes one prompt at a time and re-runs the evaluation. A score is
    # meaningless unless you can say which wording produced it, so the version has to
    # come along rather than being implied.
    prompt = load("bull")

    assert prompt.id == "bull@v1"


def test_a_missing_prompt_says_what_does_exist() -> None:
    with pytest.raises(FileNotFoundError, match="available"):
        load("nonexistent")


def test_bull_and_bear_are_genuinely_different() -> None:
    # If these ever converged, the adversarial structure would be theatre: two agents
    # with the same instructions producing two versions of the same answer.
    assert load("bull").text != load("bear").text


def test_both_debate_prompts_demand_citations() -> None:
    # The one rule that separates this from an opinion generator.
    for name in ("bull", "bear"):
        text = load(name).text.lower()
        assert "cite" in text
        assert "chunk id" in text


def test_both_debate_prompts_forbid_inventing_ids() -> None:
    for name in ("bull", "bear"):
        assert "never invent a chunk id" in load(name).text.lower()
