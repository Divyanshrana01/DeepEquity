import pytest

from deepequity.agents.routing import AgentRole, is_cacheable, model_for
from deepequity.agents.schemas import AgentCost
from deepequity.core.config import Settings, get_settings
from deepequity.core.costs import PRICES, cost_usd, price_for


#the whole point of pricing separately is that output tokens cost more than input ones.
#if this ever stops being true the cost numbers are wrong in the direction that flatters
#us, because output is the half we generate the most of under load.
def test_output_costs_more_than_input() -> None:
    for model, price in PRICES.items():
        assert price.output_per_million > price.input_per_million, model


def test_cost_adds_input_and_output_separately() -> None:
    #1M in at $0.10 plus 1M out at $0.50
    assert cost_usd("openai/gpt-oss-20b", 1_000_000, 1_000_000) == pytest.approx(0.60)


#a cheap call is fractions of a cent, so rounding has to keep enough places to show it.
#rounding to two would report every single call in this system as free.
def test_small_call_is_not_rounded_to_zero() -> None:
    assert cost_usd("openai/gpt-oss-20b", 4000, 800) > 0


#an unrecognised model must not price at zero. a silent zero is the one failure mode that
#makes the cost report actively misleading rather than merely incomplete.
def test_unknown_model_is_not_free() -> None:
    unknown = price_for("some/model-nobody-added")
    assert unknown.input_per_million > 0
    assert cost_usd("some/model-nobody-added", 10_000, 1_000) > 0


def test_synthesis_gets_the_strong_model_and_the_rest_do_not() -> None:
    settings = get_settings()
    assert model_for(AgentRole.SYNTHESIS) == settings.llm_strong_model
    for role in (AgentRole.PLANNER, AgentRole.BULL, AgentRole.BEAR):
        assert model_for(role) == settings.llm_fast_model


#routing is config driven so a Phase 6 experiment is a setting change, not a code change.
#this checks the wiring actually reads the setting rather than having the list baked in.
def test_routing_follows_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    tuned = Settings(llm_strong_roles="bull,synthesis")
    monkeypatch.setattr("deepequity.agents.routing.get_settings", lambda: tuned)

    assert model_for(AgentRole.BULL) == tuned.llm_strong_model
    assert model_for(AgentRole.BEAR) == tuned.llm_fast_model


#synthesis is the expensive call and the obvious one to cache, which is why leaving it out
#needs a test holding it there. the note is the only part anyone reads, and a cached note
#is one nobody reasoned about this time: returned confidently, with a fresh timestamp,
#after the evidence underneath it moved. reusing the debate is a saving, reusing the
#conclusion is a stale answer wearing a new date.
def test_the_note_itself_is_never_served_from_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = Settings(semantic_cache_enabled=True)
    monkeypatch.setattr("deepequity.agents.routing.get_settings", lambda: live)

    assert is_cacheable(AgentRole.SYNTHESIS) is False
    assert is_cacheable(AgentRole.BULL) is True
    assert is_cacheable(AgentRole.BEAR) is True
    assert is_cacheable(AgentRole.PLANNER) is True


def test_cache_can_be_turned_off_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    off = Settings(semantic_cache_enabled=False)
    monkeypatch.setattr("deepequity.agents.routing.get_settings", lambda: off)
    assert not any(is_cacheable(role) for role in AgentRole)


#a cache hit costs nothing but should still say what it would have cost, otherwise the
#saving is invisible and the whole feature is unmeasurable
def test_cached_call_records_a_saving_and_no_spend() -> None:
    hit = AgentCost(
        agent="bull",
        model="openai/gpt-oss-20b",
        prompt_tokens=5000,
        completion_tokens=900,
        cost_usd=0.0,
        cached=True,
        saved_usd=cost_usd("openai/gpt-oss-20b", 5000, 900),
    )
    assert hit.cost_usd == 0.0
    assert hit.saved_usd > 0
    assert hit.total_tokens == 5900
