from __future__ import annotations

from enum import StrEnum

from deepequity.core.config import get_settings


#the four jobs in the graph that talk to a model. named rather than passed as loose
#strings so cost reporting, caching and routing all agree on what to call each agent,
#and a typo becomes an error instead of a second entry in the cost breakdown.
class AgentRole(StrEnum):
    PLANNER = "planner"
    BULL = "bull"
    BEAR = "bear"
    SYNTHESIS = "synthesis"


#picks which model an agent gets.
#
#this is the money decision. the big model costs 1.5x the small one on input and prices
#its output higher again, so handing it work that doesn't need it is pure waste. scoping
#and thesis building are mostly careful instruction following, which the small model does
#fine. synthesis has to weigh two arguments against each other and decide which the
#evidence supports, and that is where a wrong call is both most likely and most visible.
#
#which roles get the big model is a setting rather than a hardcoded list on purpose.
#"does the cheap model hold its score on extraction, and where exactly does it break" is
#a Phase 6 experiment, and it should be a config change and a rerun of the eval suite,
#not a code edit.
def model_for(role: AgentRole) -> str:
    settings = get_settings()
    strong = {name.strip() for name in settings.llm_strong_roles.split(",") if name.strip()}
    return settings.llm_strong_model if role.value in strong else settings.llm_fast_model


#whether this agent's calls may be served from the semantic cache. same reasoning as
#above, kept in config so it can be turned off per role without touching code.
def is_cacheable(role: AgentRole) -> bool:
    settings = get_settings()
    if not settings.semantic_cache_enabled:
        return False
    allowed = {name.strip() for name in settings.semantic_cache_roles.split(",") if name.strip()}
    return role.value in allowed
