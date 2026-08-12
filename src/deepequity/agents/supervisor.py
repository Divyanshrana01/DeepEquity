from __future__ import annotations

from enum import StrEnum

from deepequity.agents.state import ResearchState
from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger

logger = get_logger("deepequity.agents.supervisor")


#where the graph goes next
class Route(StrEnum):
    PLAN = "plan"
    RETRIEVE = "retrieve"
    DEBATE = "debate"
    SYNTHESISE = "synthesise"
    END = "end"


#why a run stopped, kept on the state so a note can be read in the light of how it
#finished. a note that ran out of budget halfway is not the same as a finished one.
class StopReason(StrEnum):
    COMPLETED = "completed"
    MAX_ROUNDS = "max_rounds_reached"
    TOKEN_BUDGET = "token_budget_exceeded"
    NO_EVIDENCE = "no_evidence_found"


#decides what happens next. this is the whole supervisor: a small routing decision made
#from the state, not an llm call.
#
#doing this in code rather than asking a model is deliberate. routing is a narrow choice
#with a right answer, and an llm would add latency, cost and the chance of a wrong turn
#for no gain. the model's judgement is worth paying for in the debate and the synthesis,
#not here.
def decide_next(state: ResearchState) -> Route:
    settings = get_settings()

    #budget first, before anything that would spend more. checking it after the work is
    #how you end up over budget and only then noticing.
    if state.get("tokens_used", 0) >= settings.max_tokens_per_run:
        logger.warning(
            "token_budget_exceeded",
            tokens_used=state.get("tokens_used", 0),
            limit=settings.max_tokens_per_run,
        )
        #if there's anything to work with, still write the note, a partial answer beats
        #nothing. otherwise stop.
        return Route.SYNTHESISE if state.get("theses") else Route.END

    if state.get("scope") is None:
        return Route.PLAN

    if not state.get("evidence"):
        #planned but nothing gathered yet
        if state.get("round_count", 0) == 0:
            return Route.RETRIEVE
        #retrieval ran and found nothing. another round would search the same corpus the
        #same way, so there's no reason to expect a different result.
        logger.warning("no_evidence_after_retrieval", ticker=state.get("ticker"))
        return Route.END

    if not state.get("theses"):
        return Route.DEBATE

    #both sides have argued. is another round worth it?
    if _needs_another_round(state):
        return Route.DEBATE

    return Route.SYNTHESISE


#whether the debate should go again.
#
#the ceiling is the important part. without it a model that keeps saying "I need more
#evidence" would loop until the money ran out. every extra round costs real tokens, so
#the limit is low and deliberate.
def _needs_another_round(state: ResearchState) -> bool:
    settings = get_settings()
    rounds = state.get("round_count", 0)

    if rounds >= settings.max_debate_rounds:
        logger.info("max_rounds_reached", rounds=rounds)
        return False

    #go again only if both sides said they were missing something. one side complaining
    #usually means that side is losing, not that the evidence is thin.
    theses = state.get("theses", [])
    recent = theses[-2:] if len(theses) >= 2 else theses
    both_have_gaps = len(recent) == 2 and all(thesis.evidence_gaps for thesis in recent)

    if both_have_gaps:
        logger.info("another_round_warranted", rounds=rounds)
    return both_have_gaps


#works out why a run ended, for the record on the finished state
def stop_reason_for(state: ResearchState) -> StopReason:
    settings = get_settings()

    if state.get("tokens_used", 0) >= settings.max_tokens_per_run:
        return StopReason.TOKEN_BUDGET
    if not state.get("evidence"):
        return StopReason.NO_EVIDENCE
    if state.get("round_count", 0) >= settings.max_debate_rounds:
        return StopReason.MAX_ROUNDS
    return StopReason.COMPLETED
