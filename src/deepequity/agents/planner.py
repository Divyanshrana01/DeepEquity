from __future__ import annotations

from deepequity.agents.llm import complete_structured
from deepequity.agents.memory import MemoryEntry, format_memory
from deepequity.agents.routing import AgentRole
from deepequity.agents.schemas import AgentCost, ResearchScope
from deepequity.core.logging import get_logger
from deepequity.prompts.loader import load

logger = get_logger("deepequity.agents.planner")


#decides what the research should look for, before any evidence is gathered.
#
#this runs first for a reason. without it the system is a script: every ticker gets the
#same searches regardless of whether the interesting thing about the company is its
#margins, a lawsuit, or where it manufactures. the planner is what makes it reason about
#its own research rather than just execute a fixed recipe.
#
#it uses the small model. scoping is mostly instruction following, and the expensive
#model is worth saving for the synthesis where a wrong call actually costs something.
#
#past notes on the same company are handed in when we have them. this is the point where
#long-term memory earns its keep: the second run on a ticker shouldn't re-derive what the
#first one already established, it should go looking at what changed and at what the last
#run admitted it couldn't answer.
async def plan_research(
    ticker: str, memories: list[MemoryEntry] | None = None
) -> tuple[ResearchScope, AgentCost]:
    prompt = load("planner")

    user_parts = [
        f"Ticker: {ticker}",
        "",
        "Plan the research. Decide what actually matters for this company and write "
        "the searches that would surface evidence on both sides of the argument.",
    ]

    recalled = format_memory(memories or [])
    if recalled:
        user_parts += ["", recalled]

    response = await complete_structured(
        system_prompt=prompt.text,
        user_prompt="\n".join(user_parts),
        schema=ResearchScope,
        role=AgentRole.PLANNER,
    )

    scope = response.parsed
    logger.info(
        "research_planned",
        ticker=ticker,
        prompt_id=prompt.id,
        focus=scope.focus[:120],
        queries=len(scope.search_queries),
        memories_used=len(memories or []),
        tokens=response.usage.total,
        cost_usd=response.cost_usd,
        cached=response.cached,
    )
    return scope, response.cost_record()
