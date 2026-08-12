from __future__ import annotations

from deepequity.agents.llm import complete_structured
from deepequity.agents.schemas import ResearchScope
from deepequity.core.config import get_settings
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
async def plan_research(ticker: str) -> tuple[ResearchScope, int]:
    settings = get_settings()
    prompt = load("planner")

    response = await complete_structured(
        system_prompt=prompt.text,
        user_prompt=(
            f"Ticker: {ticker}\n\n"
            "Plan the research. Decide what actually matters for this company and write "
            "the searches that would surface evidence on both sides of the argument."
        ),
        schema=ResearchScope,
        model=settings.llm_fast_model,
    )

    scope = response.parsed
    logger.info(
        "research_planned",
        ticker=ticker,
        prompt_id=prompt.id,
        focus=scope.focus[:120],
        queries=len(scope.search_queries),
        tokens=response.usage.total,
    )
    return scope, response.usage.total
