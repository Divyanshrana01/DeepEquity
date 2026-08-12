from __future__ import annotations

import anyio

from deepequity.agents.llm import complete_structured
from deepequity.agents.retriever import format_evidence, select_for_prompt
from deepequity.agents.schemas import Stance, Thesis
from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger
from deepequity.prompts.loader import load
from deepequity.retrieval.models import RetrievedChunk

logger = get_logger("deepequity.agents.debate")


#strips out citations pointing at passages the agent was never given.
#
#this is the hallucination guard, and it matters more than it looks. a model under
#pressure to cite everything will occasionally cite a plausible-looking id that doesn't
#exist, and an invented citation is worse than a missing one, because it looks checkable
#and isn't. anyone skimming sees a reference and assumes someone verified it.
#
#we drop the bad citation rather than the whole claim, then let the claim stand or fall on
#what's left. a claim with every citation stripped becomes visibly unsupported, which is
#exactly what it is, and the count gets logged so hallucination rate is measurable rather
#than assumed.
def strip_invalid_citations(thesis: Thesis, valid_chunk_ids: set[int]) -> tuple[Thesis, int]:
    removed = 0
    cleaned_claims = []

    for claim in thesis.claims:
        good = []
        already_cited: set[int] = set()
        for citation in claim.citations:
            if citation.chunk_id not in valid_chunk_ids:
                removed += 1
                continue
            #the same passage cited three times on one claim is one piece of evidence,
            #not three. left in, it makes a claim look better supported than it is,
            #which is the exact impression the citation count exists to prevent.
            if citation.chunk_id in already_cited:
                continue
            already_cited.add(citation.chunk_id)
            good.append(citation)

        cleaned_claims.append(claim.model_copy(update={"citations": good}))

    return thesis.model_copy(update={"claims": cleaned_claims}), removed


#runs one side of the argument.
#
#bull and bear see exactly the same evidence and differ only in their mandate. that's the
#whole design: it isolates the effect of the framing, so a difference between the two
#theses comes from how the evidence was read rather than from one side having been handed
#better material.
async def build_thesis(
    stance: Stance,
    ticker: str,
    evidence: list[RetrievedChunk],
    focus: str,
    previous: Thesis | None = None,
) -> tuple[Thesis, int]:
    settings = get_settings()
    prompt = load("bull" if stance is Stance.BULL else "bear")

    #only the strongest few passages are sent, so work out which ones those are here and
    #validate citations against exactly that set. checking against the whole retrieved
    #pool would be too lenient: it would accept an id the agent was never shown, which is
    #precisely the invented citation we're trying to catch.
    shown = select_for_prompt(evidence)

    user_parts = [
        f"Ticker: {ticker}",
        f"Research focus: {focus}",
        "",
        "Evidence:",
        format_evidence(evidence),
    ]

    #on a second round the agent sees what it said last time, so it can build on the
    #argument instead of writing the same thesis again from scratch
    if previous is not None:
        user_parts += [
            "",
            "Your previous thesis on this company:",
            previous.summary,
            "",
            "New evidence has been gathered since. Revise your case: strengthen claims "
            "the new evidence supports, drop any it undermines, and do not simply repeat "
            "what you said before.",
        ]

    response = await complete_structured(
        system_prompt=prompt.text,
        user_prompt="\n".join(user_parts),
        schema=Thesis,
        model=settings.llm_fast_model,
    )

    thesis = response.parsed
    #the model is asked for a stance and given a schema that constrains it, but the
    #mandate is what actually matters here, so make sure the label matches the seat the
    #agent was sitting in rather than trusting it to have set the field correctly
    if thesis.stance is not stance:
        thesis = thesis.model_copy(update={"stance": stance})

    valid_ids = {chunk.child_chunk_id for chunk in shown}
    thesis, dropped = strip_invalid_citations(thesis, valid_ids)

    supported = sum(1 for claim in thesis.claims if claim.is_supported)
    logger.info(
        "thesis_built",
        stance=stance,
        ticker=ticker,
        prompt_id=prompt.id,
        claims=len(thesis.claims),
        supported_claims=supported,
        invented_citations_dropped=dropped,
        gaps=len(thesis.evidence_gaps),
        tokens=response.usage.total,
    )
    return thesis, response.usage.total


#runs both sides at once. they don't read each other's output within a round, which is
#deliberate, it keeps each case built from the evidence rather than shaped as a rebuttal.
#reconciling them is the synthesis agent's job.
async def run_debate_round(
    ticker: str,
    evidence: list[RetrievedChunk],
    focus: str,
    previous: list[Thesis] | None = None,
) -> tuple[list[Thesis], int]:
    prior = {thesis.stance: thesis for thesis in (previous or [])}
    results: dict[Stance, tuple[Thesis, int]] = {}

    async def run_side(stance: Stance) -> None:
        results[stance] = await build_thesis(
            stance, ticker, evidence, focus, previous=prior.get(stance)
        )

    if get_settings().debate_concurrent:
        async with anyio.create_task_group() as tg:
            tg.start_soon(run_side, Stance.BULL)
            tg.start_soon(run_side, Stance.BEAR)
    else:
        #one after the other, because two requests of this size together exceed the free
        #tier's per-minute allowance and both get rejected. slower, but it finishes.
        await run_side(Stance.BULL)
        await run_side(Stance.BEAR)

    #stable order, bull then bear, so downstream code and the trace don't depend on which
    #request happened to finish first
    theses = [results[Stance.BULL][0], results[Stance.BEAR][0]]
    tokens = results[Stance.BULL][1] + results[Stance.BEAR][1]
    return theses, tokens
