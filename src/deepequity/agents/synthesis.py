from __future__ import annotations

from deepequity.agents.llm import complete_structured
from deepequity.agents.schemas import (
    Claim,
    ConfidenceBreakdown,
    ResearchNote,
    Thesis,
)
from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger
from deepequity.prompts.loader import load
from deepequity.retrieval.models import RetrievedChunk

logger = get_logger("deepequity.agents.synthesis")


#turns a thesis into the text the synthesis agent reads. citations are printed alongside
#each claim so the reconciliation can carry them through rather than having to guess
#where a point came from.
def _format_thesis(thesis: Thesis) -> str:
    lines = [f"### {thesis.stance.upper()} CASE", thesis.summary, "", "Claims:"]
    for claim in thesis.claims:
        cites = ", ".join(str(c.chunk_id) for c in claim.citations) or "UNSUPPORTED"
        lines.append(f"- [confidence {claim.confidence:.2f}] [chunks: {cites}] {claim.statement}")
    if thesis.evidence_gaps:
        lines += ["", "Gaps this analyst flagged:"]
        lines += [f"- {gap}" for gap in thesis.evidence_gaps]
    return "\n".join(lines)


#some of the confidence dimensions are facts about the note, not opinions about it, and
#those we work out ourselves instead of asking.
#
#the reason is simple: a model scoring its own work grades generously, and it tends to
#return five similar numbers rather than answering five different questions. source
#diversity in particular is just counting, there is no judgement in it, so measuring beats
#asking. we keep the model's scores for the two that genuinely need judgement and replace
#the two that don't.
def _measure_confidence(
    model_scores: ConfidenceBreakdown, claims: list[Claim], evidence: list[RetrievedChunk]
) -> ConfidenceBreakdown:
    supported = [claim for claim in claims if claim.is_supported]

    #how much of the note is actually backed by a passage. an unsupported claim is the
    #thing a reader most needs to know about, so it drags this down directly.
    if claims:
        supported_share = len(supported) / len(claims)
    else:
        supported_share = 0.0

    #how many distinct passages the note rests on. one passage doing all the work is
    #fragile even when the note reads well, and that's exactly what this is meant to
    #catch. measured against the evidence available rather than an arbitrary target.
    cited_chunks = {c.chunk_id for claim in supported for c in claim.citations}
    available = len({chunk.child_chunk_id for chunk in evidence}) or 1
    diversity = min(len(cited_chunks) / min(available, 6), 1.0)

    #blend the measured evidence strength with what the model said. the model can see
    #whether a passage really supports its claim, which counting can't, so neither number
    #alone is right. the lower of the two would be harsh, the average is fair.
    evidence_strength = (supported_share + model_scores.evidence_strength) / 2

    #overall must not outrun the evidence. a tidy, internally consistent note built on
    #nothing is still a note built on nothing, and this is the guard against a confident
    #sounding summary carrying a high score it hasn't earned.
    overall = min(model_scores.overall, evidence_strength + 0.1)

    return ConfidenceBreakdown(
        evidence_strength=round(evidence_strength, 3),
        reasoning_consistency=model_scores.reasoning_consistency,
        source_diversity=round(diversity, 3),
        data_freshness=model_scores.data_freshness,
        overall=round(max(overall, 0.0), 3),
    )


#reconciles the two theses into the final note.
#
#this is the one step that uses the big model. everything before it is scoping and
#extraction, where the job is mostly to follow instructions carefully. this step has to
#weigh two arguments against each other and decide which the evidence actually supports,
#and it's where an unforced error is most expensive because it's the part anyone reads.
async def synthesise(
    ticker: str, theses: list[Thesis], evidence: list[RetrievedChunk]
) -> tuple[ResearchNote, int]:
    settings = get_settings()
    prompt = load("synthesis")

    response = await complete_structured(
        system_prompt=prompt.text,
        user_prompt="\n\n".join(
            [f"Ticker: {ticker}", *[_format_thesis(thesis) for thesis in theses]]
            + ["Reconcile these into one note."]
        ),
        schema=ResearchNote,
        model=settings.llm_strong_model,
    )

    note = response.parsed

    #the ticker is ours to know, not the model's to decide
    if note.ticker != ticker:
        note = note.model_copy(update={"ticker": ticker})

    #same citation guard as the debate agents: the note may only cite passages that were
    #actually argued from. a chunk id appearing here that neither analyst used would be
    #invented at the last step, where it's least likely to be noticed.
    argued_ids = {
        citation.chunk_id
        for thesis in theses
        for claim in thesis.claims
        for citation in claim.citations
    }
    kept_claims = []
    dropped = 0
    for claim in note.supporting_claims:
        good = [c for c in claim.citations if c.chunk_id in argued_ids]
        dropped += len(claim.citations) - len(good)
        kept_claims.append(claim.model_copy(update={"citations": good}))

    note = note.model_copy(update={"supporting_claims": kept_claims})
    note = note.model_copy(
        update={"confidence": _measure_confidence(note.confidence, kept_claims, evidence)}
    )

    supported = sum(1 for claim in kept_claims if claim.is_supported)
    logger.info(
        "note_synthesised",
        ticker=ticker,
        prompt_id=prompt.id,
        model=settings.llm_strong_model,
        agreements=len(note.agreements),
        disagreements=len(note.disagreements),
        claims=len(kept_claims),
        supported_claims=supported,
        invented_citations_dropped=dropped,
        overall_confidence=note.confidence.overall,
        tokens=response.usage.total,
    )
    return note, response.usage.total
