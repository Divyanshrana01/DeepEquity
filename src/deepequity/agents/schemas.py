from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

#Every agent returns one of these, never prose. Free text can't be checked, scored or
#tested, and there'd be no way to tell whether a claim came with evidence or the model
#simply sounded confident. A schema makes "did it cite anything" a property we can
#actually assert on.


class Stance(StrEnum):
    BULL = "bull"
    BEAR = "bear"


#What one agent's call cost. Kept per call rather than as a single total for the run,
#because "the run used 40k tokens" doesn't tell you where to look and "synthesis was 60%
#of the bill" does. It's also how the model routing gets checked: if the planner shows up
#on the expensive model, the routing is wrong and this is where it shows.
class AgentCost(BaseModel):
    agent: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    #served from the semantic cache. the tokens are still recorded so the saving can be
    #measured, but nothing was actually spent.
    cached: bool = False
    #what this call would have cost had the cache missed. zero on a real call.
    saved_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


#Points at the exact passage a claim rests on. This is what makes a note checkable: a
#reader can follow any sentence back to the text it came from, and we can verify after
#the fact that the passage really says what the agent said it does.
class Citation(BaseModel):
    chunk_id: int = Field(description="The child chunk this claim is drawn from")
    document_id: int
    #a short lift from the source, so a reader doesn't have to go and look it up to see
    #whether the claim is fair
    quote: str = Field(max_length=500, description="Short verbatim quote from the source")


#One argument, with the evidence attached to it rather than gathered loosely at the end.
#Tying evidence to the individual claim is deliberate, a note with a pile of sources at
#the bottom lets an unsupported sentence hide among the supported ones.
class Claim(BaseModel):
    statement: str = Field(max_length=600, description="A single specific claim")
    #the agent's own read of how well the evidence backs this particular point, which is
    #more useful than one confidence number for the whole thesis
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)

    #a claim nothing supports. we don't throw these away, we count them, because "how
    #much of this note is unsupported" is exactly what a reader needs to know.
    @property
    def is_supported(self) -> bool:
        return len(self.citations) > 0


#What Bull and Bear each produce. Same shape for both so the synthesis step can compare
#them like for like instead of parsing two different formats.
class Thesis(BaseModel):
    stance: Stance
    summary: str = Field(max_length=1500)
    claims: list[Claim] = Field(default_factory=list)
    #what the agent looked for and couldn't find. an honest gap is more useful than a
    #padded argument, and it gives the supervisor something concrete to decide another
    #round on.
    evidence_gaps: list[str] = Field(default_factory=list)


#The Planner's output: how to scope the research before any evidence is pulled. This is
#the step that separates a system reasoning about its own research from a fixed script,
#a question about a lawsuit and a question about margins should not search the same way.
class ResearchScope(BaseModel):
    #what the planner thinks the real question is
    focus: str = Field(max_length=500)
    #the searches to run. several, because one query rarely covers a thesis.
    search_queries: list[str] = Field(min_length=1, max_length=8)
    #which filing types matter here
    form_types: list[str] = Field(default_factory=lambda: ["10-K"])
    #why it scoped things this way, kept so the decision shows up in the trace and can be
    #argued with rather than taken on trust
    reasoning: str = Field(max_length=1000)


#Where the two theses agree, disagree, and where one side is simply better supported.
class Disagreement(BaseModel):
    topic: str = Field(max_length=300)
    bull_position: str = Field(max_length=600)
    bear_position: str = Field(max_length=600)
    #which side the evidence actually favours, or neither
    better_supported: str = Field(description="bull, bear, or neither")


#Confidence broken into its parts rather than collapsed to one number. A single score
#hides exactly what a reader needs: a note can be internally consistent and still rest
#on one stale document, and one number can't tell you that.
class ConfidenceBreakdown(BaseModel):
    evidence_strength: float = Field(ge=0.0, le=1.0, description="Amount and quality of support")
    reasoning_consistency: float = Field(ge=0.0, le=1.0, description="Internal consistency")
    source_diversity: float = Field(ge=0.0, le=1.0, description="Spread across documents")
    data_freshness: float = Field(ge=0.0, le=1.0, description="How recent the evidence is")
    overall: float = Field(ge=0.0, le=1.0, description="Weighted combination")


#The finished research note.
class ResearchNote(BaseModel):
    ticker: str
    summary: str = Field(max_length=3000)
    agreements: list[str] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    confidence: ConfidenceBreakdown
    #every claim carried through from both sides, so the citations survive into the final
    #output instead of being summarised away
    supporting_claims: list[Claim] = Field(default_factory=list)
    #not optional. this is a system that reads company filings and argues about them, it
    #is not advice, and the output should say so on its face.
    disclaimer: str = Field(
        default=(
            "This is an automated research summary generated from public filings. "
            "It is not financial advice and should not be relied on for investment decisions."
        )
    )
