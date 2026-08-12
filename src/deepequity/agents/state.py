from __future__ import annotations

from typing import Annotated, TypedDict

from deepequity.agents.memory import MemoryEntry
from deepequity.agents.schemas import AgentCost, ResearchNote, ResearchScope, Thesis
from deepequity.retrieval.models import RetrievedChunk


#adds lists together instead of replacing them, so a second debate round adds evidence
#to what's already there rather than wiping it. langgraph uses these annotations to work
#out how to merge each field when a node returns an update.
def append_chunks(
    current: list[RetrievedChunk], incoming: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    seen = {chunk.child_chunk_id for chunk in current}
    return current + [chunk for chunk in incoming if chunk.child_chunk_id not in seen]


def append_theses(current: list[Thesis], incoming: list[Thesis]) -> list[Thesis]:
    return current + incoming


def add_ints(current: int, incoming: int) -> int:
    return current + incoming


def append_costs(current: list[AgentCost], incoming: list[AgentCost]) -> list[AgentCost]:
    return current + incoming


#Everything flowing through the graph. One typed object rather than agents passing
#arguments around, which means any node can see the whole picture and the entire run can
#be checkpointed and resumed by saving one thing.
class ResearchState(TypedDict, total=False):
    #what was asked
    ticker: str

    #the id this run is checkpointed under. carried in the state rather than only in the
    #graph config because the synthesis step writes the finished note into long-term
    #memory keyed by it, and a node can only see the state.
    run_id: str

    #what we concluded about this company on earlier runs. recalled once, before planning,
    #and kept in the state so the trace shows what the planner was actually working from.
    memories: list[MemoryEntry]

    #what the planner decided before any evidence was gathered
    scope: ResearchScope | None

    #evidence gathered so far. accumulates across rounds rather than being replaced,
    #so a second round builds on the first instead of starting over.
    evidence: Annotated[list[RetrievedChunk], append_chunks]

    #both sides' arguments. a list rather than two fields because a second round adds
    #another pair, and keeping the history is what lets the synthesis step see whether
    #the debate actually moved.
    theses: Annotated[list[Thesis], append_theses]

    #the finished note, set once at the end
    note: ResearchNote | None

    #how many rounds of debate have run. the ceiling on this is what stops the graph
    #looping forever.
    #
    #the reducer is load bearing. without it langgraph replaces this value rather than
    #adding to it, so a debate node returning 1 each round leaves the count stuck at 1,
    #the ceiling never trips, and the graph loops until the recursion limit kills it.
    #that's the exact runaway the ceiling exists to prevent, and it fails silently.
    round_count: Annotated[int, add_ints]

    #running total, so the token budget can be enforced mid-run rather than discovered
    #afterwards
    tokens_used: Annotated[int, add_ints]

    #one entry per model call, with the model, the tokens and the money. the running total
    #above answers "are we near the ceiling", this answers "where did it all go", and the
    #second question is the one you need when a run costs more than you expected.
    costs: Annotated[list[AgentCost], append_costs]

    #why the run finished: completed, or which limit stopped it. worth recording, a note
    #produced after hitting the budget deserves to be read differently from one that
    #finished properly.
    stop_reason: str

    #anything that went wrong but didn't kill the run, surfaced rather than swallowed
    errors: Annotated[list[str], lambda a, b: a + b]
