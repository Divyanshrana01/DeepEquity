from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, StateGraph

from deepequity.agents.checkpoint import get_checkpointer
from deepequity.agents.debate import run_debate_round
from deepequity.agents.memory import recall, remember
from deepequity.agents.planner import plan_research
from deepequity.agents.retriever import gather_evidence
from deepequity.agents.run_context import set_current_run
from deepequity.agents.state import ResearchState
from deepequity.agents.supervisor import Route, decide_next, stop_reason_for
from deepequity.agents.synthesis import synthesise
from deepequity.core.logging import get_logger

logger = get_logger("deepequity.agents.graph")


#Each node does one job and returns only the fields it changed. langgraph merges those
#updates into the state using the reducers on ResearchState, which is what lets a second
#debate round add to the evidence rather than overwrite it.


async def plan_node(state: ResearchState) -> dict:
    ticker = state["ticker"]

    #memory is looked up here rather than in the planner itself so the recalled notes land
    #in the state. that keeps them in the checkpoint and in the trace, which matters:
    #if a plan looks odd, the first question is what it was told about previous runs.
    memories = await recall(ticker, query=f"{ticker} equity research thesis and risks")

    scope, cost = await plan_research(ticker, memories=memories)
    return {
        "scope": scope,
        "memories": memories,
        "tokens_used": cost.total_tokens,
        "costs": [cost],
    }


async def retrieve_node(state: ResearchState) -> dict:
    scope = state["scope"]
    if scope is None:
        return {"errors": ["retrieve ran without a scope"]}

    evidence = await gather_evidence(scope, state["ticker"], state.get("evidence", []))
    #retrieval costs no tokens, it's the embedding model and postgres, both local
    return {"evidence": evidence}


async def debate_node(state: ResearchState) -> dict:
    scope = state["scope"]
    focus = scope.focus if scope else state["ticker"]

    theses, costs = await run_debate_round(
        ticker=state["ticker"],
        evidence=state.get("evidence", []),
        focus=focus,
        #the previous round's theses, so each side revises its case rather than writing
        #the same one again
        previous=state.get("theses", [])[-2:] or None,
    )
    return {
        "theses": theses,
        "tokens_used": sum(cost.total_tokens for cost in costs),
        "costs": costs,
        "round_count": 1,
    }


async def synthesise_node(state: ResearchState) -> dict:
    ticker = state["ticker"]
    note, cost = await synthesise(
        ticker=ticker,
        theses=state.get("theses", []),
        evidence=state.get("evidence", []),
    )

    #the note goes into long-term memory as the last thing that happens, after it exists
    #and is about to be returned. remember() swallows its own failures, so a database
    #problem here loses us the memory and not the research.
    scope = state.get("scope")
    await remember(
        ticker=ticker,
        run_id=state.get("run_id", ""),
        focus=scope.focus if scope else ticker,
        note=note,
    )

    return {"note": note, "tokens_used": cost.total_tokens, "costs": [cost]}


#works out what a run is currently doing, from what's in the state. used for progress
#reporting, so someone polling a four minute run learns something more useful than that
#it hasn't finished.
def _current_stage(state: ResearchState) -> str:
    if state.get("note") is not None:
        return "complete"
    if state.get("theses"):
        return "debating" if decide_next(state) is Route.DEBATE else "synthesising"
    if state.get("evidence"):
        return "debating"
    if state.get("scope") is not None:
        return "retrieving"
    return "planning"


#the supervisor's decision, translated into the name of the next node. keeping the
#decision itself in supervisor.py means it can be tested on plain state objects without
#building a graph.
def route(state: ResearchState) -> str:
    decision = decide_next(state)
    logger.info(
        "supervisor_routed",
        decision=decision,
        round_count=state.get("round_count", 0),
        evidence=len(state.get("evidence", [])),
        theses=len(state.get("theses", [])),
        tokens_used=state.get("tokens_used", 0),
    )
    return END if decision is Route.END else decision.value


#Builds the graph.
#
#Every node routes back through the supervisor rather than pointing at whichever node
#comes next. That's what makes it a supervisor pattern instead of a pipeline with extra
#steps: no node decides what happens after it, so the round ceiling and the token budget
#are enforced in one place and can't be bypassed by a node handing straight off to
#another.
def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node(Route.PLAN.value, plan_node)
    graph.add_node(Route.RETRIEVE.value, retrieve_node)
    graph.add_node(Route.DEBATE.value, debate_node)
    graph.add_node(Route.SYNTHESISE.value, synthesise_node)

    graph.set_conditional_entry_point(
        route,
        {
            Route.PLAN.value: Route.PLAN.value,
            Route.RETRIEVE.value: Route.RETRIEVE.value,
            Route.DEBATE.value: Route.DEBATE.value,
            Route.SYNTHESISE.value: Route.SYNTHESISE.value,
            END: END,
        },
    )

    for node in (Route.PLAN.value, Route.RETRIEVE.value, Route.DEBATE.value):
        graph.add_conditional_edges(
            node,
            route,
            {
                Route.PLAN.value: Route.PLAN.value,
                Route.RETRIEVE.value: Route.RETRIEVE.value,
                Route.DEBATE.value: Route.DEBATE.value,
                Route.SYNTHESISE.value: Route.SYNTHESISE.value,
                END: END,
            },
        )

    #synthesis is the last thing that happens, there's nothing to decide after it
    graph.add_edge(Route.SYNTHESISE.value, END)

    return graph


_compiled = None


async def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile(checkpointer=await get_checkpointer())
    return _compiled


#runs a full research pass for one ticker.
#
#thread_id is what ties a run to its checkpoints. passing the run id means a run that
#died halfway can be re-invoked with the same id and langgraph will pick up from the last
#node that finished rather than starting over.
#
#on_stage is called as each node completes, so a caller polling the api sees which stage
#a four minute run is on instead of a silent wait.
async def run_research(
    ticker: str,
    run_id: str | None = None,
    on_stage: Callable[[str, ResearchState], Awaitable[None]] | None = None,
) -> ResearchState:
    graph = await get_compiled_graph()

    thread_id = run_id or str(uuid.uuid4())
    #everything below this point runs inside the run's context, which is how the semantic
    #cache knows not to serve a run its own earlier answers
    set_current_run(thread_id)

    initial: ResearchState = {
        "ticker": ticker.strip().upper(),
        "run_id": thread_id,
        "scope": None,
        "memories": [],
        "evidence": [],
        "theses": [],
        "note": None,
        "round_count": 0,
        "tokens_used": 0,
        "costs": [],
        "errors": [],
    }

    #recursion_limit is a backstop below the supervisor's own limits. if a routing bug
    #ever produced a cycle the supervisor didn't catch, this stops it rather than letting
    #it spin.
    #a fresh id when none is given, NOT one derived from the ticker. the thread id is what
    #langgraph resumes from, so deriving it from the ticker would make a second run for
    #the same company silently pick up the first run's checkpoint and skip straight to
    #synthesis instead of doing new research.
    config: dict[str, Any] = {
        "recursion_limit": 25,
        "configurable": {"thread_id": thread_id},
    }

    logger.info("research_started", ticker=initial["ticker"], run_id=run_id)

    if on_stage is None:
        final: ResearchState = await graph.ainvoke(initial, config)
    else:
        #streaming the node updates rather than awaiting the whole thing, so progress can
        #be reported as it happens. the accumulated state is rebuilt at the end.
        final = initial
        async for update in graph.astream(initial, config, stream_mode="values"):
            final = update
            stage = _current_stage(final)
            await on_stage(stage, final)

    #recorded here rather than inside a node, because a run can end without ever reaching
    #synthesis (no evidence found, or the budget gone before the debate). set in a node,
    #those are exactly the runs that would finish with no explanation of why they stopped.
    final["stop_reason"] = stop_reason_for(final)

    costs = final.get("costs", [])
    logger.info(
        "research_finished",
        ticker=initial["ticker"],
        rounds=final.get("round_count", 0),
        tokens=final.get("tokens_used", 0),
        cost_usd=round(sum(cost.cost_usd for cost in costs), 6),
        saved_usd=round(sum(cost.saved_usd for cost in costs), 6),
        cached_calls=sum(1 for cost in costs if cost.cached),
        llm_calls=len(costs),
        stop_reason=final.get("stop_reason"),
        has_note=final.get("note") is not None,
    )
    return final
