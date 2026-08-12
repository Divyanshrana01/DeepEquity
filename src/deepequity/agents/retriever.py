from __future__ import annotations

import anyio

from deepequity.agents.schemas import ResearchScope
from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger
from deepequity.retrieval.models import RetrievedChunk
from deepequity.retrieval.search import hybrid_search

logger = get_logger("deepequity.agents.retriever")

#how many passages each of the planner's queries contributes. deliberately small per
#query: five or six queries at six passages each is already a lot of text for the debate
#agents to hold, and past a point more evidence makes their arguments worse rather than
#better because the relevant passages get diluted.
_PER_QUERY = 6


#runs every search the planner asked for and collects the results into one evidence pool.
#
#the queries run at the same time rather than one after another. they're independent, and
#six sequential searches with a reranker each would be most of a minute of waiting for no
#reason.
async def gather_evidence(
    scope: ResearchScope, ticker: str, already_have: list[RetrievedChunk] | None = None
) -> list[RetrievedChunk]:
    seen = {chunk.child_chunk_id for chunk in (already_have or [])}
    collected: list[list[RetrievedChunk]] = [[] for _ in scope.search_queries]

    async def run_one(index: int, query: str) -> None:
        result = await hybrid_search(query, top_k=_PER_QUERY, ticker=ticker)
        collected[index] = result.chunks

    async with anyio.create_task_group() as tg:
        for index, query in enumerate(scope.search_queries):
            tg.start_soon(run_one, index, query)

    #different queries pull overlapping passages, which is a good sign for relevance but
    #a waste of context if we send the same text twice. keep the first occurrence, which
    #is from the earlier and usually more central query.
    evidence: list[RetrievedChunk] = []
    for chunks in collected:
        for chunk in chunks:
            if chunk.child_chunk_id not in seen:
                seen.add(chunk.child_chunk_id)
                evidence.append(chunk)

    logger.info(
        "evidence_gathered",
        ticker=ticker,
        queries=len(scope.search_queries),
        unique_chunks=len(evidence),
        duplicates_dropped=sum(len(c) for c in collected) - len(evidence),
    )
    return evidence


#picks which passages actually go to the agents, best scoring first.
#
#we don't send everything retrieved. thirty-five passages of parent text is both more
#than the free tier allows in one request and more than helps: the handful that really
#answer the question get diluted by the ones that merely matched, and the arguments come
#out worse. taking the strongest few is the better answer on both counts.
def select_for_prompt(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    settings = get_settings()
    ranked = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
    return ranked[: settings.max_evidence_chunks]


#renders the evidence into the text the agents actually read.
#
#the chunk id is printed on every passage because that's what the agents cite. the parent
#text is used rather than the matched child: the child is what scored well, but it's a
#sentence or two, and an agent asked to argue from it has no surrounding context. this is
#the payoff from parent-child chunking, precision when searching, context when reading.
def format_evidence(chunks: list[RetrievedChunk], select: bool = True) -> str:
    if not chunks:
        return "(no evidence found)"

    settings = get_settings()
    chosen = select_for_prompt(chunks) if select else chunks

    blocks: list[str] = []
    for chunk in chosen:
        body = chunk.parent_text or chunk.child_text
        #trim from the end, the match was usually nearer the start so the tail is the
        #least useful part to keep
        limit = settings.max_evidence_chars_per_chunk
        if len(body) > limit:
            body = body[:limit].rstrip() + " [...]"
        blocks.append(
            f"[chunk {chunk.child_chunk_id}] ({chunk.ticker} {chunk.doc_type})\n{body}"
        )
    return "\n\n---\n\n".join(blocks)
