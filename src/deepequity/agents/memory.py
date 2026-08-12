from __future__ import annotations

from datetime import datetime

from psycopg.rows import dict_row
from pydantic import BaseModel

from deepequity.agents.schemas import ResearchNote
from deepequity.core.config import get_settings
from deepequity.core.db import get_pool
from deepequity.core.logging import get_logger
from deepequity.ingestion.embeddings import embed_texts

logger = get_logger("deepequity.agents.memory")


#one thing we concluded about a company on some earlier run
class MemoryEntry(BaseModel):
    run_id: str
    ticker: str
    focus: str
    summary: str
    key_risks: list[str] = []
    confidence: float | None = None
    created_at: datetime
    #how close this past note is to what the current run is asking about. recall is
    #semantic rather than just "the last three", so a run about a lawsuit surfaces the
    #previous lawsuit note ahead of last quarter's margin note.
    similarity: float = 0.0

    #rough age in days, which is what actually decides whether a past conclusion is worth
    #anything. a note from last week about a quarterly filing is useful, the same note
    #from two years ago is history.
    def age_days(self, now: datetime | None = None) -> int:
        reference = now or datetime.now(self.created_at.tzinfo)
        return max((reference - self.created_at).days, 0)


#pgvector wants a literal like '[0.1,0.2]', a python list will not cast
def _to_vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"


#the text we embed for a note. the focus goes in alongside the summary because recall
#matches on what a run was about, and the summary alone often describes the answer
#without ever naming the question.
def _memory_text(focus: str, note: ResearchNote) -> str:
    risks = " ".join(note.key_risks)
    return f"{focus}\n{note.summary}\n{risks}".strip()


#writes a finished note into long-term memory.
#
#this happens after synthesis and is deliberately not allowed to break the run. the note
#has already been produced and returned to the caller by the time anything here matters,
#so a database problem at this point should cost us the memory, not the research.
async def remember(ticker: str, run_id: str, focus: str, note: ResearchNote) -> bool:
    if not get_settings().memory_enabled:
        return False

    try:
        vectors = await embed_texts([_memory_text(focus, note)])
        if not vectors:
            return False

        pool = await get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO research_memory
                    (ticker, run_id, focus, summary, key_risks, confidence, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (run_id) DO NOTHING
                """,
                (
                    ticker.upper(),
                    run_id,
                    focus[:500],
                    note.summary,
                    "\n".join(note.key_risks),
                    note.confidence.overall,
                    _to_vector_literal(vectors[0]),
                ),
            )
    except Exception as exc:  # noqa: BLE001 - memory is an optimisation, never a blocker
        logger.warning("memory_write_failed", ticker=ticker, run_id=run_id, error=str(exc))
        return False

    logger.info("memory_written", ticker=ticker, run_id=run_id)
    return True


#pulls up what we concluded about this company before.
#
#scoped to the ticker first and ranked by meaning second. the ticker filter is not an
#optimisation, it is correctness: a note about one company's supply chain must never be
#recalled as background for another company just because the wording lines up.
async def recall(
    ticker: str, query: str, limit: int | None = None
) -> list[MemoryEntry]:
    settings = get_settings()
    if not settings.memory_enabled:
        return []

    limit = limit or settings.memory_recall_limit

    try:
        vectors = await embed_texts([query])
        if not vectors:
            return []

        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT run_id, ticker, focus, summary, key_risks, confidence,
                           created_at,
                           1 - (embedding <=> %(vector)s::vector) AS similarity
                    FROM research_memory
                    WHERE ticker = %(ticker)s AND embedding IS NOT NULL
                    ORDER BY embedding <=> %(vector)s::vector
                    LIMIT %(limit)s
                    """,
                    {
                        "vector": _to_vector_literal(vectors[0]),
                        "ticker": ticker.upper(),
                        "limit": limit,
                    },
                )
                rows = await cur.fetchall()
    except Exception as exc:  # noqa: BLE001 - same as above, a run works without memory
        logger.warning("memory_recall_failed", ticker=ticker, error=str(exc))
        return []

    entries = [
        MemoryEntry(
            run_id=row["run_id"],
            ticker=row["ticker"],
            focus=row["focus"],
            summary=row["summary"],
            key_risks=[line for line in (row["key_risks"] or "").split("\n") if line],
            confidence=row["confidence"],
            created_at=row["created_at"],
            similarity=round(float(row["similarity"]), 4),
        )
        for row in rows
    ]

    logger.info("memory_recalled", ticker=ticker, entries=len(entries))
    return entries


#renders past notes into the block the planner reads.
#
#the framing matters more than the formatting. handed a previous conclusion with no
#instruction, a model agrees with it, and the system quietly stops doing research and
#starts repeating itself with growing confidence. so the text says plainly what these are
#for: work out what has changed and what was left unresolved, not what to conclude. the
#age and the old confidence are printed for the same reason, an eighteen month old note
#that was unsure at the time should not steer anything.
def format_memory(entries: list[MemoryEntry]) -> str:
    if not entries:
        return ""

    lines = [
        "Previous research this system produced on this company. These are our own past "
        "conclusions, not source documents. Do not treat them as evidence and do not set "
        "out to confirm them. Use them to work out what has changed since, what was left "
        "unresolved, and what is worth checking again.",
        "",
    ]
    for entry in entries:
        confidence = f"{entry.confidence:.2f}" if entry.confidence is not None else "unknown"
        lines.append(
            f"- {entry.age_days()} days ago (confidence {confidence}), "
            f"focus was: {entry.focus}"
        )
        lines.append(f"  concluded: {entry.summary[:600]}")
        if entry.key_risks:
            lines.append(f"  risks flagged: {'; '.join(entry.key_risks[:4])}")
    return "\n".join(lines)
