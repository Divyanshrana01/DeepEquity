from __future__ import annotations

from typing import Any

from deepequity.core.logging import get_logger

logger = get_logger("deepequity.mcp.transcripts")


# Main tool: earnings-call transcripts. Deliberately a stub for now, good free transcript
# sources are scarce and the paid ones need a key we haven't picked yet. It returns the
# real response shape (status/ticker/quarter/segments) with a not_implemented status, so
# agents and the ingestion worker can code against the final schema today and the only
# thing that changes later is swapping this body for a real fetch.
async def fetch_transcript(ticker: str, quarter: str) -> dict[str, Any]:
    ticker = ticker.strip().upper()
    quarter = quarter.strip().upper()

    logger.info("transcript_stub_called", ticker=ticker, quarter=quarter)

    return {
        "status": "not_implemented",
        "message": (
            "Transcript retrieval is not wired up yet. A data source has not been "
            "chosen, this tool returns the final schema so callers can build against it."
        ),
        "ticker": ticker,
        "quarter": quarter,
        # segments is the shape a real transcript will fill: one entry per speaker turn.
        "segments": [],
    }
