"""Placeholder for the custom MCP server.

Phase 2 turns this into a real MCP server exposing four tools:
fetch_filing, fetch_transcript, fetch_price_history, fetch_news. For now
it's just a container with a health check, so docker-compose already has
the right shape and networking before the real tools exist.
"""

from fastapi import FastAPI

app = FastAPI(title="deepequity-mcp-server")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
