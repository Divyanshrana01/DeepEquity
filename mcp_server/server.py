from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from deepequity.core.config import get_settings
from deepequity.core.logging import configure_logging
from mcp_server.tools.filings import fetch_filing as _fetch_filing
from mcp_server.tools.news import fetch_news as _fetch_news
from mcp_server.tools.prices import fetch_price_history as _fetch_price_history
from mcp_server.tools.transcripts import fetch_transcript as _fetch_transcript

# The custom MCP server. Agents in later phases connect to this over HTTP and call the
# four tools below instead of hitting SEC/yfinance/NewsAPI directly. stateless_http=True
# means each tool call is independent with no session to track, which is all we need.
mcp = FastMCP("deepequity-mcp", stateless_http=True)


# Each tool is registered as a thin async wrapper around the real implementation in
# tools/. Keeping the logic in tools/ (not inline here) means we can unit-test each one
# without spinning up the whole MCP server.
@mcp.tool()
async def fetch_filing(ticker: str, form_type: str = "10-K") -> dict[str, Any]:
    """Fetch the most recent SEC filing of a given form type (e.g. 10-K, 10-Q, 8-K)
    for a ticker. Returns filing metadata and a slice of the document text."""
    return await _fetch_filing(ticker, form_type)


@mcp.tool()
async def fetch_transcript(ticker: str, quarter: str) -> dict[str, Any]:
    """Fetch an earnings-call transcript for a ticker and quarter (e.g. Q1-2025).
    Currently a stub returning the final schema, no data source wired yet."""
    return await _fetch_transcript(ticker, quarter)


@mcp.tool()
async def fetch_price_history(ticker: str, period: str = "6mo") -> dict[str, Any]:
    """Fetch daily price history for a ticker over a period (e.g. 1mo, 6mo, 1y).
    Returns summary stats over the window plus the most recent daily bars."""
    return await _fetch_price_history(ticker, period)


@mcp.tool()
async def fetch_news(ticker: str, days: int = 7) -> dict[str, Any]:
    """Fetch recent news articles about a ticker over the last N days via NewsAPI.
    Returns an empty list with a not_configured status if no API key is set."""
    return await _fetch_news(ticker, days)


# Plain liveness check for docker/compose, same idea as the api's /health. Lives outside
# the MCP protocol so a simple curl or healthcheck can confirm the container is up.
async def health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


# Build the ASGI app uvicorn serves. The MCP protocol itself is mounted at /mcp by the
# SDK, we just bolt the /health route onto the same Starlette app.
configure_logging(get_settings().log_level)
app = mcp.streamable_http_app()
app.router.routes.append(Route("/health", health, methods=["GET"]))
