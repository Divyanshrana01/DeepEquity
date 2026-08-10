from __future__ import annotations

from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger

logger = get_logger("deepequity.ingestion.mcp_client")


#Calls a tool on our MCP server and hands back the structured result. We open a fresh
#session per call rather than holding one open, ingestion calls are infrequent and a
#long-lived session would need reconnect handling for no real gain.
async def call_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    url = get_settings().mcp_server_url
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

    #structuredContent is the parsed dict our tools return. If it's missing something
    #went wrong at the protocol level, so fail loudly rather than returning junk.
    if result.structuredContent is None:
        raise RuntimeError(f"MCP tool {tool_name} returned no structured content")
    return dict(result.structuredContent)


#Thin wrapper for the one tool ingestion needs today. Keeps the tool name and argument
#shape in one place instead of scattered through the api and worker.
async def fetch_filing(ticker: str, form_type: str) -> dict[str, Any]:
    logger.info("mcp_fetch_filing", ticker=ticker, form_type=form_type)
    return await call_tool("fetch_filing", {"ticker": ticker, "form_type": form_type})
