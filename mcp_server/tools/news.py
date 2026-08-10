from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger

logger = get_logger("deepequity.mcp.news")

_NEWS_URL = "https://newsapi.org/v2/everything"

# Cap how many articles we return. The agent wants a feel for recent coverage, not a
# hundred near-duplicate headlines.
_MAX_ARTICLES = 15


# Main tool: pulls recent news articles about a ticker over the last N days via
# NewsAPI.org. If no API key is configured it says so plainly instead of erroring out,
# so the rest of the system keeps working while news stays optional.
async def fetch_news(ticker: str, days: int = 7) -> dict[str, Any]:
    ticker = ticker.strip().upper()
    settings = get_settings()

    if not settings.news_api_key:
        return {
            "status": "not_configured",
            "message": "NEWS_API_KEY is not set, news retrieval is disabled.",
            "ticker": ticker,
            "articles": [],
        }

    # NewsAPI wants a from-date, so we work backwards from today by the requested window.
    from_date = (datetime.now(UTC) - timedelta(days=days)).date().isoformat()
    params: dict[str, str | int] = {
        "q": ticker,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": _MAX_ARTICLES,
        "apiKey": settings.news_api_key,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(_NEWS_URL, params=params)
        # NewsAPI returns a json body with its own error message on a 4xx, so surface
        # that instead of a bare status code, it's far more useful when debugging.
        if resp.status_code != 200:
            detail = resp.json().get("message", resp.text)
            return {"status": "error", "error": detail, "ticker": ticker, "articles": []}
        payload = resp.json()

    # Keep only the fields an agent actually cites, drop the rest of NewsAPI's envelope.
    articles = [
        {
            "title": item.get("title"),
            "source": (item.get("source") or {}).get("name"),
            "published_at": item.get("publishedAt"),
            "url": item.get("url"),
            "description": item.get("description"),
        }
        for item in payload.get("articles", [])
    ]

    logger.info("fetched_news", ticker=ticker, days=days, count=len(articles))

    return {
        "status": "ok",
        "ticker": ticker,
        "days": days,
        "article_count": len(articles),
        "articles": articles,
    }
