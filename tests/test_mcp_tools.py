from __future__ import annotations

import httpx
import pytest
import respx

import mcp_server.tools.filings as filings
from deepequity.core.config import get_settings
from mcp_server.tools.filings import fetch_filing
from mcp_server.tools.news import fetch_news
from mcp_server.tools.prices import fetch_price_history
from mcp_server.tools.transcripts import fetch_transcript


# The ticker->CIK map is cached at module level for the life of the process. Tests would
# leak that cache into each other, so we clear it before every test in this file.
@pytest.fixture(autouse=True)
def _clear_filing_cache() -> None:
    filings._ticker_cik_cache = None


# --- filings -------------------------------------------------------------------------


@respx.mock
async def test_fetch_filing_returns_latest_matching_form() -> None:
    respx.get(filings._TICKERS_URL).mock(
        return_value=httpx.Response(200, json={"0": {"cik_str": 320193, "ticker": "AAPL"}})
    )
    respx.get(filings._SUBMISSIONS_URL.format(cik=320193)).mock(
        return_value=httpx.Response(
            200,
            json={
                "filings": {
                    "recent": {
                        # 8-K comes first, so a naive "take the newest" would grab it, the
                        # tool must skip past it to the 10-K we actually asked for.
                        "form": ["8-K", "10-K"],
                        "accessionNumber": ["0000-24-000001", "0000320193-24-000123"],
                        "primaryDocument": ["ev.htm", "aapl-20240928.htm"],
                        "filingDate": ["2024-11-01", "2024-11-01"],
                        "reportDate": ["2024-10-30", "2024-09-28"],
                    }
                }
            },
        )
    )
    doc_url = filings._ARCHIVE_URL.format(
        cik=320193, accession_nodash="000032019324000123", doc="aapl-20240928.htm"
    )
    respx.get(doc_url).mock(return_value=httpx.Response(200, text="Item 1A. Risk Factors ..."))

    result = await fetch_filing("aapl", "10-K")

    assert result["status"] == "ok"
    assert result["form_type"] == "10-K"
    assert result["accession_number"] == "0000320193-24-000123"
    assert result["report_date"] == "2024-09-28"
    assert "Risk Factors" in result["text"]


@respx.mock
async def test_fetch_filing_unknown_ticker() -> None:
    respx.get(filings._TICKERS_URL).mock(
        return_value=httpx.Response(200, json={"0": {"cik_str": 320193, "ticker": "AAPL"}})
    )

    result = await fetch_filing("NOPE", "10-K")

    assert result["status"] == "error"
    assert "Unknown ticker" in result["error"]


@respx.mock
async def test_fetch_filing_missing_form_type() -> None:
    respx.get(filings._TICKERS_URL).mock(
        return_value=httpx.Response(200, json={"0": {"cik_str": 320193, "ticker": "AAPL"}})
    )
    respx.get(filings._SUBMISSIONS_URL.format(cik=320193)).mock(
        return_value=httpx.Response(
            200,
            json={
                "filings": {
                    "recent": {
                        "form": ["10-Q"],
                        "accessionNumber": ["0000320193-24-000200"],
                        "primaryDocument": ["q.htm"],
                        "filingDate": ["2024-08-01"],
                        "reportDate": ["2024-06-30"],
                    }
                }
            },
        )
    )

    result = await fetch_filing("AAPL", "10-K")

    assert result["status"] == "error"
    assert "No 10-K filing found" in result["error"]


# --- prices --------------------------------------------------------------------------


async def test_fetch_price_history_summarises(monkeypatch: pytest.MonkeyPatch) -> None:
    # yfinance is blocking and hits the network, so we swap the sync fetch for canned bars.
    fake_bars = [
        {"date": "2024-01-02", "open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1000},
        {"date": "2024-01-03", "open": 100.0, "high": 112.0, "low": 98.0,
         "close": 110.0, "volume": 1200},
    ]
    monkeypatch.setattr(
        "mcp_server.tools.prices._fetch_history_sync", lambda ticker, period: fake_bars
    )

    result = await fetch_price_history("AAPL", "1mo")

    assert result["status"] == "ok"
    assert result["start_close"] == 100.0
    assert result["end_close"] == 110.0
    assert result["pct_change"] == 10.0
    assert result["period_high"] == 112.0
    assert result["period_low"] == 98.0


async def test_fetch_price_history_no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mcp_server.tools.prices._fetch_history_sync", lambda ticker, period: []
    )

    result = await fetch_price_history("ZZZZ", "1mo")

    assert result["status"] == "error"


# --- news ----------------------------------------------------------------------------


async def test_fetch_news_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEWS_API_KEY", "")
    get_settings.cache_clear()

    result = await fetch_news("AAPL", 7)

    assert result["status"] == "not_configured"
    assert result["articles"] == []
    get_settings.cache_clear()


@respx.mock
async def test_fetch_news_returns_articles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEWS_API_KEY", "test-key")
    get_settings.cache_clear()

    respx.get("https://newsapi.org/v2/everything").mock(
        return_value=httpx.Response(
            200,
            json={
                "articles": [
                    {
                        "title": "Apple beats earnings",
                        "source": {"name": "Reuters"},
                        "publishedAt": "2024-11-01T12:00:00Z",
                        "url": "https://example.com/a",
                        "description": "Strong quarter.",
                    }
                ]
            },
        )
    )

    result = await fetch_news("AAPL", 7)

    assert result["status"] == "ok"
    assert result["article_count"] == 1
    assert result["articles"][0]["source"] == "Reuters"
    get_settings.cache_clear()


# --- transport security -------------------------------------------------------------


def test_allowed_hosts_parses_and_includes_docker_service() -> None:
    # Regression guard: with the docker service name missing from this list the MCP
    # server answers container-to-container calls with 421 Misdirected Request, which
    # is exactly what broke the first live ingest run.
    get_settings.cache_clear()
    hosts = get_settings().allowed_hosts()

    assert "mcp-server:8000" in hosts
    assert all(host == host.strip() for host in hosts)


def test_allowed_hosts_ignores_whitespace_and_blanks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "a:1, b:2 ,,")
    get_settings.cache_clear()

    assert get_settings().allowed_hosts() == ["a:1", "b:2"]
    get_settings.cache_clear()


# --- transcripts (stub) --------------------------------------------------------------


async def test_fetch_transcript_is_stubbed() -> None:
    result = await fetch_transcript("AAPL", "Q1-2025")

    assert result["status"] == "not_implemented"
    assert result["ticker"] == "AAPL"
    assert result["segments"] == []
