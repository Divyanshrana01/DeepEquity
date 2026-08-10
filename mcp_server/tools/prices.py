from __future__ import annotations

from typing import Any

import anyio
import yfinance as yf  # type: ignore[import-untyped]

from deepequity.core.logging import get_logger

logger = get_logger("deepequity.mcp.prices")

# How many of the most recent daily bars to hand back. The full history for a long
# period can be hundreds of rows, which is noise for an agent, so we return summary
# stats over the whole window plus just the tail for a recent-trend feel.
_RECENT_BARS = 10


# yfinance is a plain blocking library (it does sync HTTP under the hood), so we can't
# just await it. This does the actual fetch and lives on its own so we can hand it to
# a worker thread and keep the async event loop free.
def _fetch_history_sync(ticker: str, period: str) -> list[dict[str, Any]]:
    data = yf.Ticker(ticker).history(period=period)
    bars: list[dict[str, Any]] = []
    for index, row in data.iterrows():
        bars.append(
            {
                "date": index.date().isoformat(),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            }
        )
    return bars


# Main tool: pulls daily price history for a ticker over a period (like "1mo", "6mo",
# "1y") and boils it down to something an agent can actually reason about, the start
# and end price, the move over the window, the high/low, and the most recent few bars.
async def fetch_price_history(ticker: str, period: str = "6mo") -> dict[str, Any]:
    ticker = ticker.strip().upper()
    period = period.strip().lower()

    # Run the blocking yfinance call in a thread so we don't stall the event loop.
    bars = await anyio.to_thread.run_sync(_fetch_history_sync, ticker, period)

    if not bars:
        return {
            "status": "error",
            "error": f"No price data for {ticker} over period '{period}'",
        }

    first_close = bars[0]["close"]
    last_close = bars[-1]["close"]
    # Guard against a zero first close so we never divide by zero on a bad feed.
    pct_change = (
        round((last_close - first_close) / first_close * 100, 2) if first_close else None
    )

    logger.info("fetched_prices", ticker=ticker, period=period, bars=len(bars))

    return {
        "status": "ok",
        "ticker": ticker,
        "period": period,
        "start_date": bars[0]["date"],
        "end_date": bars[-1]["date"],
        "start_close": first_close,
        "end_close": last_close,
        "pct_change": pct_change,
        "period_high": max(bar["high"] for bar in bars),
        "period_low": min(bar["low"] for bar in bars),
        "recent_bars": bars[-_RECENT_BARS:],
    }
