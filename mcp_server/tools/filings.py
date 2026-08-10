from __future__ import annotations

from typing import Any

import httpx

from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger

logger = get_logger("deepequity.mcp.filings")

# SEC endpoints. The ticker->CIK map is one big json file we fetch once and keep in
# memory. The submissions endpoint lists a company's recent filings by form type.
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{doc}"

# Filings can be megabytes of legal text. We cap what we hand back so a single tool call
# doesn't dump the whole thing into an agent's context, the ingestion worker pulls the
# full document separately when it's time to chunk and embed.
_MAX_TEXT_CHARS = 20_000

# Cache the ticker->CIK map for the life of the process so we don't refetch a 1MB file
# on every single filing lookup.
_ticker_cik_cache: dict[str, int] | None = None


# SEC blocks requests that don't identify themselves, so every call carries the
# User-Agent from settings. One shared client per call keeps this simple for now.
def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": get_settings().sec_user_agent},
        timeout=30.0,
    )


# Downloads (once) the big ticker->CIK table SEC publishes and turns it into a plain
# dict keyed by uppercase ticker. Everything downstream needs the CIK, not the ticker.
async def _load_ticker_cik_map(client: httpx.AsyncClient) -> dict[str, int]:
    global _ticker_cik_cache
    if _ticker_cik_cache is not None:
        return _ticker_cik_cache

    resp = await client.get(_TICKERS_URL)
    resp.raise_for_status()
    raw = resp.json()

    # The file is a dict of numbered rows, each row has cik_str and ticker fields.
    mapping = {row["ticker"].upper(): int(row["cik_str"]) for row in raw.values()}
    _ticker_cik_cache = mapping
    return mapping


# Turns a ticker like "AAPL" into its SEC CIK number, or None if SEC doesn't know it.
async def _resolve_cik(client: httpx.AsyncClient, ticker: str) -> int | None:
    mapping = await _load_ticker_cik_map(client)
    return mapping.get(ticker.upper())


# Main tool: pulls the most recent filing of a given form type (10-K, 10-Q, 8-K, etc)
# for a ticker straight from SEC EDGAR. Returns the filing's metadata plus a capped
# slice of the actual document text, so an agent can cite it and the ingestion worker
# knows exactly which document to fetch in full later.
async def fetch_filing(ticker: str, form_type: str = "10-K") -> dict[str, Any]:
    ticker = ticker.strip().upper()
    form_type = form_type.strip().upper()

    async with _client() as client:
        cik = await _resolve_cik(client, ticker)
        if cik is None:
            return {"status": "error", "error": f"Unknown ticker: {ticker}"}

        resp = await client.get(_SUBMISSIONS_URL.format(cik=cik))
        resp.raise_for_status()
        submissions = resp.json()

        # SEC gives recent filings as parallel arrays (forms[i], accessionNumbers[i], ...),
        # so we walk them together and stop at the first row matching the form type.
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])

        match_index = next((i for i, f in enumerate(forms) if f.upper() == form_type), None)
        if match_index is None:
            return {
                "status": "error",
                "error": f"No {form_type} filing found for {ticker}",
            }

        accession = accessions[match_index]
        primary_doc = primary_docs[match_index]
        accession_nodash = accession.replace("-", "")
        doc_url = _ARCHIVE_URL.format(
            cik=cik, accession_nodash=accession_nodash, doc=primary_doc
        )

        # Pull the primary document itself so the agent has real text to work with,
        # not just a link. We truncate and flag it rather than returning the full file.
        doc_resp = await client.get(doc_url)
        doc_resp.raise_for_status()
        full_text = doc_resp.text
        truncated = len(full_text) > _MAX_TEXT_CHARS

        logger.info(
            "fetched_filing", ticker=ticker, form_type=form_type, accession=accession
        )

        return {
            "status": "ok",
            "ticker": ticker,
            "cik": cik,
            "form_type": form_type,
            "accession_number": accession,
            "filing_date": filing_dates[match_index] if filing_dates else None,
            "report_date": report_dates[match_index] if report_dates else None,
            "document_url": doc_url,
            "truncated": truncated,
            "text": full_text[:_MAX_TEXT_CHARS],
        }
