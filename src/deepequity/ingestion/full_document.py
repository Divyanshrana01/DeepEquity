from __future__ import annotations

import httpx

from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger
from deepequity.ingestion.errors import PermanentIngestionError, TransientIngestionError

logger = get_logger("deepequity.ingestion.full_document")


#downloads the complete filing. the mcp tool deliberately caps what it returns at 20k
#characters so a tool call can't flood an agent's context, but 20k of a 10-K is barely
#past the cover page. the parts worth retrieving, risk factors and the md&a, sit deep in
#the document, so ingestion goes and gets the whole thing itself.
async def fetch_full_document(url: str) -> str:
    settings = get_settings()

    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": settings.sec_user_agent},
            timeout=120.0,
            follow_redirects=True,
        ) as client:
            #streamed so we can stop early on something pathologically large instead of
            #pulling the whole thing into memory first and then deciding
            async with client.stream("GET", url) as response:
                if response.status_code >= 500:
                    #sec having a bad moment is worth another go later
                    raise TransientIngestionError(
                        f"sec returned {response.status_code} for {url}"
                    )
                if response.status_code >= 400:
                    #a 404 means the url is wrong and will stay wrong
                    raise PermanentIngestionError(
                        f"sec returned {response.status_code} for {url}"
                    )

                chunks: list[bytes] = []
                total = 0
                async for piece in response.aiter_bytes():
                    total += len(piece)
                    if total > settings.max_document_bytes:
                        raise PermanentIngestionError(
                            f"document at {url} exceeds max_document_bytes "
                            f"({settings.max_document_bytes})"
                        )
                    chunks.append(piece)

    except httpx.TimeoutException as exc:
        raise TransientIngestionError(f"timed out fetching {url}") from exc
    except httpx.HTTPError as exc:
        raise TransientIngestionError(f"network error fetching {url}: {exc}") from exc

    #filings are usually utf-8 but older ones can carry odd bytes, replace rather than
    #fail the whole document over a stray character
    body = b"".join(chunks).decode("utf-8", errors="replace")
    logger.info("full_document_fetched", url=url, bytes=total)
    return body
