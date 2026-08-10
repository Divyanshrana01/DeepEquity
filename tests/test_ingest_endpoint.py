from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import deepequity.api.routes.ingest as ingest_route
from deepequity.api.auth import create_access_token
from deepequity.ingestion.errors import PermanentIngestionError
from deepequity.ingestion.models import Document, DocumentStatus, IngestResponse


def _auth_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token('analyst-1')}"}


def test_ingest_requires_auth(client: TestClient) -> None:
    response = client.post("/ingest", json={"ticker": "AAPL", "form_type": "10-K"})

    assert response.status_code == 401


def test_ingest_returns_202_with_document_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_request(ticker: str, form_type: str) -> IngestResponse:
        return IngestResponse(
            document_id=42,
            status=DocumentStatus.PENDING,
            ticker=ticker,
            doc_type=form_type,
            source_ref="0000-24-1",
            already_ingested=False,
        )

    monkeypatch.setattr(ingest_route.service, "request_ingestion", fake_request)

    response = client.post(
        "/ingest", json={"ticker": "AAPL", "form_type": "10-K"}, headers=_auth_header()
    )

    # 202 not 200: the work happens in the worker, we've only accepted the job.
    assert response.status_code == 202
    body = response.json()
    assert body["document_id"] == 42
    assert body["already_ingested"] is False


def test_ingest_bad_ticker_returns_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_request(ticker: str, form_type: str) -> IngestResponse:
        raise PermanentIngestionError("Unknown ticker: NOPE")

    monkeypatch.setattr(ingest_route.service, "request_ingestion", fake_request)

    response = client.post(
        "/ingest", json={"ticker": "NOPE", "form_type": "10-K"}, headers=_auth_header()
    )

    assert response.status_code == 400
    assert "Unknown ticker" in response.json()["detail"]


def test_status_endpoint_hides_raw_text(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_get(document_id: int) -> Document:
        return Document(
            id=document_id,
            ticker="AAPL",
            doc_type="10-K",
            source_ref="0000-24-1",
            raw_text="a very long filing body that nobody polling wants echoed back",
            status=DocumentStatus.COMPLETE,
        )

    monkeypatch.setattr(ingest_route.repository, "get", fake_get)

    response = client.get("/ingest/42", headers=_auth_header())

    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert body["status"] == "complete"
    assert body["raw_text"] is None


def test_status_endpoint_404s_for_unknown_document(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_get(document_id: int) -> Document | None:
        return None

    monkeypatch.setattr(ingest_route.repository, "get", fake_get)

    response = client.get("/ingest/999", headers=_auth_header())

    assert response.status_code == 404
