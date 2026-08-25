from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from deepequity.api.auth import decode_access_token
from deepequity.core.config import get_settings


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_CLIENT_ID", "demo")
    monkeypatch.setenv("DEMO_CLIENT_SECRET", "a-real-secret")
    get_settings.cache_clear()


#the endpoint is off unless a secret is set, and it says so rather than failing oddly.
#a credential that works out of the box is the sort of thing that gets deployed and
#forgotten, and an auth endpoint handing tokens to anyone is worse than no endpoint.
def test_disabled_when_no_secret_is_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEMO_CLIENT_SECRET", "")
    get_settings.cache_clear()

    response = client.post("/token", json={"client_id": "demo", "client_secret": "x"})

    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_good_credentials_return_a_usable_token(
    client: TestClient, configured: None
) -> None:
    response = client.post(
        "/token", json={"client_id": "demo", "client_secret": "a-real-secret"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0

    #the real check: the token this endpoint issues is one the rest of the api accepts.
    #a token that decodes but gets rejected on every route would look fine here.
    assert decode_access_token(body["access_token"])["sub"] == "demo"
    protected = client.get("/stats", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert protected.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"client_id": "demo", "client_secret": "wrong"},
        {"client_id": "wrong", "client_secret": "a-real-secret"},
        {"client_id": "wrong", "client_secret": "wrong"},
    ],
)
def test_bad_credentials_are_rejected(
    client: TestClient, configured: None, payload: dict[str, str]
) -> None:
    response = client.post("/token", json=payload)

    assert response.status_code == 401
    #the same message whichever half was wrong, so the response doesn't tell an attacker
    #that they've got a valid client id and only need the secret
    assert response.json()["detail"] == "Invalid client credentials"


def test_the_secret_never_appears_in_the_response(
    client: TestClient, configured: None
) -> None:
    for secret in ("a-real-secret", "wrong"):
        response = client.post("/token", json={"client_id": "demo", "client_secret": secret})
        assert "a-real-secret" not in response.text
