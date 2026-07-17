import pytest
from fastapi import HTTPException

from deepequity.api.auth import create_access_token, decode_access_token


def test_token_round_trip() -> None:
    token = create_access_token(subject="analyst-1")

    payload = decode_access_token(token)

    assert payload["sub"] == "analyst-1"


def test_invalid_token_is_rejected() -> None:
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token("not-a-real-token")

    assert exc_info.value.status_code == 401


def test_expired_token_is_rejected() -> None:
    token = create_access_token(subject="analyst-1", expires_minutes=-1)

    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token)

    assert exc_info.value.status_code == 401
