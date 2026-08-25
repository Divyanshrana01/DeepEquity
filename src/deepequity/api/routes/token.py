from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from deepequity.api.auth import create_access_token
from deepequity.core.config import get_settings
from deepequity.core.logging import get_logger

logger = get_logger("deepequity.api.token")

router = APIRouter(tags=["auth"])


class TokenRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=200)
    client_secret: str = Field(min_length=1, max_length=500)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    #seconds, so a client can refresh before it expires rather than finding out with a 401
    expires_in: int


#hands out a token so something other than a python one-liner can call this api.
#
#client credentials rather than a username and password, because the caller here is a
#machine: the demo page, a script, a curl. there are no user accounts in this system and
#adding a users table with password hashing to serve one demo would be depth that isn't
#really there. this is a real oauth2 grant type doing exactly what it says.
#
#the endpoint turns itself off when no secret is configured. a default credential that
#works out of the box is the kind of thing that gets deployed and forgotten, and an auth
#endpoint that hands tokens to anyone is worse than having no endpoint at all.
@router.post("/token")
async def issue_token(payload: TokenRequest) -> TokenResponse:
    settings = get_settings()

    if not settings.demo_client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Token issuing is not configured. Set DEMO_CLIENT_SECRET in .env to "
                "enable it."
            ),
        )

    #compare_digest rather than ==, so the comparison takes the same time whether the
    #first character is wrong or the last one is. plain equality leaks the answer a
    #character at a time to anyone patient enough to measure.
    id_ok = secrets.compare_digest(payload.client_id, settings.demo_client_id)
    secret_ok = secrets.compare_digest(payload.client_secret, settings.demo_client_secret)

    #both checks always run before the decision, so a wrong client id doesn't return
    #faster than a wrong secret
    if not (id_ok and secret_ok):
        #the client id is logged, the secret never is
        logger.warning("token_request_rejected", client_id=payload.client_id[:50])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid client credentials"
        )

    logger.info("token_issued", client_id=payload.client_id)
    return TokenResponse(
        access_token=create_access_token(payload.client_id),
        expires_in=settings.jwt_expire_minutes * 60,
    )
