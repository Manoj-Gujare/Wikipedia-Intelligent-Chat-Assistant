"""Registration, sign-in, password change, and API-key validation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from ...core.accounts import (
    EmailAlreadyRegistered,
    InvalidCredentials,
    get_account_store,
)
from ...core.embeddings import client_for_key
from ...core.ratelimit import get_login_throttle
from ...core.tokens import ROLE_USER, issue_token
from ...core.vector_store import get_vector_store
from ...models import ChangePasswordRequest, LoginRequest, LoginResponse, RegisterRequest
from ..dependencies import RequestIdentity, require_identity, require_openai

logger = logging.getLogger(__name__)
router = APIRouter()


def _session_for(account) -> LoginResponse:
    token, expires_in = issue_token(account.id, account.email)
    store = get_vector_store()
    summary = store.namespace_summary(account.id)
    return LoginResponse(
        access_token=token,
        expires_in=expires_in,
        account_id=account.id,
        email=account.email,
        role=ROLE_USER,
        shared_chunks=store.count(),
        personal_chunks=sum(summary["chunks"].values()),
        personal_articles=sum(summary["article_counts"].values()),
    )


@router.post("/auth/register", response_model=LoginResponse, status_code=201)
async def register(request: RegisterRequest) -> LoginResponse:
    """Create an account and sign in.

    Stores an email and a bcrypt hash — no API key, and no recoverable password.
    """
    try:
        account = get_account_store().register(request.email, request.password)
    except EmailAlreadyRegistered as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidCredentials as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _session_for(account)


def _throttle_key(http_request: Request, email: str) -> str:
    """Per (client, email). Keying on email alone would let anyone lock a known
    user out of their own account simply by failing logins on their behalf."""
    client = http_request.client.host if http_request.client else "unknown"
    return f"{client}|{(email or '').strip().lower()}"


@router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest, http_request: Request) -> LoginResponse:
    """Exchange email and password for a session token.

    The OpenAI key is not involved: it is a runtime credential supplied per
    request, so a user can rotate it without affecting their account or data.

    Repeated failures lock the caller out for a few minutes — bcrypt makes each
    guess expensive for us, but only a limiter makes guessing expensive for the
    attacker. The lockout is reported as 429 with `Retry-After`, distinct from
    the 401 a single wrong password gets, so the UI can say which happened.
    """
    throttle = get_login_throttle()
    key = _throttle_key(http_request, request.email)

    wait = throttle.retry_after(key)
    if wait:
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed sign-in attempts. Try again in {int(wait) + 1}s.",
            headers={"Retry-After": str(int(wait) + 1)},
        )

    try:
        account = get_account_store().authenticate(request.email, request.password)
    except InvalidCredentials as exc:
        throttle.record_failure(key)
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    throttle.reset(key)
    return _session_for(account)


@router.post("/auth/password")
async def change_password(
    request: ChangePasswordRequest,
    identity: RequestIdentity = Depends(require_identity),
) -> dict[str, bool]:
    try:
        get_account_store().change_password(
            identity.account_id, request.current_password, request.new_password
        )
    except InvalidCredentials as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"changed": True}


@router.post("/auth/verify-key")
async def verify_key(identity: RequestIdentity = Depends(require_openai)) -> dict[str, bool]:
    """Confirm the supplied key actually works, with the cheapest call available.

    Shape validation cannot tell a revoked key from a live one; only OpenAI can.
    """
    try:
        await client_for_key(identity.api_key).models.list()
    except Exception as exc:  # noqa: BLE001 - the reason is the user's to see
        raise HTTPException(status_code=400, detail="OpenAI rejected that key.") from exc
    return {"valid": True}
