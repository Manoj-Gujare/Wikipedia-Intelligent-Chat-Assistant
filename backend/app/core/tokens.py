"""JWT issue and verification.

The access rule this enforces is ownership: a caller may read the public shared
corpus plus documents they added, and nothing else. The point of the token is
that the *server* decides who you are — identity is recovered from a verified
signature, never from anything the client asserts in a body, query string or
plain header.

What is deliberately **not** in the token:

* the OpenAI API key. A JWT is signed, not encrypted; its payload is
  base64 — readable by anyone holding it. Putting a spendable credential in
  there would publish it. The key travels on its own header and is never stored.
* anything the client would like to be true. Role and subject are written by
  the server at login and re-read from the signature on every request.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import jwt

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
ISSUER = "wikipedia-chat-assistant"

# Every account is an ordinary user today. The claim exists so a future
# privileged role is a policy change rather than a token-format migration.
ROLE_USER = "user"


class InvalidToken(ValueError):
    """Raised when a token is missing, malformed, expired or badly signed."""


@dataclass(slots=True)
class TokenClaims:
    account_id: str
    email: str
    role: str


def issue_token(account_id: str, email: str, role: str = ROLE_USER,
                settings: Settings | None = None) -> tuple[str, int]:
    """Sign a token for an authenticated account. Returns (token, expires_in)."""
    settings = settings or get_settings()
    ttl = settings.jwt_ttl_seconds
    now = int(time.time())
    payload = {
        "sub": account_id,
        "email": email,
        "role": role,
        "iss": ISSUER,
        "iat": now,
        "exp": now + ttl,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM), ttl


def verify_token(token: str, settings: Settings | None = None) -> TokenClaims:
    """Decode and validate a token, or raise InvalidToken.

    `algorithms` is pinned to a single value: accepting a list the attacker can
    influence is how the classic "alg: none" and RS256-to-HS256 confusion
    attacks work.
    """
    settings = settings or get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[ALGORITHM],
            issuer=ISSUER,
            options={"require": ["exp", "sub", "iss"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise InvalidToken("Your session has expired. Sign in again.") from exc
    except jwt.InvalidTokenError as exc:
        raise InvalidToken("Invalid session token. Sign in again.") from exc

    account_id = str(payload.get("sub") or "")
    if not account_id:
        raise InvalidToken("Token carries no account.")

    return TokenClaims(
        account_id=account_id,
        email=str(payload.get("email") or ""),
        role=str(payload.get("role") or ROLE_USER),
    )
