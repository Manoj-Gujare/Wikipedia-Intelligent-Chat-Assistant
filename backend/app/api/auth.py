"""Per-request identity.

Two separate things arrive on a request, and the distinction matters:

    Authorization: Bearer <jwt>   who you are      — verified by signature
    X-OpenAI-Key:  sk-…           what you spend   — never stored, never in the JWT

Identity is recovered from the token's signature, so a caller cannot assert an
account id, a role, or anyone else's ownership by editing a header or body
field. That is the whole access-control property: **the client never gets to
say who it is.**

The key is required only by endpoints that actually call OpenAI. Listing your
conversations should not demand a spendable credential, so there are two
dependencies: `require_identity` for anything read-only, and `require_openai`
for chat and knowledge-base building.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException

from ..core.accounts import Account, get_account_store
from ..core.tokens import InvalidToken, TokenClaims, verify_token


@dataclass(slots=True)
class RequestIdentity:
    """Server-derived identity for one request.

    `api_key` is empty on endpoints that do not call OpenAI.
    """

    claims: TokenClaims
    api_key: str = ""

    @property
    def account_id(self) -> str:
        return self.claims.account_id

    @property
    def email(self) -> str:
        return self.claims.email

    @property
    def role(self) -> str:
        return self.claims.role

    def may_read(self, owner_id: str | None) -> bool:
        """The access rule, in one place.

        Public content (no owner) is readable by everyone; owned content is
        readable only by its owner. Anything else is denied — an unrecognised
        owner fails closed rather than defaulting to public.
        """
        if not owner_id:
            return True
        return owner_id == self.account_id


def _bearer(header: str | None) -> str:
    if not header or not header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return header[7:].strip()


async def require_identity(
    authorization: str | None = Header(default=None),
    x_openai_key: str | None = Header(default=None),
) -> RequestIdentity:
    """Verify the session token. The API key is attached when present."""
    try:
        claims = verify_token(_bearer(authorization))
    except InvalidToken as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    return RequestIdentity(claims=claims, api_key=(x_openai_key or "").strip())


async def require_openai(
    identity: RequestIdentity = Depends(require_identity),
) -> RequestIdentity:
    """Verify the token *and* require a key, for endpoints that spend money.

    A distinct 428 rather than 401: the session is perfectly valid, the caller
    just has not supplied a key yet. The UI uses that to prompt for the key
    instead of bouncing the user back to the login screen.
    """
    if not identity.api_key:
        raise HTTPException(
            status_code=428,
            detail="Add your OpenAI API key to ask questions.",
        )
    return identity


def account_for(identity: RequestIdentity) -> Account | None:
    return get_account_store().get(identity.account_id)
