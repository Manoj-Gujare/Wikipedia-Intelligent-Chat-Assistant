"""Access control: tokens, ownership, and the retrieval boundary.

Weighted toward negative cases on purpose. A RAG access bug does not look like
an error — it looks like a fluent, well-cited answer built from a document the
caller was never allowed to see, which is exactly why it has to be asserted
rather than assumed.
"""

from __future__ import annotations

import time

import jwt
import pytest

from app.api.dependencies import RequestIdentity
from app.config import Settings
from app.core.retriever import _may_read
from app.core.tokens import ALGORITHM, InvalidToken, ROLE_USER, issue_token, verify_token
from app.core.vector_store import SearchHit

# Account ids are random uuids, unrelated to email or API key.
ALICE = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
BOB = "0f9e8d7c6b5a4938271605f4e3d2c1b0"


# HS256 wants >= 32 bytes; shorter keys make PyJWT warn and are bad practice
# to model in tests even where the value itself is throwaway.
SECRET = "test-secret-" + "0" * 32
ATTACKER_SECRET = "attacker-secret-" + "9" * 32


def _settings(**overrides) -> Settings:
    overrides.setdefault("jwt_secret", SECRET)
    return Settings(openai_api_key="test", **overrides)


def _identity(account_id: str) -> RequestIdentity:
    from app.core.tokens import TokenClaims

    return RequestIdentity(
        claims=TokenClaims(account_id=account_id, email="u@example.com", role=ROLE_USER),
        api_key="sk-test",
    )


# --------------------------------------------------------------------- tokens


def test_token_round_trips_the_account():
    settings = _settings()
    token, expires_in = issue_token(ALICE, "alice@example.com", settings=settings)

    claims = verify_token(token, settings)

    assert claims.account_id == ALICE
    assert claims.role == ROLE_USER
    assert expires_in > 0


def test_token_payload_never_contains_the_api_key():
    # A JWT is signed, not encrypted — anything in the payload is readable by
    # whoever holds the token.
    settings = _settings()
    token, _ = issue_token(ALICE, "alice@example.com", settings=settings)

    payload = jwt.decode(token, options={"verify_signature": False})

    assert "sk-" not in str(payload)
    assert set(payload) == {"sub", "email", "role", "iss", "iat", "exp"}


def test_token_signed_with_another_secret_is_rejected():
    forged, _ = issue_token(
        BOB, "bob@example.com", settings=_settings(jwt_secret=ATTACKER_SECRET)
    )

    with pytest.raises(InvalidToken):
        verify_token(forged, _settings())


def test_unsigned_alg_none_token_is_rejected():
    # The classic JWT downgrade: re-sign with "alg": "none" and hope the
    # verifier trusts the header. Pinning `algorithms` prevents it.
    payload = {"sub": BOB, "email": "bob@example.com", "role": "user",
               "iss": "wikipedia-chat-assistant", "iat": int(time.time()),
               "exp": int(time.time()) + 600}
    forged = jwt.encode(payload, key="", algorithm="none")

    with pytest.raises(InvalidToken):
        verify_token(forged, _settings())


def test_expired_token_is_rejected():
    settings = _settings(jwt_ttl_seconds=-1)
    token, _ = issue_token(ALICE, "alice@example.com", settings=settings)

    with pytest.raises(InvalidToken, match="expired"):
        verify_token(token, _settings())


def test_tampered_payload_is_rejected():
    settings = _settings()
    token, _ = issue_token(ALICE, "alice@example.com", settings=settings)
    header, payload, signature = token.split(".")
    swapped = jwt.encode({"sub": BOB}, ATTACKER_SECRET, algorithm=ALGORITHM).split(".")[1]

    with pytest.raises(InvalidToken):
        verify_token(f"{header}.{swapped}.{signature}", settings)


def test_garbage_is_rejected():
    with pytest.raises(InvalidToken):
        verify_token("not-a-token", _settings())


# ------------------------------------------------------------------ ownership


def _hit(owner: str | None) -> SearchHit:
    metadata = {"title": "Salary Review", "section_title": "Introduction", "lang": "en"}
    if owner is not None:
        metadata["owner_id"] = owner
    return SearchHit(id="c1", text="body", score=0.9, metadata=metadata)


def test_public_chunks_are_readable_by_everyone():
    assert _may_read(_hit(""), ALICE) is True
    assert _may_read(_hit(""), None) is True


def test_owner_can_read_their_own_chunk():
    assert _may_read(_hit(ALICE), ALICE) is True


def test_another_users_chunk_is_denied():
    # The core rule: Bob's document must never reach Alice's context, no matter
    # how relevant it scores.
    assert _may_read(_hit(BOB), ALICE) is False


def test_signed_out_caller_cannot_read_owned_chunks():
    assert _may_read(_hit(BOB), None) is False


def test_chunk_with_missing_owner_metadata_is_treated_as_public():
    # Legacy shared-corpus chunks predate the owner stamp. They are public
    # Wikipedia content, so readable — but the collection they live in is the
    # shared one, never a user namespace.
    assert _may_read(_hit(None), ALICE) is True


def test_identity_may_read_matches_the_retrieval_rule():
    # The API-layer check and the retrieval-layer check must not disagree;
    # a mismatch is how a filter gets bypassed on one path but not another.
    identity = _identity(ALICE)

    assert identity.may_read("") is True
    assert identity.may_read(None) is True
    assert identity.may_read(ALICE) is True
    assert identity.may_read(BOB) is False


# -------------------------------------------------------------- store tagging


def test_added_chunks_are_stamped_with_their_owner(tmp_path):
    from app.core.chunker import Chunk
    from app.core.vector_store import VectorStore

    settings = Settings(openai_api_key="test", chroma_path=str(tmp_path / "chroma"))
    store = VectorStore(settings)
    chunk = Chunk(id="x1", text="body", metadata={"lang": "en", "title": "Doc"})

    store.add([chunk], [[0.1] * 8], namespace=ALICE)

    assert chunk.metadata["owner_id"] == ALICE


def test_shared_corpus_chunks_are_stamped_public(tmp_path):
    from app.core.chunker import Chunk
    from app.core.vector_store import VectorStore

    settings = Settings(openai_api_key="test", chroma_path=str(tmp_path / "chroma"))
    store = VectorStore(settings)
    chunk = Chunk(id="x2", text="body", metadata={"lang": "en", "title": "Doc"})

    store.add([chunk], [[0.1] * 8])

    assert chunk.metadata["owner_id"] == ""
