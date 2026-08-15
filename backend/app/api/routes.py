"""HTTP API: session, chat, conversation history, and knowledge-base building."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..config import Settings, get_settings
from ..core.accounts import (
    EmailAlreadyRegistered,
    InvalidCredentials,
    get_account_store,
)
from ..core.embeddings import client_for_key
from ..core.cache import get_cache
from ..core.conversation import get_conversation_store
from ..core.knowledge_base import SUGGESTED_TOPICS, get_kb_builder
from ..core.ratelimit import get_login_throttle
from ..core.tokens import ROLE_USER, issue_token
from ..core.vector_store import get_vector_store
from ..graph.build import services_for
from ..graph.runner import run_turn, stream_turn
from ..models import (
    AddArticlesRequest,
    ChangePasswordRequest,
    ChatRequest,
    ChatResponse,
    ConversationSummary,
    HealthResponse,
    IngestRequest,
    KnowledgeBaseResponse,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    SessionResponse,
    StatsResponse,
)
from .auth import RequestIdentity, require_identity, require_openai

logger = logging.getLogger(__name__)
router = APIRouter()


# --------------------------------------------------------------------- session


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


@router.post("/session", response_model=SessionResponse)
async def start_session(identity: RequestIdentity = Depends(require_identity)) -> SessionResponse:
    """Current account state for an already-signed-in caller."""
    store = get_vector_store()
    summary = store.namespace_summary(identity.account_id)
    return SessionResponse(
        account_id=identity.account_id,
        email=identity.email,
        role=identity.role,
        shared_chunks=store.count(),
        personal_chunks=sum(summary["chunks"].values()),
        personal_articles=sum(summary["article_counts"].values()),
    )


# ------------------------------------------------------------------------ chat


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, identity: RequestIdentity = Depends(require_openai)
) -> ChatResponse:
    """Run one graph turn against the shared corpus plus this account's own."""
    services = services_for(identity.api_key, identity.account_id)
    try:
        return await run_turn(
            request.message,
            session_id=request.session_key,
            lang=request.lang,
            services=services,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="That conversation isn't yours.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail="Failed to generate an answer") from exc


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest, identity: RequestIdentity = Depends(require_openai)
) -> StreamingResponse:
    """Server-sent events: `meta`, `intent`, `rewrite`, `retrieval`, `token`, `done`."""
    services = services_for(identity.api_key, identity.account_id)

    async def event_source():
        try:
            async for event, payload in stream_turn(
                request.message,
                session_id=request.session_key,
                lang=request.lang,
                services=services,
            ):
                yield f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except PermissionError:
            yield f"event: error\ndata: {json.dumps({'detail': 'That conversation is not yours.'})}\n\n"
        except Exception as exc:  # noqa: BLE001 - surface errors inside the stream
            logger.exception("Streaming chat failed")
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Stops nginx and friends from buffering the stream away.
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------- conversations


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    identity: RequestIdentity = Depends(require_identity),
) -> list[ConversationSummary]:
    """Sidebar list: this account's threads, newest first."""
    return [
        ConversationSummary(
            conversation_id=c.id,
            title=c.title or "New conversation",
            lang=c.lang,
            updated_at=c.updated_at,
        )
        for c in get_conversation_store().list_for_account(identity.account_id)
    ]


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str, identity: RequestIdentity = Depends(require_identity)
) -> dict:
    """Stored history for a conversation, so a reloaded UI can resume it."""
    conversation = get_conversation_store().peek(conversation_id)
    if conversation is None:
        return {"conversation_id": conversation_id, "lang": "en", "turns": []}
    if conversation.account_id and conversation.account_id != identity.account_id:
        raise HTTPException(status_code=403, detail="That conversation isn't yours.")
    return {
        "conversation_id": conversation.id,
        "lang": conversation.lang,
        "turns": [
            {"role": t.role, "content": t.content, "meta": t.meta}
            for t in conversation.turns
        ],
    }


@router.delete("/conversations/{conversation_id}")
async def clear_conversation(
    conversation_id: str, identity: RequestIdentity = Depends(require_identity)
) -> dict[str, bool]:
    store = get_conversation_store()
    conversation = store.peek(conversation_id)
    if conversation and conversation.account_id and conversation.account_id != identity.account_id:
        raise HTTPException(status_code=403, detail="That conversation isn't yours.")
    return {"deleted": store.clear(conversation_id)}


# -------------------------------------------------------------- knowledge base


@router.get("/kb", response_model=KnowledgeBaseResponse)
async def knowledge_base(
    identity: RequestIdentity = Depends(require_identity),
) -> KnowledgeBaseResponse:
    """What this account has added, plus suggested topics and any active build."""
    store = get_vector_store()
    summary = store.namespace_summary(identity.account_id)
    builder = get_kb_builder()
    active = builder.active_for_account(identity.account_id)

    return KnowledgeBaseResponse(
        shared_chunks=store.count(),
        personal_chunks=sum(summary["chunks"].values()),
        personal_articles=sum(summary["article_counts"].values()),
        articles_by_language=summary["article_counts"],
        titles=summary["articles"].get("en", [])[:200],
        suggested_topics=SUGGESTED_TOPICS,
        active_job=active.to_dict() if active else None,
        recent_jobs=[j.to_dict() for j in builder.list_for_account(identity.account_id)[:5]],
    )


@router.post("/kb/ingest")
async def start_ingest(
    request: IngestRequest, identity: RequestIdentity = Depends(require_openai)
) -> dict:
    """Queue a background build. Embedding is billed to the caller's own key."""
    try:
        job = get_kb_builder().submit(
            account_id=identity.account_id,
            api_key=identity.api_key,
            topics=request.topics,
            articles_per_topic=request.articles_per_topic,
            lang=request.lang,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.to_dict()


@router.post("/kb/articles")
async def add_articles(
    request: AddArticlesRequest, identity: RequestIdentity = Depends(require_openai)
) -> dict:
    """Add specific Wikipedia articles by exact title.

    This is the path for "the answer wasn't in the index": the assistant offers
    the articles live search found, and the user adds the one they meant. The
    articles land in the caller's own namespace, owned by them.
    """
    try:
        job = get_kb_builder().submit(
            account_id=identity.account_id,
            api_key=identity.api_key,
            topics=request.titles,
            articles_per_topic=1,
            lang=request.lang,
            exact=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.to_dict()


@router.get("/kb/jobs/{job_id}")
async def job_status(
    job_id: str, identity: RequestIdentity = Depends(require_identity)
) -> dict:
    job = get_kb_builder().get(job_id, identity.account_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job.")
    return job.to_dict()


@router.post("/kb/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str, identity: RequestIdentity = Depends(require_identity)
) -> dict[str, bool]:
    return {"cancelled": get_kb_builder().cancel(job_id, identity.account_id)}


@router.delete("/kb")
async def reset_knowledge_base(
    identity: RequestIdentity = Depends(require_identity),
) -> dict[str, int]:
    """Delete everything this account added. The shared corpus is untouched."""
    removed = get_vector_store().drop_namespace(identity.account_id)
    # Cached answers may have been grounded in the articles just removed.
    get_cache().invalidate(identity.account_id)
    return {"collections_removed": removed}


# ----------------------------------------------------------------- diagnostics


@router.get("/stats", response_model=StatsResponse)
async def stats(settings: Settings = Depends(get_settings)) -> StatsResponse:
    store = get_vector_store()
    return StatsResponse(
        total_chunks=store.count(),
        languages=store.indexed_languages(),
        collection=settings.chroma_collection,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
        cache=get_cache().stats(),
    )


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    try:
        indexed = get_vector_store().count()
        status = "ok" if indexed > 0 else "empty_index"
    except Exception:  # noqa: BLE001
        logger.exception("Vector store unavailable")
        indexed, status = 0, "degraded"

    return HealthResponse(
        status=status,
        indexed_chunks=indexed,
        openai_configured=settings.openai_configured,
    )
