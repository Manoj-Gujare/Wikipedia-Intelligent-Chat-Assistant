"""Starting a session, and the two ways to ask a question.

`/chat` returns the finished answer; `/chat/stream` emits the same turn as
server-sent events so the UI can render tokens as they arrive."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...core.vector_store import get_vector_store
from ...graph.build import services_for
from ...graph.runner import run_turn, stream_turn
from ...models import ChatRequest, ChatResponse, SessionResponse
from ..dependencies import RequestIdentity, require_identity, require_openai

logger = logging.getLogger(__name__)
router = APIRouter()


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
