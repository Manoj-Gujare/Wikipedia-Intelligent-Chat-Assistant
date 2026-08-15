"""Listing, restoring and clearing a caller's saved conversations."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ...core.conversation import get_conversation_store
from ...models import ConversationSummary
from ..dependencies import RequestIdentity, require_identity

logger = logging.getLogger(__name__)
router = APIRouter()


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
