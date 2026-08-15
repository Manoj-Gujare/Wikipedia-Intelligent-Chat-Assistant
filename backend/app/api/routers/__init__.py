"""One router assembled from the per-resource modules.

`main.py` mounts this single object twice -- once under `/api` and once at the
root -- so the split below stays invisible to callers.
"""

from fastapi import APIRouter

from . import auth, chat, conversations, knowledge_base, system

router = APIRouter()
router.include_router(auth.router)
router.include_router(chat.router)
router.include_router(conversations.router)
router.include_router(knowledge_base.router)
router.include_router(system.router)

__all__ = ["router"]
