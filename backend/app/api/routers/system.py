"""Diagnostics: index statistics and the health probe."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ...config import Settings, get_settings
from ...core.cache import get_cache
from ...core.vector_store import get_vector_store
from ...models import HealthResponse, StatsResponse

logger = logging.getLogger(__name__)
router = APIRouter()


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
