"""Inspecting and building the caller's slice of the index.

Ingestion is a background job, so these endpoints hand back a job id and
let the client poll rather than holding a request open for minutes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ...core.cache import get_cache
from ...core.knowledge_base import SUGGESTED_TOPICS, get_kb_builder
from ...core.vector_store import get_vector_store
from ...models import AddArticlesRequest, IngestRequest, KnowledgeBaseResponse
from ..dependencies import RequestIdentity, require_identity, require_openai

logger = logging.getLogger(__name__)
router = APIRouter()


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
