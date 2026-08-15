"""Per-account knowledge-base building.

A user picks categories and how many articles each should contribute; we crawl
Wikipedia, chunk, embed with **their** API key, and write into **their**
namespace. Ingesting hundreds of articles takes minutes, so the work runs as a
background job and the UI polls for progress rather than holding a request open.

Cost is the user's, so the job reports articles and chunks as it goes — the
number of chunks is what they are billed to embed.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock
from typing import Literal

from ..config import Settings, get_settings
from .cache import get_cache
from .chunker import chunk_article
from .embeddings import Embedder, client_for_key
from .vector_store import VectorStore, get_vector_store
from .wikipedia import WikipediaClient

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]

# Ingest in waves so progress moves visibly and memory stays bounded.
BATCH = 10
MAX_ARTICLES_PER_TOPIC = 200
MAX_TOPICS_PER_JOB = 10

# Suggested starting points offered in the UI. Each is a real MediaWiki
# category; users can also type any category or article title of their own.
SUGGESTED_TOPICS: list[dict[str, str]] = [
    {"label": "Space & Astronomy", "category": "Category:Astronomy"},
    {"label": "Medicine & Health", "category": "Category:Medicine"},
    {"label": "Machine Learning", "category": "Category:Machine learning"},
    {"label": "Economics & Finance", "category": "Category:Economics"},
    {"label": "Philosophy", "category": "Category:Philosophy"},
    {"label": "World Cinema", "category": "Category:Cinema"},
    {"label": "Sports", "category": "Category:Sports"},
    {"label": "Cricket", "category": "Category:Cricket"},
    {"label": "Music", "category": "Category:Music"},
    {"label": "Law & Government", "category": "Category:Law"},
    {"label": "Environment & Climate", "category": "Category:Climate"},
    {"label": "Food & Cooking", "category": "Category:Food and drink"},
]


@dataclass
class IngestJob:
    id: str
    account_id: str
    topics: list[str]
    articles_per_topic: int
    lang: str = "en"
    # True when `topics` are exact article titles to fetch as-is, rather than
    # categories or search terms to expand.
    exact: bool = False
    status: JobStatus = "queued"
    articles_done: int = 0
    articles_total: int = 0
    chunks_written: int = 0
    current: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.id,
            "status": self.status,
            "topics": self.topics,
            "lang": self.lang,
            "articles_done": self.articles_done,
            "articles_total": self.articles_total,
            "chunks_written": self.chunks_written,
            "current": self.current,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class KnowledgeBaseBuilder:
    def __init__(self, settings: Settings | None = None, store: VectorStore | None = None):
        self.settings = settings or get_settings()
        self.store = store or get_vector_store()
        self._jobs: dict[str, IngestJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = Lock()

    # -------------------------------------------------------------- job admin

    def get(self, job_id: str, account_id: str) -> IngestJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
        # Scoped by account so a guessed job id cannot reveal another user's work.
        return job if job and job.account_id == account_id else None

    def list_for_account(self, account_id: str) -> list[IngestJob]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.account_id == account_id]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:20]

    def active_for_account(self, account_id: str) -> IngestJob | None:
        return next(
            (
                j
                for j in self.list_for_account(account_id)
                if j.status in ("queued", "running")
            ),
            None,
        )

    def cancel(self, job_id: str, account_id: str) -> bool:
        job = self.get(job_id, account_id)
        if not job or job.status not in ("queued", "running"):
            return False
        task = self._tasks.get(job_id)
        if task:
            task.cancel()
        job.status = "cancelled"
        job.finished_at = time.time()
        return True

    # ------------------------------------------------------------- submission

    def submit(
        self,
        account_id: str,
        api_key: str,
        topics: list[str],
        articles_per_topic: int,
        lang: str = "en",
        exact: bool = False,
    ) -> IngestJob:
        """Queue a build. Raises ValueError on invalid input."""
        topics = [t.strip() for t in topics if t and t.strip()][:MAX_TOPICS_PER_JOB]
        if not topics:
            raise ValueError("Choose at least one topic or category.")

        articles_per_topic = (
            1 if exact else max(1, min(int(articles_per_topic), MAX_ARTICLES_PER_TOPIC))
        )

        if self.active_for_account(account_id):
            raise ValueError("A knowledge-base build is already running.")

        job = IngestJob(
            id=uuid.uuid4().hex[:16],
            account_id=account_id,
            topics=topics,
            articles_per_topic=articles_per_topic,
            lang=lang,
            exact=exact,
            articles_total=len(topics) * articles_per_topic,
        )
        with self._lock:
            self._jobs[job.id] = job

        # The key is passed to the task and never stored on the job record,
        # so job state can be logged or serialised without leaking it.
        self._tasks[job.id] = asyncio.create_task(self._run(job, api_key))
        return job

    # ----------------------------------------------------------------- runner

    async def _run(self, job: IngestJob, api_key: str) -> None:
        job.status = "running"
        embedder = Embedder(self.settings, client=client_for_key(api_key))

        try:
            async with WikipediaClient(self.settings) as wiki:
                known = self.store.indexed_page_ids(job.lang, namespace=job.account_id)

                for topic in job.topics:
                    job.current = topic
                    titles = await self._titles_for(wiki, topic, job)
                    await self._ingest_titles(job, wiki, embedder, titles, known)

            job.status = "done"
            job.current = ""
            # Answers cached before this build were computed without these
            # articles, so the same question should now be re-answered.
            get_cache().invalidate(job.account_id)
            logger.info(
                "KB job %s finished: %d articles, %d chunks",
                job.id, job.articles_done, job.chunks_written,
            )
        except asyncio.CancelledError:
            job.status = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 - surface the reason to the user
            job.status = "failed"
            job.error = _user_facing_error(exc)
            logger.exception("KB job %s failed", job.id)
        finally:
            job.finished_at = time.time()
            self._tasks.pop(job.id, None)

    async def _titles_for(
        self, wiki: WikipediaClient, topic: str, job: IngestJob
    ) -> list[str]:
        """Resolve a topic into article titles.

        A topic can be a MediaWiki category ("Category:Astronomy") or free text,
        in which case we use the search API — so a user can type "cricket" and
        get useful articles without knowing Wikipedia's category tree.
        """
        if job.exact:
            return [topic]

        limit = job.articles_per_topic
        if topic.lower().startswith("category:"):
            try:
                titles = await wiki.category_members(topic, lang=job.lang, limit=limit)
                if titles:
                    return titles
            except Exception:  # noqa: BLE001 - fall through to search
                logger.warning("Category %r unreadable; falling back to search", topic)

        results = await wiki.search(topic, lang=job.lang, limit=limit)
        return [r["title"] for r in results]

    async def _ingest_titles(
        self,
        job: IngestJob,
        wiki: WikipediaClient,
        embedder: Embedder,
        titles: list[str],
        known: set[int],
    ) -> None:
        for start in range(0, len(titles), BATCH):
            batch = titles[start : start + BATCH]
            articles = await wiki.fetch_articles(batch, lang=job.lang)
            fresh = [a for a in articles if a.page_id not in known]

            chunks = []
            for article in fresh:
                chunks.extend(chunk_article(article, self.settings))

            if chunks:
                embeddings = await embedder.embed_documents([c.text for c in chunks])
                self.store.add(chunks, embeddings, namespace=job.account_id)
                job.chunks_written += len(chunks)

            known.update(a.page_id for a in fresh)
            job.articles_done += len(batch)


def _user_facing_error(exc: Exception) -> str:
    """Turn an exception into something worth showing a user."""
    text = str(exc)
    if "authentication" in text.lower() or "invalid_api_key" in text or "401" in text:
        return "OpenAI rejected your API key. Check it and try again."
    if "insufficient_quota" in text or "quota" in text.lower():
        return "Your OpenAI account has no remaining quota for embeddings."
    if "rate limit" in text.lower() or "429" in text:
        return "OpenAI rate-limited the request. Try a smaller batch shortly."
    return "The build failed. Check the server logs for details."


@lru_cache
def get_kb_builder() -> KnowledgeBaseBuilder:
    return KnowledgeBaseBuilder()
