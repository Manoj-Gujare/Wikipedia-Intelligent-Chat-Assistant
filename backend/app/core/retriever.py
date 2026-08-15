"""Retrieval: semantic search, MMR diversification, and disambiguation handling."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import Settings, get_settings
from .embeddings import Embedder
from .vector_store import SearchHit, VectorStore, get_vector_store
from .wikipedia import WikipediaClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DisambiguationOption:
    title: str
    url: str


@dataclass(slots=True)
class RetrievalResult:
    hits: list[SearchHit] = field(default_factory=list)
    disambiguation: list[DisambiguationOption] = field(default_factory=list)
    disambiguation_term: str | None = None
    used_live_search: bool = False
    suggested_articles: list[dict[str, str]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.hits


def _may_read(hit: SearchHit, namespace: str | None) -> bool:
    """Public chunks are readable by all; owned chunks only by their owner.

    Fails closed: a chunk whose owner cannot be matched is dropped rather than
    treated as public.
    """
    owner = str(hit.metadata.get("owner_id") or "")
    if not owner:
        return True
    return owner == (namespace or "")


def _mmr(hits: list[SearchHit], k: int, lambda_mult: float) -> list[SearchHit]:
    """Maximal Marginal Relevance.

    Plain top-k on a Wikipedia index tends to return five near-identical chunks
    from the same section. MMR trades a little relevance for coverage, which
    matters for multi-part questions and for citing more than one article.

    Relevance comes from ``hit.score`` (similarity plus any lead-section boost)
    so ranking adjustments made upstream survive; embeddings are only used to
    measure redundancy between candidates.
    """
    usable = [h for h in hits if h.embedding]
    if len(usable) <= k:
        return hits[:k]

    matrix = np.asarray([h.embedding for h in usable], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms

    relevance = np.asarray([h.score for h in usable], dtype=np.float32)
    selected: list[int] = [int(np.argmax(relevance))]

    while len(selected) < k:
        redundancy = np.max(matrix @ matrix[selected].T, axis=1)
        scores = lambda_mult * relevance - (1 - lambda_mult) * redundancy
        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))

    return [usable[i] for i in selected]


class Retriever:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        wiki: WikipediaClient | None = None,
        settings: Settings | None = None,
        client=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_vector_store()
        # `client` carries the requesting user's API key when one is supplied.
        self.embedder = embedder or Embedder(self.settings, client=client)
        self.wiki = wiki or WikipediaClient(self.settings)

    async def retrieve(
        self, query: str, lang: str = "en", namespace: str | None = None
    ) -> RetrievalResult:
        query_vector = await self.embedder.embed_query(query)
        # Two pools per namespace, one embedding call for all of them. The
        # general pool finds specific passages; the lead-section pool keeps the
        # canonical summary facts in contention, because a question like "what
        # is the periodic table organised by?" is answered verbatim in the lead
        # while deep subsections crowd it out of a plain top-k.
        #
        # `namespace=None` is the shared base corpus; a namespace is the
        # account's own additions. Searching both means a user's private
        # articles compete with the shared ones on relevance alone.
        searches = [
            self.store.query(
                query_vector,
                top_k=self.settings.retrieval_top_k,
                lang=lang,
                include_embeddings=True,
            ),
            self.store.query(
                query_vector,
                top_k=self.settings.intro_pool_k,
                lang=lang,
                include_embeddings=True,
                section_title="Introduction",
            ),
        ]
        if namespace:
            searches += [
                self.store.query(
                    query_vector,
                    top_k=self.settings.retrieval_top_k,
                    lang=lang,
                    include_embeddings=True,
                    namespace=namespace,
                ),
                self.store.query(
                    query_vector,
                    top_k=self.settings.intro_pool_k,
                    lang=lang,
                    include_embeddings=True,
                    section_title="Introduction",
                    namespace=namespace,
                ),
            ]

        pools = await asyncio.gather(*searches)

        candidates: list[SearchHit] = []
        seen_ids: set[str] = set()
        for pool in pools:
            for hit in pool:
                if hit.id not in seen_ids:
                    seen_ids.add(hit.id)
                    candidates.append(hit)

        # Access check, applied once at the single point every hit passes
        # through. Collections already keep accounts apart; this re-checks the
        # owner stamped on each chunk, so a leak needs *both* the wrong
        # collection and a mislabelled chunk. It runs before ranking, so a
        # denied chunk cannot reach the prompt, the citations, or the article
        # links — a title alone would disclose what a document is about.
        candidates = [h for h in candidates if _may_read(h, namespace)]
        for hit in candidates:
            # Kept so downstream checks can still see raw similarity; the boost
            # is a ranking preference, not evidence about the query.
            hit.base_score = hit.score
            if hit.metadata.get("section_title") == "Introduction":
                hit.score += self.settings.intro_boost
        candidates.sort(key=lambda h: h.score, reverse=True)

        relevant = [h for h in candidates if h.score >= self.settings.min_relevance_score]

        if not relevant:
            # Nothing indexed for this question — fall back to the live API so
            # the assistant still routes the user somewhere useful.
            return await self._live_fallback(query, lang)

        # A disambiguation page ranking first means the question is ambiguous.
        top = relevant[0]
        if top.metadata.get("is_disambiguation"):
            options = await self.wiki.disambiguation_options(top.title, lang=lang)
            if options:
                return RetrievalResult(
                    hits=[],
                    disambiguation=[DisambiguationOption(**o) for o in options],
                    disambiguation_term=top.title,
                )

        # Disambiguation chunks are navigation lists, never answer material.
        answerable = [h for h in relevant if not h.metadata.get("is_disambiguation")]
        if not answerable:
            return await self._live_fallback(query, lang)

        selected = _mmr(
            answerable,
            k=self.settings.retrieval_final_k,
            lambda_mult=self.settings.mmr_lambda,
        )
        selected.sort(key=lambda h: h.score, reverse=True)
        return RetrievalResult(hits=selected)

    async def _live_fallback(self, query: str, lang: str) -> RetrievalResult:
        """Index miss: ask the MediaWiki search API for article suggestions.

        We deliberately do not synthesise an answer from these — we route the
        user to the real articles instead, which keeps the "never answer without
        indexed evidence" guarantee intact.
        """
        try:
            results = await self.wiki.search(query, lang=lang, limit=4)
        except Exception:  # noqa: BLE001 - fallback must never break the request
            logger.exception("Live Wikipedia search failed")
            return RetrievalResult(used_live_search=True)

        return RetrievalResult(
            used_live_search=True,
            suggested_articles=[
                {
                    "title": r["title"],
                    "url": WikipediaClient.article_url(lang, r["title"]),
                }
                for r in results
            ],
        )
