"""ChromaDB-backed vector store.

Kept behind a narrow interface (`add`, `query`, `count`, …) so swapping in
Pinecone or Weaviate later means writing one adapter, not touching the RAG
pipeline. Chroma's client is synchronous, so query paths hop to a worker thread
to keep the FastAPI event loop free.

Collection layout: one collection per language (``wikipedia_en``), plus a small
per-language collection holding only lead-section chunks (``wikipedia_en_intro``).
This exists for speed: Chroma's ``where`` metadata filters do a SQLite scan that
measured 250-950ms per query on a 9k-chunk index, while an unfiltered query on
the right collection takes ~15ms. Routing by collection instead of filtering by
metadata keeps retrieval off the latency budget entirely.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

# Chroma reads this at import time. Without it, its telemetry hook fires against
# whichever posthog version happens to be installed and spams the logs on every
# call. Must be set before `import chromadb`.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402

from ..config import Settings, get_settings
from .chunker import Chunk

logger = logging.getLogger(__name__)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
# "Number of requested results greater than number of elements" — expected when
# a collection is still small; the query just returns fewer hits.
logging.getLogger("chromadb.segment.impl.vector.local_hnsw").setLevel(logging.ERROR)

INTRO_SECTION = "Introduction"


@dataclass(slots=True)
class SearchHit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]
    embedding: list[float] | None = None
    # Similarity before any ranking boost is applied. Ranking preferences (the
    # lead-section boost) must not be mistaken for semantic signal: judging
    # ambiguity on boosted scores made "Venus" stop being ambiguous purely
    # because its lead chunk had been promoted. ``None`` means unboosted.
    base_score: float | None = None

    @property
    def raw_score(self) -> float:
        return self.score if self.base_score is None else self.base_score

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))

    @property
    def section_url(self) -> str:
        return str(self.metadata.get("section_url") or self.metadata.get("url", ""))


class VectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        # keyed by (namespace, lang, is_intro)
        self._collections: dict[tuple[str, str, bool], Any] = {}

    # ------------------------------------------------------------ collections

    def _name(self, lang: str, intro: bool = False, namespace: str | None = None) -> str:
        """Collection name.

        ``namespace=None`` is the shared base index every account can read.
        A namespace is an account id, giving that account its own collections
        alongside the shared ones — so a user's private additions are never
        visible to anyone else, and nobody re-pays to embed the common corpus.
        """
        prefix = self.settings.chroma_collection if namespace is None else f"u{namespace}"
        base = f"{prefix}_{lang}"
        return f"{base}_intro" if intro else base

    def _collection(self, lang: str, intro: bool = False, namespace: str | None = None):
        key = (namespace or "", lang, intro)
        if key not in self._collections:
            self._collections[key] = self._client.get_or_create_collection(
                name=self._name(lang, intro, namespace),
                # Cosine matches how OpenAI embeddings are normalised.
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[key]

    def _collection_names(self) -> list[str]:
        # Chroma 0.6 returns plain names; older versions return Collection
        # objects (whose .name may raise NotImplementedError under 0.6 shims).
        names = []
        for entry in self._client.list_collections():
            if isinstance(entry, str):
                names.append(entry)
            else:
                try:
                    names.append(entry.name)
                except NotImplementedError:
                    names.append(str(entry))
        return names

    def _language_names(self) -> dict[str, str]:
        """Maps language code -> main collection name."""
        prefix = f"{self.settings.chroma_collection}_"
        return {
            name[len(prefix):]: name
            for name in self._collection_names()
            if name.startswith(prefix) and not name.endswith("_intro")
        }

    # ----------------------------------------------------------------- write

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        namespace: str | None = None,
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        by_lang: dict[str, list[tuple[Chunk, list[float]]]] = {}
        for chunk, embedding in zip(chunks, embeddings):
            # Stamp ownership onto the chunk itself. The collection already
            # isolates accounts; this makes every chunk carry its own access
            # label so a retrieval bug that reads the wrong collection is still
            # caught downstream. Public corpus chunks carry "" and are readable
            # by everyone.
            chunk.metadata["owner_id"] = namespace or ""
            lang = str(chunk.metadata.get("lang", "en"))
            by_lang.setdefault(lang, []).append((chunk, embedding))

        for lang, pairs in by_lang.items():
            self._upsert(self._collection(lang, namespace=namespace), pairs)
            # Lead-section chunks are duplicated into a small dedicated
            # collection so the retriever's intro pool needs no metadata filter.
            intro_pairs = [
                p for p in pairs if p[0].metadata.get("section_title") == INTRO_SECTION
            ]
            if intro_pairs:
                self._upsert(
                    self._collection(lang, intro=True, namespace=namespace), intro_pairs
                )

    @staticmethod
    def _upsert(collection, pairs: list[tuple[Chunk, list[float]]]) -> None:
        collection.upsert(
            ids=[c.id for c, _ in pairs],
            embeddings=[e for _, e in pairs],
            documents=[c.text for c, _ in pairs],
            metadatas=[c.metadata for c, _ in pairs],
        )

    def reset(self) -> None:
        base = self.settings.chroma_collection
        for name in self._collection_names():
            if name == base or name.startswith(f"{base}_"):
                self._client.delete_collection(name)
        self._collections.clear()

    # ------------------------------------------------------------------ read

    def count(self) -> int:
        return sum(self.indexed_languages().values())

    def indexed_languages(self) -> dict[str, int]:
        """Chunk count per language, from collection names — no metadata scan."""
        return {
            lang: self._client.get_collection(name).count()
            for lang, name in self._language_names().items()
        }

    def indexed_page_ids(self, lang: str, namespace: str | None = None) -> set[int]:
        """Page ids already stored for a language, so ingest can skip them."""
        result = self._collection(lang, namespace=namespace).get(include=["metadatas"])
        return {
            int(meta["page_id"])
            for meta in (result.get("metadatas") or [])
            if meta and "page_id" in meta
        }

    def namespace_summary(self, namespace: str) -> dict:
        """What one account has added: chunk counts and article titles per language."""
        languages: dict[str, int] = {}
        articles: dict[str, set[str]] = {}
        prefix = f"u{namespace}_"

        for name in self._collection_names():
            if not name.startswith(prefix) or name.endswith("_intro"):
                continue
            lang = name[len(prefix):]
            collection = self._client.get_collection(name)
            languages[lang] = collection.count()
            rows = collection.get(include=["metadatas"])
            titles = {
                str(m.get("title"))
                for m in (rows.get("metadatas") or [])
                if m and m.get("title")
            }
            articles[lang] = titles

        return {
            "chunks": languages,
            "article_counts": {lang: len(t) for lang, t in articles.items()},
            "articles": {lang: sorted(t) for lang, t in articles.items()},
        }

    def drop_namespace(self, namespace: str) -> int:
        """Delete every collection belonging to one account."""
        prefix = f"u{namespace}_"
        removed = 0
        for name in self._collection_names():
            if name.startswith(prefix):
                self._client.delete_collection(name)
                removed += 1
        self._collections = {
            k: v for k, v in self._collections.items() if k[0] != namespace
        }
        return removed

    def _query_sync(
        self,
        embedding: list[float],
        top_k: int,
        lang: str,
        include_embeddings: bool,
        section_title: str | None,
        namespace: str | None,
    ) -> list[SearchHit]:
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")

        collection = self._collection(
            lang, intro=(section_title == INTRO_SECTION), namespace=namespace
        )

        # This was `if collection.count() == 0: return []`, which cost a
        # measured 11ms per call and ran on both pools of every retrieval — the
        # majority of the 35ms Chroma spends on a turn, paid to ask a question
        # whose answer is "no" only before anything has been ingested. Querying
        # an empty collection is not an error in Chroma; it returns no rows, and
        # a malformed one raises, which the caller already has to tolerate.
        try:
            result = collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=include,
            )
        except Exception:  # noqa: BLE001 - an unusable collection has no hits
            logger.debug("Query against %r failed; treating as empty", collection.name)
            return []

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        embeddings = (result.get("embeddings") or [[]])[0] if include_embeddings else []

        hits: list[SearchHit] = []
        for i, chunk_id in enumerate(ids):
            # Chroma returns cosine distance in [0, 2]; convert to similarity.
            similarity = 1.0 - float(distances[i])
            hits.append(
                SearchHit(
                    id=chunk_id,
                    text=documents[i],
                    score=similarity,
                    metadata=dict(metadatas[i] or {}),
                    embedding=list(embeddings[i]) if include_embeddings and len(embeddings) > i else None,
                )
            )
        return hits

    async def query(
        self,
        embedding: list[float],
        top_k: int,
        lang: str = "en",
        include_embeddings: bool = False,
        section_title: str | None = None,
        namespace: str | None = None,
    ) -> list[SearchHit]:
        return await asyncio.to_thread(
            self._query_sync,
            embedding,
            top_k,
            lang,
            include_embeddings,
            section_title,
            namespace,
        )


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()
