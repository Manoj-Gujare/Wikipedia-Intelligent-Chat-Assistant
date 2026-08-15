"""OpenAI embedding helpers: batched, retried, and cached for repeat queries."""

from __future__ import annotations

import logging
from functools import lru_cache

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

# OpenAI accepts up to 2048 inputs per embeddings call; stay well under so a
# single failed batch is cheap to retry.
BATCH_SIZE = 96

def is_retryable(exc: BaseException) -> bool:
    """Retry transient failures only.

    A 401/403/400 will fail identically on every attempt, so retrying it just
    delays the error the caller needs to see.
    """
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False


@lru_cache
def _shared_http_client() -> httpx.AsyncClient:
    """One connection pool shared by every per-request OpenAI client.

    This is what makes request-scoped credentials affordable: building a fresh
    `AsyncOpenAI` per request is cheap, but a fresh TLS handshake per request is
    not. The pool carries no authentication of its own — the key travels on the
    request headers — so nothing user-specific is retained between requests.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )


def client_for_key(api_key: str) -> AsyncOpenAI:
    """An OpenAI client bound to a caller-supplied key.

    The key lives on this object for the duration of the request that created
    it and is never written to disk, logged, or cached across requests.
    """
    if not api_key or not api_key.strip():
        raise RuntimeError("No OpenAI API key supplied for this request.")
    return AsyncOpenAI(
        api_key=api_key.strip(), max_retries=0, http_client=_shared_http_client()
    )


@lru_cache
def get_openai_client() -> AsyncOpenAI:
    """Client built from the server's own key (ingestion CLI, evaluation).

    User-facing requests use :func:`client_for_key` instead — the server key is
    only for operator-run tooling.
    """
    settings = get_settings()
    if not settings.openai_configured:
        raise RuntimeError(
            "OPENAI_API_KEY is missing or still set to the placeholder. "
            "Edit backend/.env and set a real key."
        )
    return AsyncOpenAI(
        api_key=settings.openai_api_key, max_retries=0, http_client=_shared_http_client()
    )


class Embedder:
    """Wraps the embeddings endpoint with batching and backoff."""

    def __init__(
        self, settings: Settings | None = None, client: AsyncOpenAI | None = None
    ) -> None:
        self.settings = settings or get_settings()
        # A caller-supplied client carries that user's key; the default is the
        # server key, used by ingestion tooling and evaluation.
        self._client = client or get_openai_client()

    @retry(
        retry=retry_if_exception(is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
            dimensions=self.settings.embedding_dimensions,
        )
        # The API preserves input order, but sort by index to be explicit.
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            vectors.extend(await self._embed_batch(batch))
            logger.debug("Embedded %d/%d chunks", len(vectors), len(texts))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed_batch([text])
        return vectors[0]
