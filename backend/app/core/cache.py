"""TTL cache for repeated questions.

Wikipedia content is static and evaluation suites ask the same questions over
and over, so caching answers for identical (question, language) pairs is the
cheapest win against the <3s response-time target.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from functools import lru_cache
from threading import Lock
from typing import Any

from ..config import Settings, get_settings

MAX_ENTRIES = 512


class TTLCache:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        # Per-namespace generation counters; bumping one retires its keys.
        self._generations: dict[str, int] = {}
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def key(self, question: str, lang: str, namespace: str = "") -> str:
        """Cache key scoped to the account and its knowledge-base generation.

        The namespace matters because two accounts asking the same question can
        legitimately get different answers — each searches the shared corpus
        plus their own articles. The generation matters because adding articles
        changes what the same question should return; bumping it retires every
        key for that account without walking the cache.
        """
        normalised = " ".join(question.lower().split())
        generation = self._generations.get(namespace, 0)
        return hashlib.sha1(
            f"{namespace}:{generation}:{lang}:{normalised}".encode()
        ).hexdigest()

    def invalidate(self, namespace: str) -> None:
        """Retire every cached answer for one account."""
        with self._lock:
            self._generations[namespace] = self._generations.get(namespace, 0) + 1

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if expires_at <= time.time():
                del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._entries[key] = (time.time() + self.settings.cache_ttl_seconds, value)
            self._entries.move_to_end(key)
            while len(self._entries) > MAX_ENTRIES:
                self._entries.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses}


@lru_cache
def get_cache() -> TTLCache:
    return TTLCache()
