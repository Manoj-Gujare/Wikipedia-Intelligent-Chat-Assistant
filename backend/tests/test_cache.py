"""Answer cache: key scoping, invalidation and expiry."""

from __future__ import annotations


from app.config import Settings
from app.core.cache import TTLCache



def _cache() -> TTLCache:
    return TTLCache(Settings(openai_api_key="test", cache_ttl_seconds=60))


def test_cache_key_ignores_case_and_whitespace():
    cache = _cache()

    assert cache.key("  What   is DNA? ", "en") == cache.key("what is dna?", "en")


def test_cache_key_is_language_scoped():
    cache = _cache()

    assert cache.key("What is DNA?", "en") != cache.key("What is DNA?", "es")


def test_cache_key_is_account_scoped():
    # Two accounts search different corpora (shared + their own additions), so
    # the same question can legitimately have different answers.
    cache = _cache()

    assert cache.key("What is DNA?", "en", "acct-a") != cache.key(
        "What is DNA?", "en", "acct-b"
    )


def test_invalidating_an_account_retires_its_cached_answers():
    # After a knowledge-base build the same question must be re-answered.
    cache = _cache()
    before = cache.key("who is MS Dhoni", "en", "acct-a")
    cache.set(before, "stale")

    cache.invalidate("acct-a")

    assert cache.key("who is MS Dhoni", "en", "acct-a") != before
    assert cache.get(cache.key("who is MS Dhoni", "en", "acct-a")) is None


def test_invalidation_does_not_affect_other_accounts():
    cache = _cache()
    other = cache.key("what is DNA", "en", "acct-b")
    cache.set(other, "still good")

    cache.invalidate("acct-a")

    assert cache.get(cache.key("what is DNA", "en", "acct-b")) == "still good"


def test_cache_returns_stored_values_and_counts_hits():
    cache = TTLCache(Settings(openai_api_key="test", cache_ttl_seconds=60))
    cache.set("k", {"answer": "42"})

    assert cache.get("k") == {"answer": "42"}
    assert cache.get("missing") is None
    assert cache.stats() == {"entries": 1, "hits": 1, "misses": 1}


def test_expired_entries_are_dropped():
    cache = TTLCache(Settings(openai_api_key="test", cache_ttl_seconds=0))
    cache.set("k", "v")

    assert cache.get("k") is None
