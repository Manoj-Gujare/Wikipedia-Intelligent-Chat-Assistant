"""Failed-login throttling.

Password auth without a limiter is an online guessing oracle. bcrypt at work
factor 12 makes each guess expensive *for the server*, which is the point — but
nothing in the hash cost stops an attacker issuing those guesses concurrently.
A short lockout after a handful of failures removes the online attack while
staying invisible to someone who mistypes their own password once or twice.

Counted per (client, email) rather than per email alone: keying on the email
only would let anyone lock a known user out of their own account by failing
logins on their behalf, turning a defence into a denial of service.

State is in-process and per-worker, which matches where the rest of this app's
ephemeral state lives (the answer cache has the same shape and the same
caveat). Behind multiple workers each holds its own counter, so the effective
limit scales with worker count; a fleet would move this to Redis, the same swap
the conversation store documents.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)

MAX_FAILURES = 5
# Failures older than this stop counting, so occasional typos never accumulate
# into a lockout across a long session.
WINDOW_SECONDS = 300.0
LOCKOUT_SECONDS = 300.0
# Bound on distinct keys held, so a spray across many addresses cannot grow this
# without limit. Eviction drops the entries closest to expiry first.
MAX_TRACKED_KEYS = 4096


@dataclass
class _Attempts:
    failures: list[float] = field(default_factory=list)
    locked_until: float = 0.0


class LoginThrottle:
    """Sliding-window failure counter with a fixed lockout."""

    def __init__(
        self,
        max_failures: int = MAX_FAILURES,
        window_seconds: float = WINDOW_SECONDS,
        lockout_seconds: float = LOCKOUT_SECONDS,
    ) -> None:
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self.lockout_seconds = lockout_seconds
        self._entries: dict[str, _Attempts] = {}

    def retry_after(self, key: str, now: float | None = None) -> float:
        """Seconds until `key` may try again; 0 when it may try now."""
        now = time.monotonic() if now is None else now
        entry = self._entries.get(key)
        if entry is None:
            return 0.0
        remaining = entry.locked_until - now
        return remaining if remaining > 0 else 0.0

    def record_failure(self, key: str, now: float | None = None) -> float:
        """Count a failed attempt. Returns the lockout it triggered, if any."""
        now = time.monotonic() if now is None else now
        self._evict(now)
        entry = self._entries.setdefault(key, _Attempts())
        cutoff = now - self.window_seconds
        entry.failures = [t for t in entry.failures if t > cutoff]
        entry.failures.append(now)

        if len(entry.failures) >= self.max_failures:
            entry.locked_until = now + self.lockout_seconds
            # Cleared so the next lockout needs a fresh run of failures rather
            # than triggering again on the first attempt after expiry.
            entry.failures = []
            logger.warning("Login locked out for %ss (key=%s)", self.lockout_seconds, key)
            return self.lockout_seconds
        return 0.0

    def reset(self, key: str) -> None:
        """Forget a key's history — called on a successful sign-in."""
        self._entries.pop(key, None)

    def _evict(self, now: float) -> None:
        if len(self._entries) < MAX_TRACKED_KEYS:
            return
        cutoff = now - self.window_seconds
        stale = [
            k
            for k, e in self._entries.items()
            if e.locked_until <= now and not [t for t in e.failures if t > cutoff]
        ]
        for key in stale:
            del self._entries[key]
        # Still full of live entries: drop the ones expiring soonest.
        if len(self._entries) >= MAX_TRACKED_KEYS:
            ordered = sorted(self._entries.items(), key=lambda kv: kv[1].locked_until)
            for key, _ in ordered[: len(self._entries) - MAX_TRACKED_KEYS + 1]:
                del self._entries[key]


@lru_cache
def get_login_throttle() -> LoginThrottle:
    return LoginThrottle()
