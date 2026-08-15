"""Client-side throttling, so we stay inside Wikimedia's expectations."""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Simple async token bucket shared by every request from this client."""

    def __init__(self, rate_per_second: float) -> None:
        self._rate = max(rate_per_second, 0.5)
        self._interval = 1.0 / self._rate
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait_for = self._next_slot - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
                now = time.monotonic()
            self._next_slot = now + self._interval
