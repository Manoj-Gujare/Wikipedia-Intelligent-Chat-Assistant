"""Per-node wall-clock timing."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Awaitable, Callable

from ..state import ChatState

logger = logging.getLogger(__name__)

def timed(name: str):
    """Record per-node wall time into state and the log.

    Node-level timing is what proves the fast path is real: a chitchat turn
    should show one node and no retrieval, and any regression shows up as a
    node that suddenly appears where it should not.
    """

    def decorator(fn: Callable[..., Awaitable[dict]]):
        @functools.wraps(fn)
        async def wrapper(state: ChatState, *args: Any) -> dict:
            started = time.perf_counter()
            result = await fn(state, *args) or {}
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            logger.info("node=%-20s %7.1fms", name, elapsed)
            return {**result, "node_timings": [{"node": name, "ms": elapsed}]}

        return wrapper

    return decorator
