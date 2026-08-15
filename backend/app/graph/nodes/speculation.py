"""Retrieval that runs underneath the agent's decision call.

The decision takes about a second and a vector search about a third of one.
Run in sequence, the search is a second of added latency; launched
speculatively on the raw message, it is usually finished and parked before the
agent has chosen, so a real question gets its evidence for free.

The parked result is only reused when the agent's own query is already covered
by what we searched -- serving chunks retrieved for a pronoun would be a worse
failure than the latency it saves, because it looks like a successful
retrieval.
"""

from __future__ import annotations

import contextlib
import logging
import re

from ..state import ChatState

logger = logging.getLogger(__name__)

def _discard(task) -> None:
    """Cancel an unused speculative task and swallow whatever it was doing.

    Without this, a cancelled or failed retrieval surfaces as an unretrieved
    task exception on the event loop — a stack trace printed under a turn that
    actually succeeded, which is worse than useless when reading logs.

    The suppression has to reach `BaseException`: `.exception()` on a cancelled
    task *raises* `CancelledError`, and that inherits from `BaseException`, not
    `Exception`. Catching only `Exception` here logged a traceback on every
    speculation miss — the exact noise this function exists to prevent.
    """
    if task is None or not hasattr(task, "cancel"):
        return
    task.cancel()

    def _swallow(finished) -> None:
        with contextlib.suppress(BaseException):
            finished.exception()

    task.add_done_callback(_swallow)


def _similar(speculated: str, chosen: str, threshold: float) -> bool:
    """How much of the agent's query the speculated message already contained.

    Containment, not Jaccard, and the asymmetry is the point. What the agent
    does to a standalone question is *strip* it: "What is the event horizon of
    a black hole?" comes back as "event horizon of a black hole". Those two
    retrieve the same chunks, but symmetric overlap scores them 0.67 — the
    filler words the agent removed count against the match — so a Jaccard gate
    at any sane threshold rejected every real question this was built to
    accelerate. Measured on the smoke suite: 0/3 hits with Jaccard, 2/3 with
    containment.

    Asking instead "is the agent's query already covered by what we searched?"
    scores that pair 1.0, while a genuine rewrite still fails: "what about his
    wife?" -> "Albert Einstein wife" shares only *wife*, or 0.33. Word sets, no
    stopword list, so it behaves the same in every language the index holds.
    """
    parked = set(re.findall(r"[\w']+", speculated.lower()))
    wanted = set(re.findall(r"[\w']+", chosen.lower()))
    if not parked or not wanted:
        return False
    return len(wanted & parked) / len(wanted) >= threshold


async def _kb_search(state: ChatState, services, query: str, lang: str):
    """Await the parked speculative search, or run a fresh one.

    The parked result is reused only when the agent's query is close enough to
    the message it was launched on. A rewrite that changed the subject —
    "what about his wife?" becoming "Albert Einstein wife" — must not be served
    chunks retrieved for the pronoun; that would be a *worse* failure than the
    latency we are avoiding, because it looks like a successful retrieval.
    """
    task = state.get("speculative")
    if task is None:
        return await services.retrieve(query, lang=lang), False

    parked_query = state.get("speculative_query") or ""
    threshold = getattr(services.settings, "agent_speculation_reuse_threshold", 0.7)
    if _similar(parked_query, query, threshold):
        try:
            result = await task
        except Exception:  # noqa: BLE001 - speculation must never fail the turn
            logger.warning("Speculative retrieval failed; searching again")
            return await services.retrieve(query, lang=lang), False
        logger.info("speculation hit: reused search for %r", parked_query)
        return result, True

    logger.info("speculation miss: %r -> %r", parked_query, query)
    _discard(task)
    return await services.retrieve(query, lang=lang), False
