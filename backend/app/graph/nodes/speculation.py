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

import asyncio
import contextlib
import logging
import re
from dataclasses import dataclass

from ..state import ChatState

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SpeculativeAnswer:
    """A complete, unflushed answer and the chunks it was written from.

    `hits` is carried so the generator can prove the answer was written from
    the very chunks the confirmed path ended up with — identity, not equality,
    because the reused result is the same object. Without that check a
    speculative answer could be flushed for a retrieval it was not based on,
    which is the one failure this whole mechanism must not have.
    """

    hits: list
    text: str


async def speculate_answer(retrieval_task, services, question, lang, history):
    """Write the answer under the decision call, buffered rather than streamed.

    The decision call and the answer are the two serial round trips in a turn,
    and each has a floor of roughly 0.7s. They are only *logically* serial for
    the 22% of turns that route somewhere other than the knowledge base — for
    the rest, the answer could have been written while the decision was still
    being made. This writes it, and the generator decides afterwards whether
    anyone gets to see it.

    Retrieval is still upstream of generation, so this cannot start at t=0: it
    waits on the parked search, which contains an embedding round trip of its
    own. The saving is therefore the decision call *minus* retrieval, not the
    decision call outright.

    Returns None whenever the speculation is unusable — no chunks, an empty
    generation, or any failure at all. A speculative answer must never be able
    to break a turn that would otherwise have succeeded, so every error here
    ends as "no speculation" and the ordinary path runs.
    """
    # Local import: `generator` imports `_discard` from this module, so a
    # module-level import here would close the cycle.
    from .generator import build_generation_messages

    try:
        result = await retrieval_task
        hits = result.hits
        if not hits:
            return None
        text = await services.generate_stream(
            build_generation_messages(question, lang, hits, history), emit=False
        )
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - speculation must never fail the turn
        logger.warning("Speculative generation failed; the ordinary path will run")
        return None

    return SpeculativeAnswer(hits=hits, text=text) if text else None


async def collect(task):
    """Await a speculative answer, or None if it did not produce one.

    Only called once the cheap preconditions already hold, so waiting here is
    waiting for something we intend to use — never for something we are about
    to throw away.

    `CancelledError` is deliberately not caught. Catching it was tried and is
    wrong twice over: it swallows the cancellation of the *turn* — a
    disconnected client would leave the graph running with nowhere to send the
    answer — and it cannot even tell the two cases apart, because cancelling a
    task that is awaiting another task cancels the awaited one too (it is the
    outer task's `_fut_waiter`), so `task.cancelled()` reads True either way.
    The honest semantics are the simple ones: if the turn is cancelled while
    waiting for the answer it means to use, the turn is over.
    """
    if task is None:
        return None
    try:
        return await task
    except Exception:  # noqa: BLE001 - a failed speculation is just no speculation
        return None

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

    Deliberately does *not* stem, though `citations._term_present` does. Making
    membership plural-tolerant here was tried and measured: it turned the one
    remaining miss on the smoke path ("about black hole" -> "who found it",
    parked against the agent's "who discovered black holes") from 0.50 into
    0.75, saved the ~0.5s second search on every run — and turned a cited
    answer into a refusal on every run, because the stitched query retrieves
    general black-hole chunks and the discovery history only surfaces for the
    agent's own wording. The words were all covered; the *chunks* were not.
    That gap is the thing this function cannot see, so the slack in the
    threshold is what stands in for it.
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
