"""Node 1: seed the turn's budget. Every message goes to the agent from here."""

from __future__ import annotations

import time

from ..state import ChatState
from .timing import timed

@timed("gate")
async def gate(state: ChatState, services) -> dict:
    """Seed the turn's budget. Every message goes to the agent from here.

    This node used to divert small talk with a table of phrases and regexes,
    and to send opening questions straight to the index without a decision
    call. Both were latency wins bought with a classifier that could only
    recognise the messages someone had thought to list. "Do you know me?"
    matched nothing, so it was searched for in a Wikipedia index, and the user
    was told their own question was not covered by the corpus.

    The agent decides now, because the set of messages that need no evidence
    is open-ended and a phrase list is the wrong shape for it. Retrieval is
    what makes that affordable: it runs speculatively *underneath* the
    decision call (see `agent`), so a real question pays the decision and gets
    its search for free, and a greeting pays the decision and throws an
    embedding away.
    """
    # The hop budget is copied into state here because a conditional edge sees
    # only the state — it has no services and no config to read settings from.
    settings = services.settings
    return {
        "intent": "new_question",
        "started_at": time.perf_counter(),
        "hops": 0,
        "hop_deadline_ms": getattr(settings, "agent_hop_deadline_ms", 1200),
        "max_hops": getattr(settings, "agent_max_hops", 2),
    }
