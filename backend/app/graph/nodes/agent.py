"""Node 3: choose the tool for this hop, with retrieval already running underneath."""

from __future__ import annotations

import asyncio
import logging
import re

from ...core.agent_tools import DIRECT_REPLY, HISTORY_ANSWER
from ..history import substantive_turns
from ..references import _is_referential, _terms, looks_self_contained
from ..state import ChatState
from .timing import timed

logger = logging.getLogger(__name__)

@timed("agent")
async def agent(state: ChatState, services) -> dict:
    """Choose the tool for this hop, with retrieval already running underneath.

    The decision call takes about a second; a vector search takes about a third
    of one. Run them in sequence and the search is a second of added latency.
    Run the search speculatively on the raw message and it is usually finished
    and parked before the agent has finished choosing — so when the agent picks
    the knowledge base with substantially the query we guessed, the retrieval is
    free.

    Speculation is only worth it when the raw message is a plausible query.
    "What about his wife?" embeds to noise, so those turns skip it and pay the
    ordinary sequential retrieval — which is still no worse than the rewrite
    call they used to pay.
    """
    message = state["question"]
    history = services.history(state["session_id"])
    hops = state.get("hops", 0)

    speculative = state.get("speculative")
    updates: dict = {}

    # Only ever speculate before the first search. `tool_calls` is checked as
    # well as `hops` because a turn that has already run a search must not
    # launch another for a query we know comes back empty.
    if speculative is None and hops == 0 and not state.get("tool_calls"):
        guess = _speculation_query(message, history, services)
        if guess:
            speculative = asyncio.create_task(
                services.retrieve(guess, lang=state.get("lang", "en"))
            )
            updates["speculative"] = speculative
            updates["speculative_query"] = guess

    calls = await services.decide_tools(message, history, state.get("observations"))
    call = calls[0]

    query = call.query
    if call.name not in (HISTORY_ANSWER, DIRECT_REPLY):
        query = _resolve_against_current_subject(message, query, history)

    logger.info("agent hop=%d tool=%s query=%r", hops, call.name, query)

    if call.name == DIRECT_REPLY:
        updates["direct_reply"] = call.reply

    return {
        **updates,
        "pending_tool": call.name,
        "tool_query": query,
        "standalone_query": query or state.get("standalone_query"),
        "hops": hops + 1,
        "tool_calls": [{"tool": call.name, "query": query, "hop": hops}],
    }


def _resolve_against_current_subject(message: str, query: str, history: list) -> str:
    """Keep a referential follow-up pointed at the subject actually under discussion.

    The agent resolves pronouns well enough most of the time and wrongly often
    enough to matter. Asked "and his wife" after a turn about Einstein *and then*
    a turn about Dhoni, `gpt-4.1-nano` produced all three of these across runs:
    "MS Dhoni wife" (right), "his wife" (nothing resolved), and "Albert Einstein
    wife" (confidently the wrong person). The last is the dangerous one — it
    retrieves real chunks and cites them, so the answer looks grounded while
    being about someone the user stopped asking about two turns ago.

    So the agent's resolution is checked rather than trusted, on the one class of
    message where it can go wrong: a short referential one. The check is whether
    the subject it supplied appears in what the user recently asked about. If the
    agent invents a subject from nowhere in recent history, or resolves nothing
    at all, the previous question is stitched on instead — deterministic, and it
    reproduces the exact string speculation already searched, so being correct
    costs nothing extra.

    A resolution the previous question corroborates is left alone, which is most
    of them: "Who was Marie Curie?" -> "What did she discover?" -> "Marie Curie
    discoveries" shares *Marie Curie* with the question before it and passes
    untouched.

    Corroboration looks at the *immediately* previous question and no further
    back. Widening it to the last two was tried first and defeats the purpose:
    in the very conversation this exists for, Einstein *is* two questions back,
    so a two-question window corroborates the stale subject and waves it
    through. Chains still survive the narrow window, because a resolution built
    on a referential question tends to echo a word from it — "his wife" ->
    "Albert Einstein wife death" shares *wife* — so the corroboration lands
    without ever needing to see the name.
    """
    if not _is_referential(message) or len(re.findall(r"[a-z']+", message.lower())) > 8:
        return query

    previous = _last_user_message(history)
    if not previous:
        return query

    supplied = _terms(query) - _terms(message)
    if supplied and supplied & _terms(previous):
        return query

    reason = "resolved nothing" if not supplied else "named a stale subject"
    logger.info("Agent %s for %r; stitching to %r", reason, message, previous)
    return f"{previous} {message}"


def _speculation_query(message: str, history: list, services) -> str | None:
    """What to search while the agent decides, or None to not bother.

    Three cases, and the third is what makes follow-ups fast:

    * A message that already stands alone is its own best guess.
    * A referential one ("what happens at its event horizon?") embeds to noise
      on its own — but stitched to the question before it, it does not. The
      concatenation is a poor *question* and a perfectly good *search*: it
      carries the subject the pronoun refers to, which is the only thing
      retrieval was missing. The agent's own query ("black hole event horizon")
      is then contained in it, so the containment check accepts the parked
      result rather than paying for a second search.
    * Anything else — an elliptical message with no previous question to stitch
      to — is left alone, because there is nothing to build a query from.

    A wasted speculation costs an embedding call the user pays for, so this
    stays conservative; but it is now conservative about *what to search*
    rather than *whether to search at all*, which is where the follow-up
    latency was going.
    """
    if not getattr(services.settings, "agent_speculative_retrieval", True):
        return None
    if not history or looks_self_contained(message):
        return message

    previous = _last_user_message(history)
    if previous:
        return f"{previous} {message}"
    return None


def _last_user_message(history: list) -> str | None:
    """The last question worth stitching to — never a greeting.

    Stitching "Hey" onto "what about his wife?" produces a search for a
    greeting and a pronoun, which is strictly worse than not speculating.
    """
    turns = substantive_turns(history)
    return turns[-1].content if turns else None


def route_after_agent(state: ChatState) -> str:
    tool = state.get("pending_tool")
    if tool == DIRECT_REPLY:
        return "direct_responder"
    if tool == HISTORY_ANSWER:
        return "history_answerer"
    return "tool_executor"
