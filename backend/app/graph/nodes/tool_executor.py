"""Node 4: run the tool the agent chose, reusing the speculative search if it fits.

Also owns the hop gate (`route_after_tools`): an empty result can send the turn
back to the agent to try a different source, authorised by the wall clock
rather than by a hop count.
"""

from __future__ import annotations

import logging
import time

from ...core.agent_tools import KB_SEARCH, LIVE_SEARCH
from ...core.wikipedia import WikipediaClient
from ..state import ChatState
from .speculation import _discard, _kb_search
from .timing import timed

logger = logging.getLogger(__name__)

# Ambiguity by score proximity is only considered for bare entity mentions
# ("Mercury", "Java"). A real question legitimately spans several articles, so
# applying this to long queries would fire constantly on healthy retrievals.
AMBIGUITY_MAX_WORDS = 3


AMBIGUITY_SCORE_GAP = 0.02


# The top hit must actually be a good match before near-ties mean anything.
# On a miss every weak hit scores about the same, which is indistinguishable
# from ambiguity by gap alone — that produced "MS Dhoni could refer to India,
# Mahatma Gandhi, Hindi cinema". Nothing matching is a miss, not a choice.
AMBIGUITY_MIN_SCORE = 0.45


@timed("tool_executor")
async def tool_executor(state: ChatState, services) -> dict:
    """Run the tool the agent chose, reusing the speculative search if it fits."""
    tool = state.get("pending_tool") or KB_SEARCH
    query = state.get("tool_query") or state["question"]
    lang = state.get("lang", "en")

    if tool == LIVE_SEARCH:
        _discard(state.get("speculative"))
        result = await services.search_wikipedia_live(query, lang=lang)
        shaped = _shape(result, lang, live=True)
        return {**shaped, "speculative": None, "observations": _observe(tool, query, shaped)}

    result, reused = await _kb_search(state, services, query, lang)
    shaped = _shape(result, lang, live=False)

    # Ambiguity is a property of *ranking*: several indexed articles matching a
    # bare term equally well. It is checked only for knowledge-base results —
    # live hits carry a nominal score with nothing behind it, so every pair of
    # them reads as tied and the gap test would call each live search ambiguous
    # by construction.
    #
    # Tested against the *user's* words, never the agent's query. The word-count
    # gate below encodes "a bare entity mention", which is a property of what
    # someone typed; the agent routinely compresses a full question into two
    # keywords, and scoring those made "What does photosynthesis produce?" ->
    # "photosynthesis products" trip a threshold written for people typing
    # "Mercury". The result was a disambiguation prompt for a question the
    # index answers outright.
    if shaped.get("hits") and _is_ambiguous_by_score(state["question"], result.hits):
        # The index has several equally-good matches for a bare entity but no
        # disambiguation page for it — otherwise `_shape` would have caught it.
        # Ask Wikipedia for the real options before falling back to the nearest
        # indexed topics, which are neighbours in embedding space and not
        # choices a reader would recognise.
        candidates = await services.live_disambiguation(query, lang)
        kind = "page"
        if not candidates:
            candidates = _distinct_articles(result.hits, lang)
            kind = "related"
        logger.info(
            "ambiguous entity %r -> %d candidates (%s)", query, len(candidates), kind
        )
        return {
            "intent": "ambiguous",
            "retrieved_chunks": shaped["retrieved_chunks"],
            "disambiguation_kind": kind,
            "disambiguation_candidates": candidates,
            "speculative": None,
        }

    return {
        **shaped,
        "speculation_hit": reused,
        "speculative": None,
        "observations": _observe(tool, query, shaped),
    }


def _observe(tool: str, query: str, shaped: dict) -> list[dict]:
    """What this hop found, in one line, for the agent's next decision.

    Only recorded on an empty result. A hop that found evidence goes straight
    to the generator and never re-enters the agent, so describing a successful
    search to a model that will not see it again is pure token spend.
    """
    if shaped.get("hits"):
        return []
    return [{"tool": tool, "query": query, "outcome": "no relevant results"}]


def _is_ambiguous_by_score(query: str, hits: list) -> bool:
    """True when a bare entity mention matches unrelated articles equally well.

    Only bare mentions qualify. "What is the periodic table organised by?"
    spanning several articles is healthy retrieval, not ambiguity.
    """
    if len(query.split()) > AMBIGUITY_MAX_WORDS or len(hits) < 2:
        return False

    # Raw similarity, not the ranked score: the lead-section boost is a display
    # preference, and letting it move these numbers made ambiguity detection a
    # side effect of retrieval tuning.
    top, second = hits[0].raw_score, hits[1].raw_score

    # A weak top hit means we found nothing, not that we found several things.
    if top < AMBIGUITY_MIN_SCORE:
        return False

    titles = {h.title for h in hits[:3]}
    if len(titles) < 2:
        return False
    return abs(top - second) < AMBIGUITY_SCORE_GAP


def _distinct_articles(hits: list, lang: str) -> list[dict]:
    seen: set[str] = set()
    candidates: list[dict] = []
    for hit in hits:
        if hit.title in seen:
            continue
        seen.add(hit.title)
        candidates.append(
            {
                "title": hit.title,
                "url": hit.metadata.get("url")
                or WikipediaClient.article_url(lang, hit.title),
            }
        )
    return candidates[:6]


def _shape(result, lang: str, live: bool) -> dict:
    """Turn a RetrievalResult into state updates, flagging ambiguity."""
    if result.disambiguation:
        return {
            "intent": "ambiguous",
            "retrieved_chunks": [],
            "disambiguation_kind": "page",
            "disambiguation_candidates": [
                {"title": o.title, "url": o.url} for o in result.disambiguation
            ],
        }

    chunks = [
        {
            "text": hit.metadata.get("body") or hit.text,
            "article_title": hit.title,
            "url": hit.section_url,
            "article_url": hit.metadata.get("url", ""),
            "section": hit.metadata.get("section_title", ""),
            "score": round(hit.score, 4),
        }
        for hit in result.hits
    ]

    if not chunks:
        return {
            "retrieved_chunks": [],
            "hits": [],
            "used_live_search": result.used_live_search or live,
            "articles": [
                {"title": a["title"], "url": a["url"], "lang": lang, "summary": ""}
                for a in result.suggested_articles
            ],
        }

    return {"retrieved_chunks": chunks, "hits": result.hits, "used_live_search": live}


def route_after_tools(state: ChatState) -> str:
    """Clarify, hop again, or answer with what we have.

    The hop gate is wall-clock: a second search plus generation needs roughly
    1.9s, so authorising one at 1.2s lands near 3.1s and authorising one at
    1.6s does not. A hop *count* cannot tell those apart — it would let a slow
    first call spend the rest of the budget it already overran.
    """
    if state.get("intent") == "ambiguous":
        return "clarification"

    if state.get("hits"):
        return "generator"

    deadline = state.get("hop_deadline_ms", 1200)
    max_hops = state.get("max_hops", 2)

    elapsed_ms = (time.perf_counter() - state.get("started_at", 0)) * 1000
    if state.get("hops", 0) < max_hops and elapsed_ms < deadline:
        logger.info(
            "empty retrieval at %.0fms; hopping again (budget %dms)", elapsed_ms, deadline
        )
        return "agent"

    logger.info(
        "empty retrieval at %.0fms; answering without another hop", elapsed_ms
    )
    return "generator"
