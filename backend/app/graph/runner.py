"""Drives the graph for one turn and shapes the result for the API."""

from __future__ import annotations

import logging
import time
import uuid
from typing import AsyncIterator

from langchain_core.messages import HumanMessage

from ..core.cache import TTLCache, get_cache
from ..core.sources import turn_meta
from ..models import (
    AgentTrace,
    ArticleLink,
    ChatResponse,
    DisambiguationOptionModel,
    Source,
    Timings,
    ToolCallTrace,
)
from .build import get_graph, get_services
from .services import GraphServices
from .state import ChatState

logger = logging.getLogger(__name__)


def _initial_state(message: str, session_id: str, lang: str) -> ChatState:
    return {
        "messages": [HumanMessage(content=message)],
        "question": message,
        "session_id": session_id,
        "lang": lang,
        "intent": "new_question",
        "standalone_query": None,
        "retrieved_chunks": [],
        "disambiguation_candidates": None,
        "disambiguation_kind": None,
        "answer": "",
        "sources": [],
        "articles": [],
        "hits": [],
        "used_live_search": False,
        "direct_reply": None,
        "node_timings": [],
        "pending_tool": None,
        "tool_query": None,
        "tool_calls": [],
        "observations": [],
        "hops": 0,
        "speculative": None,
        "speculative_query": None,
        "speculation_hit": False,
        "speculative_answer": None,
        "speculative_answer_used": None,
    }


def _to_response(
    state: dict, session_id: str, message: str, lang: str, total_ms: int
) -> ChatResponse:
    timings = Timings(total_ms=total_ms)
    for entry in state.get("node_timings") or []:
        node, ms = entry["node"], int(entry["ms"])
        # `agent` reports as rewrite time: it is the call that replaced the
        # rewrite, and it is where a follow-up's reference resolution now
        # happens. Both accumulate, because a second hop visits each again and
        # the field should show what the turn spent, not what its last hop did.
        if node == "agent":
            timings.rewrite_ms = (timings.rewrite_ms or 0) + ms
        elif node == "tool_executor":
            timings.retrieval_ms = (timings.retrieval_ms or 0) + ms
        elif node in (
            "generator",
            "direct_responder",
            "clarification",
            "history_answerer",
        ):
            timings.generation_ms = ms

    standalone = state.get("standalone_query")
    trace = AgentTrace(
        tools=[ToolCallTrace(**c) for c in state.get("tool_calls") or []],
        hops=state.get("hops", 0),
        speculation_hit=bool(state.get("speculation_hit")),
        speculative_answer_used=state.get("speculative_answer_used"),
    )
    return ChatResponse(
        conversation_id=session_id,
        answer=state.get("answer", ""),
        sources=[Source(**s) for s in state.get("sources") or []],
        articles=[ArticleLink(**a) for a in state.get("articles") or []],
        disambiguation=[
            DisambiguationOptionModel(**c)
            for c in state.get("disambiguation_candidates") or []
        ],
        disambiguation_term=message if state.get("intent") == "ambiguous" else None,
        lang=lang,
        resolved_query=standalone if standalone and standalone != message else None,
        used_live_search=bool(state.get("used_live_search")),
        timings=timings,
        agent=trace,
    )


def _turn_meta(state: dict) -> dict | None:
    sources = state.get("sources") or []
    articles = state.get("articles") or []
    if not sources and not articles:
        return None
    return {"sources": sources, "articles": articles}


def _cache_key(
    message: str, lang: str, has_history: bool, namespace: str
) -> str | None:
    """Only opening questions are cacheable — later turns depend on history."""
    return None if has_history else get_cache().key(message, lang, namespace)


async def run_turn(
    message: str,
    session_id: str | None = None,
    lang: str = "en",
    services: "GraphServices | None" = None,
) -> ChatResponse:
    """Non-streaming turn: run the graph to completion."""
    services = services or get_services()
    conversation = services.conversations.get_or_create(
        session_id, lang=lang, account_id=services.namespace or ""
    )
    services.conversations.set_title(conversation.id, message)
    started = time.perf_counter()

    key = _cache_key(
        message,
        lang,
        bool(services.history(conversation.id)),
        services.namespace or "",
    )
    if key:
        cached = get_cache().get(key)
        if cached is not None:
            response = cached.model_copy(
                update={"conversation_id": conversation.id, "cached": True}
            )
            services.record(
                conversation.id, message, response.answer, turn_meta(response)
            )
            return response

    state = await get_graph().ainvoke(
        _initial_state(message, conversation.id, lang),
        config={"configurable": {"services": services}},
    )

    total_ms = int((time.perf_counter() - started) * 1000)
    response = _to_response(state, conversation.id, message, lang, total_ms)
    services.record(
        conversation.id,
        message,
        response.answer,
        _turn_meta(state),
        chitchat=state.get("intent") == "chitchat",
    )
    _log_path(state, total_ms)
    # Chitchat is already sub-millisecond; caching it would only add churn.
    if key and state.get("intent") != "chitchat":
        get_cache().set(key, response)
    return response


async def stream_turn(
    message: str,
    session_id: str | None = None,
    lang: str = "en",
    services: "GraphServices | None" = None,
) -> AsyncIterator[tuple[str, dict]]:
    """Streaming turn, yielding ``(event, payload)`` pairs for SSE.

    ``stream_mode=["custom", "values"]`` gives both the tokens nodes emit as
    they generate and the final accumulated state, in one pass over the graph.
    """
    services = services or get_services()
    conversation = services.conversations.get_or_create(
        session_id, lang=lang, account_id=services.namespace or ""
    )
    services.conversations.set_title(conversation.id, message)
    started = time.perf_counter()

    yield "meta", {"conversation_id": conversation.id, "lang": lang}

    final: dict = {}
    announced_intent = False
    announced_articles = False

    async for mode, chunk in get_graph().astream(
        _initial_state(message, conversation.id, lang),
        stream_mode=["custom", "values"],
        config={"configurable": {"services": services}},
    ):
        if mode == "custom":
            if chunk.get("type") == "token":
                yield "token", {"text": chunk["text"]}
            continue

        final = chunk

        # Tell the UI which path this turn took. The gate no longer knows —
        # every turn leaves it as "new_question" and only the responding node
        # settles it — so this is now reported for observability rather than
        # used to change what the UI renders while waiting.
        if not announced_intent and chunk.get("intent"):
            announced_intent = True
            yield "intent", {"intent": chunk["intent"]}

        if chunk.get("standalone_query") and chunk["standalone_query"] != message:
            yield "rewrite", {"resolved_query": chunk["standalone_query"]}

        if chunk.get("retrieved_chunks") and not announced_articles:
            announced_articles = True
            yield "retrieval", {"chunks": len(chunk["retrieved_chunks"])}

    total_ms = int((time.perf_counter() - started) * 1000)
    response = _to_response(final, conversation.id, message, lang, total_ms)
    services.record(
        conversation.id,
        message,
        response.answer,
        _turn_meta(final),
        chitchat=final.get("intent") == "chitchat",
    )
    _log_path(final, total_ms)

    yield "done", response.model_dump()


def _log_path(state: dict, total_ms: int) -> None:
    path = " → ".join(e["node"] for e in state.get("node_timings") or [])
    calls = state.get("tool_calls") or []
    tools = ",".join(c["tool"] for c in calls) or "none"
    # Whether retrieval ran underneath the decision call or after it. Every
    # turn pays the decision now, so this is the whole latency design in one
    # field: `hit` means the search was already done when the agent chose.
    spec = "hit" if state.get("speculation_hit") else "-"
    # Whether the answer written under the decision call was used or thrown
    # away. This is the field that shows the real-traffic split between the two
    # — the cost of speculating on generation is entirely in the `discard`
    # column, so it should not have to be inferred from spend.
    used = state.get("speculative_answer_used")
    gen = {True: "flush", False: "discard", None: "-"}[used]
    logger.info(
        "intent=%-12s tools=%-24s spec=%-3s gen=%-7s total=%5dms  path: %s",
        state.get("intent", "?"),
        tools,
        spec,
        gen,
        total_ms,
        path,
    )
