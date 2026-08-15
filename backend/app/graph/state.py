"""LangGraph state schema for the chat graph."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage

Intent = Literal["chitchat", "followup", "new_question", "ambiguous"]


class ChatState(TypedDict, total=False):
    """State threaded through every node.

    The first block is the schema from the specification. The second holds the
    runtime context a node needs (session, language) and the observability data
    the API surfaces — kept in state rather than in globals so a turn's timings
    travel with the turn and stay correct under concurrent requests.
    """

    # --- specified schema ---
    # `operator.add` gives append semantics; a plain list would make each node's
    # return overwrite the transcript instead of extending it.
    messages: Annotated[list[BaseMessage], operator.add]
    intent: Intent
    standalone_query: Optional[str]
    retrieved_chunks: list[dict]  # {text, article_title, url, section, score}
    disambiguation_candidates: Optional[list[dict]]
    # Where those candidates came from: "page" is a real Wikipedia
    # disambiguation page (indexed or fetched live), "related" is the weaker
    # nearest-topics fallback. The clarification node words itself differently
    # for each, because offering retrieved neighbours as if they were
    # disambiguation choices is what made "Venus" suggest *Plate tectonics*.
    disambiguation_kind: Optional[str]
    answer: str
    sources: list[dict]

    # --- runtime context ---
    session_id: str
    lang: str
    question: str  # the raw user message for this turn

    # --- outputs / observability ---
    # Raw SearchHit objects passed from retriever to generator. Kept out of the
    # serialised `retrieved_chunks` because the generator needs the embeddings
    # and metadata objects, while the API response needs plain dicts.
    hits: list
    articles: list[dict]  # smart-routing links
    used_live_search: bool
    # The reply the agent wrote when it chose `respond_directly`. It arrives on
    # the tool call rather than from a second generation call, which is what
    # keeps a greeting to one round trip.
    direct_reply: Optional[str]
    # `operator.add` makes this an accumulator: each node appends its own entry
    # instead of overwriting the previous node's.
    node_timings: Annotated[list[dict], operator.add]

    # --- agent loop ---
    # Which tool the agent chose this hop, and the query it chose it with. The
    # executor reads these; the API surfaces `tool_calls` so a turn's plan is
    # visible rather than inferred from timings.
    pending_tool: Optional[str]
    tool_query: Optional[str]
    tool_calls: Annotated[list[dict], operator.add]
    # What earlier hops in this turn found. The agent is shown these before
    # choosing again, because a model re-deciding on unchanged information
    # picks the same tool and the second hop buys nothing.
    observations: Annotated[list[dict], operator.add]
    hops: int
    # The hop budget, copied out of settings by the gate node: a conditional
    # edge receives only the state, with no services to read settings from.
    hop_deadline_ms: int
    max_hops: int
    # Wall clock (perf_counter) when the turn entered the agent, so the hop
    # gate measures the whole turn rather than the current node.
    started_at: float
    # An in-flight `services.retrieve` task on the raw message, launched before
    # the decision call and awaited only if the agent's query turns out to
    # match. Never serialised — this graph compiles without a checkpointer, so
    # state stays in-process for the life of one turn.
    speculative: Optional[object]
    speculative_query: Optional[str]
    # True when the executor served the agent from the parked speculative
    # result. Pinned by tests: the whole latency argument rests on this hitting.
    speculation_hit: bool
