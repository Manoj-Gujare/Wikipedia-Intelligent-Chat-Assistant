"""Graph wiring.

    START
      → gate
          → agent   (retrieval already running underneath the decision)
              ├─ respond_directly    → direct_responder  ─────────────→ END
              ├─ answer_from_history → history_answerer  ─────────────→ END
              └─ search_*            → tool_executor
                                          ├─ ambiguous → clarification → END
                                          ├─ nothing found, budget left → agent
                                          └─ → generator ─────────────→ END

Every message now reaches the agent, and that is a deliberate reversal. Two
branches used to route around it: a phrase table answered small talk without
any model call, and opening questions went straight to the index without a
decision call. Both were real latency wins, and both were bought by deciding
in Python what the message *was* — which worked exactly as well as the list of
phrases someone had thought to write down. "Do you know me?" was on no list,
so it was searched for in an encyclopedia.

What pays for the uniformity is speculative retrieval (see `nodes.agent`): the
search runs underneath the decision call rather than after it, so a question
gets its evidence for free and only the decision itself is on the clock. A
greeting costs that one call and throws an embedding away.

The loop back into `agent` is what makes this agentic rather than merely
routed — resolving a reference, or picking a different source after a miss.
It is gated on the wall clock, so it cannot spend a budget the turn has
already overrun.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .nodes import (
    agent,
    clarification,
    direct_responder,
    gate,
    generator,
    history_answerer,
    route_after_agent,
    route_after_tools,
    tool_executor,
)
from ..core.embeddings import client_for_key
from .services import GraphServices
from .state import ChatState


def build_graph(services: GraphServices | None = None):
    """Compile the StateGraph.

    With `services` supplied the graph is bound to that instance — used by tests
    to inject a stub. With none, each node reads the per-request services out of
    the run config, so a single compiled graph serves every account without
    leaking one caller's credentials into another's turn.
    """
    graph = StateGraph(ChatState)

    def bind(fn):
        async def node(state: ChatState, config) -> dict:
            resolved = services or (config or {}).get("configurable", {}).get("services")
            if resolved is None:
                raise RuntimeError("No GraphServices supplied for this run")
            return await fn(state, resolved)

        node.__name__ = fn.__name__
        return node

    graph.add_node("gate", bind(gate))
    graph.add_node("direct_responder", bind(direct_responder))
    graph.add_node("agent", bind(agent))
    graph.add_node("tool_executor", bind(tool_executor))
    graph.add_node("clarification", bind(clarification))
    graph.add_node("history_answerer", bind(history_answerer))
    graph.add_node("generator", bind(generator))

    graph.add_edge(START, "gate")
    # Unconditional: the gate only seeds the turn's budget now. Deciding what
    # the message *is* belongs to the agent, which is the one place that can
    # read it rather than pattern-match it.
    graph.add_edge("gate", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tool_executor": "tool_executor",
            "history_answerer": "history_answerer",
            # The agent wrote the reply itself; this node only emits it.
            "direct_responder": "direct_responder",
        },
    )
    # The cycle: an empty result can send the turn back to the agent to try a
    # different source. `route_after_tools` decides on elapsed wall time, and
    # `max_hops` is the backstop that terminates it regardless.
    graph.add_conditional_edges(
        "tool_executor",
        route_after_tools,
        {
            "clarification": "clarification",
            "agent": "agent",
            "generator": "generator",
        },
    )

    graph.add_edge("direct_responder", END)
    graph.add_edge("clarification", END)
    graph.add_edge("history_answerer", END)
    graph.add_edge("generator", END)

    return graph.compile()


def services_for(api_key: str | None = None, namespace: str | None = None) -> GraphServices:
    """Services for one request, carrying that caller's key and namespace."""
    client = client_for_key(api_key) if api_key else None
    return GraphServices(client=client, namespace=namespace)


@lru_cache
def get_services() -> GraphServices:
    """Server-key services for operator tooling (evaluation, warmup)."""
    return GraphServices()


@lru_cache
def get_graph():
    """One compiled graph for the whole process; services arrive per run."""
    return build_graph()
