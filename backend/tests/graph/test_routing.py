"""End-to-end routing through the compiled graph."""

from __future__ import annotations

import asyncio

import pytest

from app.core.agent_tools import ToolCall
from app.graph.build import build_graph
from app.core.retriever import DisambiguationOption, RetrievalResult
from .stubs import StubServices, Turn, _hit, _state


def _graph_state(**overrides):
    """Initial state for a full graph run, mirroring the runner's."""
    base = _state(**overrides)
    base.setdefault("tool_calls", [])
    base.setdefault("observations", [])
    base.setdefault("speculative", None)
    return base


@pytest.mark.asyncio
async def test_a_greeting_reaches_end_without_retrieval_or_generation():
    """No vector search and no generation — but the model did decide.

    That is the trade this design makes: a greeting pays one decision call
    instead of zero, and in exchange the set of messages that get a
    conversational answer is whatever the model recognises rather than whatever
    a phrase table lists.
    """
    services = StubServices(
        tool_call=ToolCall("respond_directly", "", "Hello! What would you like to know?")
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question="hello"))

    assert final["intent"] == "chitchat"
    assert final["answer"] == "Hello! What would you like to know?"
    assert services.generate_calls == 0
    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited == ["gate", "agent", "direct_responder"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        # The message that started this: it matched no greeting pattern, so it
        # was searched for in a Wikipedia index and the user was told their own
        # question was not covered by the corpus.
        "do you know me",
        "who are you",
        "thanks a million, that was helpful",
        "are you an AI?",
    ],
)
async def test_a_message_about_the_assistant_never_reaches_the_index(message):
    services = StubServices(
        result=RetrievalResult(hits=[_hit("Albert Einstein")]),
        tool_call=ToolCall("respond_directly", "", "I answer from Wikipedia articles."),
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question=message))

    assert final["answer"] == "I answer from Wikipedia articles."
    # A speculative search does fire under the decision call and is thrown
    # away — that is the cost of racing retrieval rather than sequencing it.
    # What must never happen is any of it reaching the user: no chunks, no
    # citations, and no trip through the executor or the generator.
    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited == ["gate", "agent", "direct_responder"]
    assert final["sources"] == []
    assert final["retrieved_chunks"] == []


@pytest.mark.asyncio
async def test_an_opening_question_still_retrieves_and_answers():
    services = StubServices(
        history=[],
        result=RetrievalResult(hits=[_hit()]),
        tool_call=ToolCall("search_knowledge_base", "what is physics"),
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question="what is physics"))

    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited == ["gate", "agent", "tool_executor", "generator"]
    # The search ran underneath the decision call rather than after it, so the
    # question paid for the decision and got its retrieval free.
    assert final["speculation_hit"] is True
    assert services.retrieve_calls == ["what is physics"]


@pytest.mark.asyncio
async def test_a_followup_retrieves_on_the_agents_resolved_query():
    services = StubServices(
        history=[Turn("user", "Tell me about India")],
        result=RetrievalResult(hits=[_hit("India")]),
        tool_call=ToolCall("search_knowledge_base", "India population"),
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question="what's its population?"))

    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited == ["gate", "agent", "tool_executor", "generator"]
    # The pronoun-laden original never reached the index on its own: the
    # speculation stitched it to its subject, and the agent's resolved query
    # ("India population") is contained in that, so the parked search was
    # reused rather than repeated.
    assert services.retrieve_calls == ["Tell me about India what's its population?"]
    assert final["speculation_hit"] is True


@pytest.mark.asyncio
async def test_an_empty_index_hops_to_live_wikipedia_and_answers():
    """The case the old graph could not express: retry against a different source."""
    services = StubServices(
        history=[],
        result=RetrievalResult(hits=[]),
        live_result=RetrievalResult(hits=[_hit("Recent event")], used_live_search=True),
        tool_call=[
            ToolCall("search_knowledge_base", "saturn rings"),
            # Told the index came back empty, the second hop looks elsewhere.
            ToolCall("search_wikipedia", "saturn rings"),
        ],
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question="what are saturn's rings"))

    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited == [
        "gate",
        "agent",
        "tool_executor",
        "agent",
        "tool_executor",
        "generator",
    ]
    assert services.live_search_calls == ["saturn rings"]
    # The second decision saw what the first search found.
    assert services.decide_calls[1] == [
        {
            "tool": "search_knowledge_base",
            "query": "saturn rings",
            "outcome": "no relevant results",
        }
    ]


@pytest.mark.asyncio
async def test_the_hop_loop_terminates_when_every_source_comes_back_empty():
    services = StubServices(
        history=[],
        result=RetrievalResult(hits=[]),
        live_result=RetrievalResult(hits=[], used_live_search=True),
        tool_call=[
            ToolCall("search_knowledge_base", "nonsense"),
            ToolCall("search_wikipedia", "nonsense"),
        ],
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question="what is a quantum widget"))

    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited.count("agent") == 2  # max_hops, then it stops
    assert visited[-1] == "generator"
    assert final["answer"]


@pytest.mark.asyncio
async def test_ambiguous_retrieval_diverts_to_clarification_not_generation():
    services = StubServices(
        history=[],
        result=RetrievalResult(
            disambiguation=[DisambiguationOption("Mercury (planet)", "u1")]
        ),
        tool_call=ToolCall("search_knowledge_base", "Mercury"),
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question="Mercury"))

    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited == ["gate", "agent", "tool_executor", "clarification"]
    assert services.generate_calls == 0


@pytest.mark.asyncio
async def test_answer_from_history_skips_retrieval_entirely():
    services = StubServices(
        history=[Turn("user", "Who is Einstein"), Turn("assistant", "A physicist.")],
        tool_call=ToolCall("answer_from_history", ""),
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question="what did you just tell me?"))

    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited == ["gate", "agent", "history_answerer"]
    # No *tool* ran, and nothing was fetched from Wikipedia. The speculation
    # did fire and was thrown away — the one path where racing the decision
    # call costs an embedding for nothing. It is rare enough to be worth the
    # win on every other follow-up, and cheap enough not to show in the turn,
    # since it never blocked anything.
    assert services.live_search_calls == []
    assert final["answer"]
