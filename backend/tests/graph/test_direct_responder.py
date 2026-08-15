"""Node 2: the agent's own reply, emitted without a second model call."""

from __future__ import annotations

import asyncio

import pytest

from app.core.agent_tools import ToolCall, parse_tool_calls
from app.graph.nodes import direct_responder
from .stubs import StubServices, _Call, _Msg, _services_with, _state


@pytest.mark.asyncio
async def test_direct_responder_emits_the_agents_reply_verbatim():
    services = StubServices()

    result = await direct_responder(
        _state(question="hey", direct_reply="Hello! Ask me anything."), services
    )

    assert result["answer"] == "Hello! Ask me anything."
    assert result["intent"] == "chitchat"
    assert result["sources"] == []
    assert services.tokens == ["Hello! Ask me anything."]


@pytest.mark.asyncio
async def test_direct_responder_makes_no_call_of_its_own():
    """One round trip for a greeting: the agent decided and wrote it together.

    A second generation call here would double the latency of the cheapest turn
    in the app, which is the trade the old pre-warmed reply pool existed to
    avoid — and it is avoided now without a pool.
    """
    services = StubServices()

    await direct_responder(_state(question="hi", direct_reply="Hi there."), services)

    assert services.retrieve_calls == []
    assert services.generate_calls == 0
    assert services.decide_calls == []


@pytest.mark.asyncio
async def test_direct_responder_throws_away_the_speculative_search():
    """This branch wanted no evidence, so a parked retrieval must not leak."""
    services = StubServices()

    async def slow():
        await asyncio.sleep(5)

    parked = asyncio.create_task(slow())
    result = await direct_responder(
        _state(question="hi", direct_reply="Hi there.", speculative=parked), services
    )
    await asyncio.sleep(0)

    assert parked.cancelled()
    assert result["speculative"] is None


def test_respond_directly_carries_its_reply_on_the_tool_call():
    """Deciding and answering in one call is what keeps a greeting to one trip."""
    message = _Msg([_Call("respond_directly", '{"reply": "Hello! Ask me anything."}')])

    calls = parse_tool_calls(message, "hello")

    assert calls == [ToolCall("respond_directly", "", "Hello! Ask me anything.")]


def test_respond_directly_without_a_reply_falls_back_to_searching():
    """An empty reply leaves nothing to say, and an empty turn is the worst outcome.

    Falling through to the default search costs a greeting a stiff "not in the
    index" answer — poor, but an answer.
    """
    calls = parse_tool_calls(_Msg([_Call("respond_directly", "{}")]), "hello")

    assert calls == [ToolCall("search_knowledge_base", "hello")]


@pytest.mark.asyncio
async def test_respond_directly_is_offered_even_on_an_opening_message():
    """Small talk is usually the *first* thing said, so the tool must be there."""
    sink: list = []

    await _services_with(sink).decide_tools("hi", history=[])

    offered = {t["function"]["name"] for t in sink[0]["tools"]}
    assert "respond_directly" in offered
