"""Node 1: the gate seeds the budget and classifies nothing."""

from __future__ import annotations

import asyncio

import pytest

from app.graph.nodes import gate
from .stubs import StubServices, _state


@pytest.mark.asyncio
async def test_the_gate_classifies_nothing_and_sends_every_message_to_the_agent():
    """The regression this whole change exists to prevent.

    The gate used to answer "hey" from a phrase table and route opening
    questions straight to the index. Both were decisions about what the message
    *was*, made by string matching, and the table could only ever hold the
    messages someone had thought to list.
    """
    services = StubServices()

    result = await gate(_state(question="hey"), services)

    assert result["intent"] == "new_question"
    assert result.get("pending_tool") is None
    assert services.retrieve_calls == []
    assert services.decide_calls == []


@pytest.mark.asyncio
async def test_the_gate_treats_an_opening_question_like_any_other_message():
    services = StubServices(history=[])

    result = await gate(_state(question="what is photosynthesis"), services)

    assert result.get("pending_tool") is None
    assert services.decide_calls == []


@pytest.mark.asyncio
async def test_gate_copies_the_hop_budget_into_state():
    # A conditional edge sees only state, so the budget has to travel in it.
    services = StubServices()

    result = await gate(_state(), services)

    assert result["hop_deadline_ms"] == 1200
    assert result["max_hops"] == 2
