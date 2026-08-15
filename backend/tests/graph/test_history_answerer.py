"""Node 6: answering from the conversation itself."""

from __future__ import annotations

import asyncio

import pytest

from app.graph.nodes import history_answerer
from .stubs import StubServices, Turn, _state


@pytest.mark.asyncio
async def test_history_answerer_cites_nothing_and_never_retrieves():
    services = StubServices(
        history=[Turn("user", "Who is Einstein"), Turn("assistant", "A physicist [1].")],
        answer="I told you Einstein was a physicist.",
    )

    result = await history_answerer(_state(question="what did you just say?"), services)

    assert result["sources"] == []
    assert services.retrieve_calls == []
    assert services.generate_calls == 1


@pytest.mark.asyncio
async def test_history_answerer_without_history_refuses_rather_than_inventing():
    # The agent can pick this on an opening message, where there is nothing to
    # summarise; generating anyway would answer from the model's own memory.
    services = StubServices(history=[])

    result = await history_answerer(_state(question="what did you just say?"), services)

    assert services.generate_calls == 0
    assert result["answer"]
