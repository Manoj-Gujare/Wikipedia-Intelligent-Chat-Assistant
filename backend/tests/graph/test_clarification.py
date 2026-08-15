"""Node 5: asking which entity was meant."""

from __future__ import annotations

import asyncio

import pytest

from app.graph.nodes import clarification
from .stubs import StubServices, _state


@pytest.mark.asyncio
async def test_clarification_lists_candidates_and_skips_generation():
    services = StubServices()
    candidates = [{"title": "Mercury (planet)", "url": "u1"},
                  {"title": "Mercury (element)", "url": "u2"}]

    result = await clarification(
        _state(question="Mercury", intent="ambiguous", disambiguation_candidates=candidates),
        services,
    )

    assert "Mercury (planet)" in result["answer"]
    assert services.generate_calls == 0


@pytest.mark.asyncio
async def test_related_fallback_does_not_claim_the_topics_are_word_senses():
    # "Venus" offering *Synodic day* as a meaning reads as broken. The wording
    # has to admit these are merely nearby topics.
    services = StubServices()
    candidates = [{"title": "Synodic day", "url": "u1"}, {"title": "Solar System", "url": "u2"}]

    result = await clarification(
        _state(
            question="Venus",
            intent="ambiguous",
            disambiguation_candidates=candidates,
            disambiguation_kind="related",
        ),
        services,
    )

    assert "could refer to" not in result["answer"]
    assert "closest" in result["answer"]
