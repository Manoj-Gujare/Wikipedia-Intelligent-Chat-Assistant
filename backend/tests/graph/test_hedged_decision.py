"""The decision call's tail-latency hedge.

The 3s budget is missed by stalled requests rather than slow work: one measured
turn spent 10,296ms of its 10,320ms inside a single decision call that took
1.6s on the very next message. These tests pin the behaviour that turns that
stall back into a normal turn — and, just as importantly, pin that the hedge
stays out of the way when nothing is slow.
"""

from __future__ import annotations

import asyncio

import pytest

from app.graph.services import GraphServices


class _Settings:
    agent_model = "stub-model"
    agent_hedge_after_ms = 50


class _FakeCompletions:
    """Records every call and hands each one a scripted delay."""

    def __init__(self, delays, results=None, errors=None):
        self._delays = list(delays)
        self._results = list(results) if results else None
        self._errors = list(errors) if errors else None
        self.calls = 0

    async def create(self, **payload):
        index = self.calls
        self.calls += 1
        delay = self._delays[min(index, len(self._delays) - 1)]
        await asyncio.sleep(delay)
        if self._errors and index < len(self._errors) and self._errors[index]:
            raise self._errors[index]
        if self._results:
            return self._results[min(index, len(self._results) - 1)]
        return f"response-{index}"


def _services(completions) -> GraphServices:
    services = GraphServices.__new__(GraphServices)
    services.settings = _Settings()
    services.client = type(
        "C", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    return services


@pytest.mark.asyncio
async def test_a_fast_call_never_hedges():
    """The common case must cost exactly one call.

    A hedge that fires on healthy turns would double the spend on ~90% of
    traffic to rescue the other 10%.
    """
    completions = _FakeCompletions(delays=[0.0])
    result = await _services(completions)._hedged_decision({})

    assert result == "response-0"
    assert completions.calls == 1


@pytest.mark.asyncio
async def test_a_stalled_call_is_overtaken_by_its_hedge():
    """The stall case: the second call answers while the first is still hanging."""
    # First call effectively never returns; the hedge returns promptly.
    completions = _FakeCompletions(delays=[30.0, 0.0])

    result = await asyncio.wait_for(_services(completions)._hedged_decision({}), 5)

    assert result == "response-1", "the hedge should win"
    assert completions.calls == 2


@pytest.mark.asyncio
async def test_the_original_still_wins_if_it_lands_first():
    """A hedge is a race, not a replacement — a first call that recovers is used."""
    completions = _FakeCompletions(delays=[0.12, 30.0])

    result = await asyncio.wait_for(_services(completions)._hedged_decision({}), 5)

    assert result == "response-0"
    assert completions.calls == 2


@pytest.mark.asyncio
async def test_a_failing_racer_does_not_sink_the_turn():
    """One racer raising must not fail the call while the other is still alive.

    This is the case that makes hedging dangerous if unhandled: adding a second
    request also adds a second way to fail.
    """
    completions = _FakeCompletions(
        delays=[30.0, 0.05],
        errors=[None, RuntimeError("hedge blew up")],
        results=["response-0", None],
    )
    # The hedge fails fast; the original is slow but eventually succeeds.
    completions._delays = [0.3, 0.05]

    result = await asyncio.wait_for(_services(completions)._hedged_decision({}), 5)

    assert result == "response-0"


@pytest.mark.asyncio
async def test_both_failing_propagates_the_error():
    """If neither racer succeeds the caller must still see a failure.

    `decide_tools` retries and then falls back to a knowledge-base search, and
    it can only do that if this surfaces the exception rather than hanging.
    """
    completions = _FakeCompletions(
        delays=[0.3, 0.05],
        errors=[RuntimeError("first"), RuntimeError("second")],
    )

    with pytest.raises(RuntimeError):
        await asyncio.wait_for(_services(completions)._hedged_decision({}), 5)


@pytest.mark.asyncio
async def test_hedging_can_be_switched_off():
    """`agent_hedge_after_ms=0` must take the plain single-call path."""
    completions = _FakeCompletions(delays=[0.2])
    services = _services(completions)
    services.settings.agent_hedge_after_ms = 0

    result = await services._hedged_decision({})

    assert result == "response-0"
    assert completions.calls == 1
