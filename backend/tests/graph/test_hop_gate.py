"""`route_after_tools`: whether an empty result earns another hop."""

from __future__ import annotations

import time


from app.graph.nodes import route_after_tools
from .stubs import _hit, _state


def test_a_hit_never_hops_again():
    assert route_after_tools(_state(hits=[_hit()], hops=1)) == "generator"


def test_an_empty_result_inside_the_budget_hops_again():
    state = _state(hits=[], hops=1, started_at=time.perf_counter())

    assert route_after_tools(state) == "agent"


def test_an_empty_result_past_the_deadline_answers_instead_of_hopping():
    """The gate is wall-clock: past 1.2s a second search cannot fit under 3s."""
    state = _state(hits=[], hops=1, started_at=time.perf_counter() - 2.0)

    assert route_after_tools(state) == "generator"


def test_max_hops_terminates_the_loop_regardless_of_the_clock():
    state = _state(hits=[], hops=2, started_at=time.perf_counter())

    assert route_after_tools(state) == "generator"


def test_ambiguity_wins_over_hopping():
    state = _state(intent="ambiguous", hits=[], hops=1)

    assert route_after_tools(state) == "clarification"
