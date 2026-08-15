"""The tool menu offered to the agent, and parsing its choice back."""

from __future__ import annotations

import asyncio

import pytest

from app.core.agent_tools import ToolCall, parse_tool_calls
from .stubs import _FakeCompletions, _Response, _FakeClient, _services_with, _Fn, _Call, _Msg, StubSettings, Turn, chitchat_turn










@pytest.mark.asyncio
async def test_answer_from_history_is_not_offered_on_an_opening_message():
    """A tool that cannot apply is not on the menu.

    The model picked `answer_from_history` for four plain opening questions the
    index answers with six hits each — 4/20 flat losses. Prompt wording did not
    stop it; removing the option did.
    """
    sink: list = []

    await _services_with(sink).decide_tools("What is machine learning?", history=[])

    offered = {t["function"]["name"] for t in sink[0]["tools"]}
    assert offered == {"search_knowledge_base", "search_wikipedia", "respond_directly"}


@pytest.mark.asyncio
async def test_answer_from_history_returns_once_there_is_a_conversation():
    sink: list = []
    history = [Turn("user", "Who is Einstein"), Turn("assistant", "A physicist.")]

    await _services_with(sink).decide_tools("what did you just say?", history=history)

    offered = {t["function"]["name"] for t in sink[0]["tools"]}
    assert "answer_from_history" in offered


@pytest.mark.asyncio
async def test_a_refusal_that_cannot_be_generated_falls_back_to_the_template():
    """A turn must not die because its apology could not be written.

    The English template that used to serve every language is kept for exactly
    this: a refusal in the wrong language is a poor answer, and no answer at all
    is worse.
    """
    from app.graph.services import GraphServices

    class _Failing:
        async def create(self, **_payload):
            raise RuntimeError("model unreachable")

    services = GraphServices.__new__(GraphServices)
    services.settings = StubSettings()
    services.settings.chat_model = "gpt-4.1-nano"
    services.client = type("C", (), {"chat": type("Ch", (), {"completions": _Failing()})()})()

    answer = await services.compose_refusal("mitochondria", ["Mitochondria"], "mr")

    assert "Mitochondria" in answer
    assert answer.strip()


@pytest.mark.asyncio
async def test_a_greeting_exchange_is_kept_out_of_the_decision_transcript():
    """Measured: a greeting in the transcript flips the next turn's routing.

    "Who are you?" routes to `respond_directly` on an empty transcript and after
    a real exchange, but after `Hey` / `Hello! How can I assist you today?` it
    was routed to `search_knowledge_base` with the query "What is ChatGPT" —
    the model trying to look up the answer to a question about itself.

    Small talk was already excluded from the *subject* the agent resolves
    against; leaving it in the transcript was the half of that idea that never
    got applied.
    """
    sink: list = []
    history = [chitchat_turn("Hey"), Turn("assistant", "Hello! How can I assist you today?")]

    await _services_with(sink).decide_tools("Who are you?", history=history)

    sent = sink[0]["messages"][-1]["content"]
    assert "Hey" not in sent
    assert "How can I assist you today" not in sent


@pytest.mark.asyncio
async def test_a_real_exchange_survives_the_same_filter():
    """The filter must remove greetings, not history."""
    sink: list = []
    history = [
        Turn("user", "who is albert einstein"),
        Turn("assistant", "A physicist."),
        chitchat_turn("thanks!"),
        Turn("assistant", "You're welcome."),
    ]

    await _services_with(sink).decide_tools("what about his wife?", history=history)

    sent = sink[0]["messages"][-1]["content"]
    assert "who is albert einstein" in sent
    assert "A physicist." in sent
    # The interleaved pleasantry and its reply both go.
    assert "thanks!" not in sent
    assert "You're welcome." not in sent


@pytest.mark.asyncio
async def test_a_decision_with_no_tool_call_still_searches():
    sink: list = []

    calls = await _services_with(sink).decide_tools("what is physics", history=[])

    assert calls == [ToolCall("search_knowledge_base", "what is physics")]








def test_a_well_formed_tool_call_is_parsed():
    message = _Msg([_Call("search_knowledge_base", '{"query": "Albert Einstein wife"}')])

    calls = parse_tool_calls(message, "fallback")

    assert calls == [ToolCall("search_knowledge_base", "Albert Einstein wife")]


def test_no_tool_call_fails_open_to_a_knowledge_search():
    # Dropping a real question because the model emitted nothing loses the
    # answer; searching when we needn't costs a few hundred milliseconds.
    calls = parse_tool_calls(_Msg([]), "what is photosynthesis")

    assert calls == [ToolCall("search_knowledge_base", "what is photosynthesis")]


def test_malformed_arguments_fall_back_to_the_raw_message():
    message = _Msg([_Call("search_knowledge_base", "{not json")])

    calls = parse_tool_calls(message, "what is photosynthesis")

    assert calls == [ToolCall("search_knowledge_base", "what is photosynthesis")]


def test_an_unknown_tool_name_is_ignored_and_fails_open():
    message = _Msg([_Call("delete_everything", "{}")])

    calls = parse_tool_calls(message, "what is photosynthesis")

    assert calls == [ToolCall("search_knowledge_base", "what is photosynthesis")]


def test_answer_from_history_needs_no_query():
    calls = parse_tool_calls(_Msg([_Call("answer_from_history", "{}")]), "fallback")

    assert calls == [ToolCall("answer_from_history", "")]


def test_only_one_tool_runs_per_hop():
    # Two retrievals racing into one context block, with no budget left to
    # reconcile them, is not a plan the deadline can absorb.
    message = _Msg(
        [
            _Call("search_knowledge_base", '{"query": "a"}'),
            _Call("search_wikipedia", '{"query": "b"}'),
        ]
    )

    assert len(parse_tool_calls(message, "fallback")) == 1
