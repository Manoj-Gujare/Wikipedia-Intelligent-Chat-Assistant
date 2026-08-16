"""Node 3: tool choice, subject resolution, and launching the speculative search."""

from __future__ import annotations

import asyncio

import pytest

from app.core.agent_tools import ToolCall
from app.graph.nodes import agent, route_after_agent, tool_executor
from app.core.retriever import RetrievalResult
from .stubs import StubServices, Turn, chitchat_turn, _hit, _state


@pytest.mark.parametrize(
    "tool,expected",
    [
        ("respond_directly", "direct_responder"),
        ("answer_from_history", "history_answerer"),
        ("search_knowledge_base", "tool_executor"),
        ("search_wikipedia", "tool_executor"),
    ],
)
def test_each_tool_reaches_its_own_branch(tool, expected):
    assert route_after_agent(_state(pending_tool=tool)) == expected


@pytest.mark.asyncio
async def test_an_unresolved_referential_message_is_stitched_to_the_current_subject():
    """The bug from the UI: "And his wife" after Einstein *then* Dhoni.

    `gpt-4.1-nano` handed the message back unchanged, and searching "And his
    wife" literally answered about whoever the words sat near — Einstein, from
    two turns earlier. The stitch keeps the *current* subject in the query.
    """
    services = StubServices(
        history=[
            Turn("user", "who is albert einstein"),
            Turn("assistant", "A physicist."),
            Turn("user", "And what about MS Dhoni"),
            Turn("assistant", "A cricketer."),
        ],
        # The agent resolved nothing.
        tool_call=ToolCall("search_knowledge_base", "And his wife"),
    )

    result = await agent(_state(question="And his wife"), services)

    assert result["tool_query"] == "And what about MS Dhoni And his wife"
    # And it matches what speculation already searched, so it costs nothing.
    assert result["speculative_query"] == "And what about MS Dhoni And his wife"


@pytest.mark.asyncio
async def test_a_topic_switch_that_opens_referentially_is_not_stitched():
    """The mirror image, and a real regression from the running UI.

    "what about black hole" opens like a follow-up but names its own subject,
    so the agent adds nothing when it answers "black hole" — not because it
    failed to resolve a reference, but because there was no reference to
    resolve. The old backstop read those two cases as one and grafted the
    previous question on, so after a Tesla refusal this searched for *"What is
    the current stock price of Tesla? what about black hole"* and answered both
    halves in one paragraph.
    """
    services = StubServices(
        history=[
            Turn("user", "What is the current stock price of Tesla?"),
            Turn("assistant", "The indexed articles don't cover that."),
        ],
        tool_call=ToolCall("search_knowledge_base", "black hole"),
    )

    result = await agent(_state(question="what about black hole"), services)

    assert result["tool_query"] == "black hole"


@pytest.mark.asyncio
async def test_a_pronoun_the_agent_left_unresolved_is_still_stitched():
    """The distinction has to survive: a pronoun really does need the history.

    Same shape as the case above — the agent supplies nothing beyond the
    message — but here that is a failure to resolve, because `his` points
    outside the message and cannot be searched as written.
    """
    services = StubServices(
        history=[
            Turn("user", "Who was Albert Einstein?"),
            Turn("assistant", "A physicist."),
        ],
        tool_call=ToolCall("search_knowledge_base", "his wife"),
    )

    result = await agent(_state(question="what about his wife?"), services)

    assert result["tool_query"] == "Who was Albert Einstein? what about his wife?"


@pytest.mark.asyncio
async def test_a_rewrite_to_a_stale_subject_is_overridden():
    """The dangerous case: confidently the wrong person.

    "Albert Einstein wife" retrieves real chunks and cites them, so the answer
    looks grounded while being about someone the user stopped asking about two
    turns ago. Observed from `gpt-4.1-nano` on this exact conversation.
    """
    services = StubServices(
        history=[
            Turn("user", "I am looking for albert einstein"),
            Turn("assistant", "A physicist."),
            Turn("user", "and ms dhoni"),
            Turn("assistant", "Not in your knowledge base."),
        ],
        tool_call=ToolCall("search_knowledge_base", "Albert Einstein wife"),
    )

    result = await agent(_state(question="and his wife"), services)

    assert result["tool_query"] == "and ms dhoni and his wife"


@pytest.mark.asyncio
async def test_a_rewrite_the_recent_questions_corroborate_is_left_alone():
    # "MS Dhoni" appears in the question before it, so the agent got it right
    # and its phrasing is better than any stitch.
    services = StubServices(
        history=[Turn("user", "And what about MS Dhoni")],
        tool_call=ToolCall("search_knowledge_base", "MS Dhoni wife"),
    )

    result = await agent(_state(question="And his wife"), services)

    assert result["tool_query"] == "MS Dhoni wife"


@pytest.mark.asyncio
async def test_a_subject_two_questions_back_still_corroborates():
    """A chain of referential turns must not lose the subject it started from."""
    services = StubServices(
        history=[
            Turn("user", "Tell me about Albert Einstein"),
            Turn("assistant", "A physicist."),
            Turn("user", "his wife"),
            Turn("assistant", "Elsa Löwenthal."),
        ],
        tool_call=ToolCall("search_knowledge_base", "Albert Einstein wife death"),
    )

    result = await agent(_state(question="when did she die?"), services)

    assert result["tool_query"] == "Albert Einstein wife death"


@pytest.mark.asyncio
async def test_a_self_contained_question_is_never_second_guessed():
    # No pronoun, so the agent is free to rewrite however it likes — a topic
    # switch mid-conversation must not be dragged back to the old subject.
    services = StubServices(
        history=[Turn("user", "Tell me about the Taj Mahal")],
        tool_call=ToolCall("search_knowledge_base", "World War II end date"),
    )

    result = await agent(_state(question="When did World War II end?"), services)

    assert result["tool_query"] == "World War II end date"


@pytest.mark.asyncio
async def test_speculation_never_stitches_onto_a_greeting():
    services = StubServices(history=[chitchat_turn("Hey"), Turn("assistant", "Hi!")])

    result = await agent(_state(question="what about his wife?"), services)

    # Nothing substantive to stitch to, so no speculation at all — searching
    # "Hey what about his wife?" is worse than not speculating.
    assert result.get("speculative") is None


@pytest.mark.asyncio
async def test_agent_records_its_tool_choice_and_query():
    services = StubServices(
        history=[Turn("user", "Tell me about India")],
        tool_call=ToolCall("search_knowledge_base", "India population"),
    )

    result = await agent(_state(question="what's its population?"), services)

    assert result["pending_tool"] == "search_knowledge_base"
    assert result["tool_query"] == "India population"
    # The rewritten query is what retrieval and the generator both use.
    assert result["standalone_query"] == "India population"
    assert result["tool_calls"] == [
        {"tool": "search_knowledge_base", "query": "India population", "hop": 0}
    ]


@pytest.mark.asyncio
async def test_agent_speculates_on_a_first_turn_question():
    services = StubServices(history=[], result=RetrievalResult(hits=[_hit()]))

    result = await agent(_state(question="What did Marie Curie discover?"), services)

    assert result["speculative"] is not None
    assert result["speculative_query"] == "What did Marie Curie discover?"
    await result["speculative"]
    assert services.retrieve_calls == ["What did Marie Curie discover?"]


@pytest.mark.asyncio
async def test_a_referential_followup_speculates_on_the_stitched_question():
    """"what about his wife?" is noise alone, and a fine search stitched to its subject."""
    services = StubServices(history=[Turn("user", "Who is MS Dhoni")])

    result = await agent(_state(question="what about his wife?"), services)

    assert result["speculative_query"] == "Who is MS Dhoni what about his wife?"
    await result["speculative"]
    assert services.retrieve_calls == ["Who is MS Dhoni what about his wife?"]


@pytest.mark.asyncio
async def test_a_stitched_speculation_is_reused_for_the_agents_resolved_query():
    """The point of stitching: the agent's query is contained in it, so it hits."""
    services = StubServices(result=RetrievalResult(hits=[_hit()]))
    stitched = "What is a black hole? What happens at its event horizon?"
    parked = asyncio.create_task(services.retrieve(stitched))

    result = await tool_executor(
        _state(
            question="What happens at its event horizon?",
            pending_tool="search_knowledge_base",
            tool_query="black hole event horizon",
            speculative=parked,
            speculative_query=stitched,
        ),
        services,
    )

    assert result["speculation_hit"] is True
    assert services.retrieve_calls == [stitched]


@pytest.mark.asyncio
async def test_no_speculation_when_there_is_nothing_to_stitch_to():
    # History exists but holds no user turn, so there is no subject to borrow.
    services = StubServices(history=[Turn("assistant", "Hello!")])

    result = await agent(_state(question="tell me more"), services)

    assert result.get("speculative") is None
    assert services.retrieve_calls == []


@pytest.mark.asyncio
async def test_the_agent_does_not_re_speculate_after_a_search_already_missed():
    """A search has run for this turn; launching another for it buys nothing."""
    services = StubServices(history=[])

    result = await agent(
        _state(
            question="what is a quantum widget",
            hops=0,
            tool_calls=[{"tool": "search_knowledge_base", "query": "x", "hop": 0}],
        ),
        services,
    )

    assert result.get("speculative") is None
    assert services.retrieve_calls == []


@pytest.mark.asyncio
async def test_speculation_can_be_switched_off():
    services = StubServices(history=[])
    services.settings.agent_speculative_retrieval = False

    result = await agent(_state(question="What did Marie Curie discover?"), services)

    assert result.get("speculative") is None
    assert services.retrieve_calls == []


@pytest.mark.asyncio
async def test_agent_does_not_speculate_twice_on_a_second_hop():
    services = StubServices(history=[])
    first = await agent(_state(question="What did Marie Curie discover?"), services)

    second = await agent(
        _state(
            question="What did Marie Curie discover?",
            hops=1,
            speculative=first["speculative"],
        ),
        services,
    )

    assert "speculative" not in second
    await first["speculative"]
    assert len(services.retrieve_calls) == 1
