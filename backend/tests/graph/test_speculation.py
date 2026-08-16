"""Reusing, discarding and recovering the search parked under the decision call."""

from __future__ import annotations

import asyncio

import pytest

from app.graph.nodes import tool_executor
from app.core.retriever import RetrievalResult
from .stubs import StubServices, _hit, _state


@pytest.mark.asyncio
async def test_executor_reuses_the_speculative_search_when_the_query_matches():
    """The whole latency argument rests on this: a matching query costs 0ms."""
    services = StubServices(result=RetrievalResult(hits=[_hit()]))
    parked = asyncio.create_task(services.retrieve("what is photosynthesis"))

    result = await tool_executor(
        _state(
            question="what is photosynthesis",
            pending_tool="search_knowledge_base",
            tool_query="what is photosynthesis",
            speculative=parked,
            speculative_query="what is photosynthesis",
        ),
        services,
    )

    assert result["speculation_hit"] is True
    # One search total — the parked one. No second trip to the index.
    assert services.retrieve_calls == ["what is photosynthesis"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw,chosen",
    [
        # The agent's usual edit is to strip filler, not to change the subject.
        # Measured against the real API; a symmetric overlap measure scored the
        # first pair 0.67 and rejected every one of these.
        ("What is the event horizon of a black hole?", "event horizon of a black hole"),
        ("What are the rings of Saturn made of?", "rings of Saturn composition"),
        ("Who was Marie Curie and what did she discover?", "Marie Curie discoveries"),
    ],
)
async def test_a_stripped_query_still_reuses_the_speculation(raw, chosen):
    """Filler the agent removed must not count against the match."""
    services = StubServices(result=RetrievalResult(hits=[_hit()]))
    parked = asyncio.create_task(services.retrieve(raw))

    result = await tool_executor(
        _state(
            question=raw,
            pending_tool="search_knowledge_base",
            tool_query=chosen,
            speculative=parked,
            speculative_query=raw,
        ),
        services,
    )

    assert result["speculation_hit"] is True
    assert services.retrieve_calls == [raw]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw,chosen",
    [
        # A resolved pronoun means a new subject. Serving chunks retrieved for
        # the pronoun would be a *worse* failure than the latency saved,
        # because it looks like a successful retrieval.
        ("what about his wife?", "Albert Einstein wife"),
        ("and the population?", "India population"),
        ("what happened next?", "Roman Empire fall"),
        # A bare term the agent disambiguated: the parked search covered the
        # ambiguous word, not the sense the agent settled on.
        ("What is Mercury?", "Mercury planet"),
    ],
)
async def test_a_resolved_rewrite_never_reuses_the_speculation(raw, chosen):
    """The costly direction — the threshold must not drift far enough to admit these."""
    services = StubServices(result=RetrievalResult(hits=[_hit()]))
    parked = asyncio.create_task(services.retrieve(raw))

    result = await tool_executor(
        _state(
            question=raw,
            pending_tool="search_knowledge_base",
            tool_query=chosen,
            speculative=parked,
            speculative_query=raw,
        ),
        services,
    )

    assert result["speculation_hit"] is False
    assert services.retrieve_calls[-1] == chosen


@pytest.mark.asyncio
async def test_an_accepted_false_positive_is_still_discarded_downstream():
    """The backstop that makes a permissive `looks_self_contained` affordable.

    Dropping the capitalised-proper-noun proxy lets genuinely elliptical
    messages speculate ("What is the capital?" — of what?). That is only
    acceptable because reuse is a separate decision: once the agent resolves
    the subject the parked chunks are for a different question, `_similar`
    scores 0.50 against a 0.60 threshold, and they are thrown away. The message
    costs one wasted embedding and never a wrong answer.
    """
    services = StubServices(result=RetrievalResult(hits=[_hit()]))
    parked = asyncio.create_task(services.retrieve("What is the capital?"))

    result = await tool_executor(
        _state(
            question="What is the capital?",
            pending_tool="search_knowledge_base",
            tool_query="France capital",
            speculative=parked,
            speculative_query="What is the capital?",
        ),
        services,
    )

    assert result["speculation_hit"] is False
    assert services.retrieve_calls[-1] == "France capital"


@pytest.mark.asyncio
async def test_a_narrower_question_about_the_same_subject_does_not_reuse():
    """The case that looks like a false miss and is not.

    "about black hole" -> "who found it" parks the stitched message and the
    agent asks for "who discovered black holes". Every content word of the
    agent's query is already covered — only `hole`/`holes` separates them — so
    this scores 0.50 and reads like the tokeniser being too literal. Making
    membership plural-tolerant does lift it to 0.75 and does save the ~0.5s
    second search.

    It also produces a refusal, on every run measured: the stitched query
    retrieves general black-hole chunks, and the discovery history the question
    actually asks for only surfaces for the agent's own wording. Word overlap
    cannot see that, so the miss must stay a miss.
    """
    services = StubServices(result=RetrievalResult(hits=[_hit()]))
    parked = asyncio.create_task(services.retrieve("about black hole who found it"))

    result = await tool_executor(
        _state(
            question="who found it",
            pending_tool="search_knowledge_base",
            tool_query="who discovered black holes",
            speculative=parked,
            speculative_query="about black hole who found it",
        ),
        services,
    )

    assert result["speculation_hit"] is False
    assert services.retrieve_calls[-1] == "who discovered black holes"


@pytest.mark.asyncio
async def test_discarding_a_speculation_leaves_no_unretrieved_exception():
    """`.exception()` on a cancelled task raises CancelledError, a BaseException.

    Suppressing only `Exception` printed a stack trace under every speculation
    miss — a turn that succeeded, logged as though it had crashed.
    """
    services = StubServices(result=RetrievalResult(hits=[_hit()]))
    seen: list = []
    asyncio.get_running_loop().set_exception_handler(
        lambda loop, context: seen.append(context)
    )

    async def slow():
        await asyncio.sleep(5)

    parked = asyncio.create_task(slow())
    await tool_executor(
        _state(
            question="what about his wife?",
            pending_tool="search_knowledge_base",
            tool_query="Albert Einstein wife",
            speculative=parked,
            speculative_query="what about his wife?",
        ),
        services,
    )
    await asyncio.sleep(0)  # let the cancellation and its callback settle

    assert seen == []


@pytest.mark.asyncio
async def test_executor_discards_a_speculation_the_agent_rewrote_away_from():
    """Serving pronoun-retrieved chunks for a resolved query is a silent wrong answer."""
    services = StubServices(result=RetrievalResult(hits=[_hit()]))
    parked = asyncio.create_task(services.retrieve("what about his wife?"))

    result = await tool_executor(
        _state(
            question="what about his wife?",
            pending_tool="search_knowledge_base",
            tool_query="Albert Einstein wife",
            speculative=parked,
            speculative_query="what about his wife?",
        ),
        services,
    )

    assert result["speculation_hit"] is False
    # The agent's resolved query ran; the speculation was thrown away.
    assert services.retrieve_calls[-1] == "Albert Einstein wife"


@pytest.mark.asyncio
async def test_a_failed_speculation_is_retried_rather_than_failing_the_turn():
    services = StubServices(result=RetrievalResult(hits=[_hit()]))

    async def boom():
        raise RuntimeError("index unavailable")

    parked = asyncio.create_task(boom())

    result = await tool_executor(
        _state(
            question="what is physics",
            pending_tool="search_knowledge_base",
            tool_query="what is physics",
            speculative=parked,
            speculative_query="what is physics",
        ),
        services,
    )

    assert result["speculation_hit"] is False
    assert services.retrieve_calls == ["what is physics"]
    assert result["hits"]
