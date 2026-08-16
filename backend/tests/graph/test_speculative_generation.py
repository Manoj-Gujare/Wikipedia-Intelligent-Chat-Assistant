"""The answer written under the decision call: flushed, or thrown away.

The failure this guards against is not a slow turn, it is a *wrong* one — a
buffer flushed for a question nobody asked, or for chunks the turn did not end
up using. Every test here is about that, except the two that pin the saving.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.core.agent_tools import ToolCall
from app.core.retriever import RetrievalResult
from app.graph.nodes import agent, generator, tool_executor
from app.graph.nodes.generator import build_generation_messages
from app.graph.nodes.speculation import SpeculativeAnswer, speculate_answer

from .stubs import StubServices, Turn, _hit, _state

# Overlaps the stub chunk's vocabulary, so citation verification keeps it.
# A sentence that shares nothing with its cited chunk is dropped by design, and
# that would mask what these tests are actually asserting.
SPECULATED = "Physics body text [1]."
ORDINARY = "Body text physics [1]."


async def _ready(value):
    return value


def _parked(value):
    return asyncio.create_task(_ready(value))


@pytest.mark.asyncio
async def test_a_confirmed_speculative_answer_is_flushed_without_generating_again():
    """The whole point: one generation, overlapped with the decision call."""
    hits = [_hit()]
    services = StubServices(answer=ORDINARY)

    result = await generator(
        _state(
            hits=hits,
            speculation_hit=True,
            speculative_answer=_parked(SpeculativeAnswer(hits=hits, text=SPECULATED)),
        ),
        services,
    )

    assert result["answer"] == SPECULATED
    assert result["speculative_answer_used"] is True
    # No second call: the answer was already written while the agent decided.
    assert services.generate_calls == 0
    assert services.tokens == [SPECULATED]


@pytest.mark.asyncio
async def test_a_speculative_generation_streams_nothing_while_it_is_speculative():
    """Nothing reaches the client until the decision confirms it."""
    services = StubServices(result=RetrievalResult(hits=[_hit()]), answer=SPECULATED)
    retrieval = asyncio.create_task(services.retrieve("who is albert einstein"))

    parked = await speculate_answer(
        retrieval, services, "who is albert einstein", "en", []
    )

    assert parked is not None
    assert parked.text == SPECULATED
    # Buffered, not streamed. This is the assertion that makes the buffer a
    # buffer rather than an early flush.
    assert services.tokens == []


@pytest.mark.asyncio
async def test_the_speculative_prompt_is_byte_identical_to_the_ordinary_one():
    """A flushed answer must be the answer the serial path would have written.

    If the two prompts could drift, "identical behaviour" would be a claim
    rather than a property, and the drift would show up as a quality change
    nobody attributed to speculation.
    """
    hits = [_hit()]
    services = StubServices(result=RetrievalResult(hits=hits), answer=SPECULATED)
    retrieval = asyncio.create_task(services.retrieve("who is albert einstein"))

    await speculate_answer(retrieval, services, "who is albert einstein", "en", [])

    assert services.generated_prompts[0] == build_generation_messages(
        "who is albert einstein", "en", hits, []
    )


@pytest.mark.asyncio
async def test_a_speculation_miss_discards_the_buffer_and_generates_afresh():
    """Different chunks mean the buffered answer is about a different search."""
    hits = [_hit(title="Chemistry")]
    stale = [_hit(title="Physics")]
    services = StubServices(answer=ORDINARY)

    result = await generator(
        _state(
            hits=hits,
            speculation_hit=False,
            speculative_answer=_parked(
                SpeculativeAnswer(hits=stale, text="STALE ANSWER [1].")
            ),
        ),
        services,
    )

    assert "STALE" not in result["answer"]
    assert result["speculative_answer_used"] is False
    assert services.generate_calls == 1


@pytest.mark.asyncio
async def test_the_discard_path_never_waits_for_the_generation_it_is_discarding():
    """A turn that has decided against the buffer must not block on it.

    Deliberately uses the *same* chunk list, so the identity check would let
    this through: what is being pinned here is `speculation_hit` on its own. If
    the buffer were awaited before that flag was consulted, a turn routed away
    from the knowledge base would pay for the speculative generation in full
    before starting the work it actually needs — turning the optimisation into
    a pessimisation on exactly the slice it cannot help.
    """
    hits = [_hit()]
    services = StubServices(answer=ORDINARY)
    started = asyncio.Event()
    slow_seconds = 3.0

    async def slow():
        started.set()
        await asyncio.sleep(slow_seconds)
        return SpeculativeAnswer(hits=hits, text="STALE [1].")

    parked = asyncio.create_task(slow())
    await started.wait()

    # Elapsed time, not a timeout: `wait_for` would cancel the generator, and
    # its cancellation — not the code under test — is what would unblock it.
    # That masked this exact guard when it was written that way.
    began = time.perf_counter()
    result = await generator(
        _state(hits=hits, speculation_hit=False, speculative_answer=parked),
        services,
    )
    elapsed = time.perf_counter() - began

    assert result["answer"] == ORDINARY
    assert result["speculative_answer_used"] is False
    assert elapsed < slow_seconds / 2, (
        f"the discard path waited {elapsed:.2f}s for a buffer it discarded"
    )
    await asyncio.sleep(0)
    assert parked.cancelled()


@pytest.mark.asyncio
async def test_a_buffer_written_from_other_chunks_is_never_flushed():
    """Belt and braces: even with the hit flag set, the chunks must match.

    `speculation_hit` and the chunk list are set by different nodes. If they
    ever disagree, the answer the user sees must follow the chunks, because the
    chunks are what the citations point at.
    """
    hits = [_hit(title="Chemistry")]
    services = StubServices(answer=ORDINARY)

    result = await generator(
        _state(
            hits=hits,
            speculation_hit=True,
            speculative_answer=_parked(
                SpeculativeAnswer(hits=[_hit(title="Physics")], text="STALE [1].")
            ),
        ),
        services,
    )

    assert "STALE" not in result["answer"]
    assert result["speculative_answer_used"] is False
    assert services.generate_calls == 1


@pytest.mark.asyncio
async def test_a_failed_speculation_leaves_the_turn_to_the_ordinary_path():
    hits = [_hit()]
    services = StubServices(answer=ORDINARY)

    async def boom():
        raise RuntimeError("generation unavailable")

    result = await generator(
        _state(hits=hits, speculation_hit=True, speculative_answer=asyncio.create_task(boom())),
        services,
    )

    assert result["answer"] == ORDINARY
    assert result["speculative_answer_used"] is False
    assert services.generate_calls == 1


@pytest.mark.asyncio
async def test_an_empty_retrieval_discards_the_buffer_rather_than_flushing_it():
    """No hits routes to the refusal path, where a buffered answer is void."""
    services = StubServices(answer=ORDINARY)
    parked = _parked(SpeculativeAnswer(hits=[_hit()], text="STALE [1]."))

    result = await generator(
        _state(hits=[], speculation_hit=True, speculative_answer=parked), services
    )

    assert "STALE" not in result["answer"]
    assert result["speculative_answer_used"] is False


@pytest.mark.asyncio
async def test_a_referential_message_never_speculates_on_the_answer():
    """Its prompt needs the agent's rewrite, which does not exist yet.

    The retrieval speculation still fires — it stitches the previous question
    on — but the *answer* cannot be written until the decision resolves the
    pronoun, because `_question_for_generator` phrases it around the resolved
    query.
    """
    services = StubServices(
        history=[Turn("user", "Tell me about Einstein")],
        result=RetrievalResult(hits=[_hit()]),
    )

    result = await agent(_state(question="what about his wife?"), services)

    assert result.get("speculative") is not None
    assert result.get("speculative_answer") is None


@pytest.mark.asyncio
async def test_a_standalone_question_speculates_on_the_answer():
    services = StubServices(result=RetrievalResult(hits=[_hit()]))

    result = await agent(_state(question="who is albert einstein"), services)

    assert result.get("speculative_answer") is not None
    await asyncio.sleep(0)  # let the task settle before the loop closes


@pytest.mark.asyncio
async def test_the_setting_turns_speculative_generation_off():
    services = StubServices(result=RetrievalResult(hits=[_hit()]))
    services.settings.agent_speculative_generation = False

    result = await agent(_state(question="who is albert einstein"), services)

    assert result.get("speculative") is not None
    assert result.get("speculative_answer") is None


@pytest.mark.asyncio
async def test_the_executor_hands_the_buffer_on_to_the_generator():
    """`tool_executor` must not clear the buffer on the path that can use it."""
    services = StubServices(result=RetrievalResult(hits=[_hit()]))
    parked = _parked(SpeculativeAnswer(hits=[_hit()], text=SPECULATED))
    retrieval = asyncio.create_task(services.retrieve("who is albert einstein"))

    result = await tool_executor(
        _state(
            question="who is albert einstein",
            pending_tool="search_knowledge_base",
            tool_query="who is albert einstein",
            speculative=retrieval,
            speculative_query="who is albert einstein",
            speculative_answer=parked,
        ),
        services,
    )

    assert "speculative_answer" not in result
    assert result["speculation_hit"] is True


@pytest.mark.asyncio
async def test_a_direct_reply_discards_the_buffer():
    """The 22% case: the turn never wanted an answer from the knowledge base."""
    services = StubServices(
        result=RetrievalResult(hits=[_hit()]),
        tool_call=ToolCall("respond_directly", "", reply="Hello!"),
    )
    from app.graph.nodes import direct_responder

    parked = _parked(SpeculativeAnswer(hits=[_hit()], text="STALE [1]."))
    result = await direct_responder(
        _state(direct_reply="Hello!", speculative_answer=parked), services
    )

    assert result["answer"] == "Hello!"
    assert result["speculative_answer"] is None
    assert "STALE" not in "".join(services.tokens)
