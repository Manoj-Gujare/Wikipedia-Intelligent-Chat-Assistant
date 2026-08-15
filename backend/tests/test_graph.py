"""Graph tests: each node in isolation, then routing through the compiled graph.

Every dependency is stubbed, so these run with no network and no API key. The
point is to pin the *routing* behaviour — that a greeting never touches the
agent, that speculative retrieval is reused only when it is safe to reuse, and
that the hop loop terminates on the clock. Those three are what make an agentic
pipeline fit inside a 3s budget, and all three are easy to regress silently.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.core.rag import ToolCall, parse_tool_calls
from app.core.vector_store import SearchHit
from app.graph.build import build_graph
from app.graph.nodes import (
    agent,
    clarification,
    direct_responder,
    gate,
    generator,
    history_answerer,
    looks_self_contained,
    route_after_agent,
    route_after_tools,
    tool_executor,
)
from app.core.retriever import DisambiguationOption, RetrievalResult
from app.models import Source


class Turn:
    def __init__(self, role: str, content: str, meta: dict | None = None) -> None:
        self.role, self.content, self.meta = role, content, meta


def chitchat_turn(content: str) -> Turn:
    """A user turn the agent classified as needing no evidence.

    Small talk is marked when it is written now, rather than re-derived on
    every history read, so a test that wants "the user said hello" has to say
    which verdict was recorded — exactly as production does.
    """
    return Turn("user", content, meta={"chitchat": True})


class StubSettings:
    agent_hop_deadline_ms = 1200
    agent_max_hops = 2
    agent_speculative_retrieval = True
    agent_speculation_reuse_threshold = 0.6


class StubServices:
    """Records what each node asked for, so tests can assert on work avoided."""

    def __init__(
        self,
        history=None,
        result=None,
        answer="Grounded answer [1].",
        live_disambiguation=None,
        tool_call=None,
        live_result=None,
    ):
        self._history = history or []
        self._result = result or RetrievalResult()
        self._answer = answer
        # Empty by default: the score-gap path should fall back to related
        # topics unless a test explicitly supplies live options.
        self._live_disambiguation = live_disambiguation or []
        self._live_result = live_result or RetrievalResult(used_live_search=True)
        self.settings = StubSettings()
        self.retrieve_calls: list[str] = []
        self.fallback_calls: list[str] = []
        self.refusal_calls: list[tuple] = []
        self.live_search_calls: list[str] = []
        self.live_disambiguation_calls: list[str] = []
        self.generate_calls = 0
        self.decide_calls: list[list[dict]] = []
        self.tokens: list[str] = []
        # A list cycles one decision per hop; a bare ToolCall repeats forever.
        self.tool_call = tool_call or ToolCall("search_knowledge_base", "rewritten query")

    def history(self, session_id):
        return self._history

    def emit_token(self, text):
        self.tokens.append(text)

    async def retrieve(self, query, lang="en"):
        self.retrieve_calls.append(query)
        return self._result

    async def search_wikipedia_live(self, query, lang="en"):
        self.live_search_calls.append(query)
        return self._live_result

    async def live_disambiguation(self, term, lang="en"):
        self.live_disambiguation_calls.append(term)
        return list(self._live_disambiguation)

    async def decide_tools(self, message, history, observations=None):
        self.decide_calls.append(list(observations or []))
        if isinstance(self.tool_call, list):
            index = min(len(self.decide_calls) - 1, len(self.tool_call) - 1)
            return [self.tool_call[index]]
        return [self.tool_call]

    async def generate_stream(self, messages):
        self.generate_calls += 1
        return self._answer

    async def fallback_articles(self, query, lang):
        self.fallback_calls.append(query)
        return [{"title": "Live result", "url": "https://en.wikipedia.org/wiki/X",
                 "lang": lang, "summary": ""}]

    async def compose_refusal(self, subject, titles, lang="en"):
        # Counted apart from `generate_calls`: wording a refusal is not the same
        # event as generating an answer, and tests assert on both.
        self.refusal_calls.append((subject, list(titles), lang))
        answer = f"[no {subject}] {', '.join(titles)}"
        self.emit_token(answer)
        return answer

    @staticmethod
    def article_links(hits, sources):
        return [{"title": h.title, "url": "https://en.wikipedia.org/wiki/X",
                 "lang": "en", "summary": ""} for h in hits[:2]]


def _hit(title="Physics", score=0.8, section="Introduction") -> SearchHit:
    return SearchHit(
        id=f"{title}-{section}",
        text=f"{title} > {section}\n\nBody text.",
        score=score,
        metadata={
            "title": title,
            "section_title": section,
            "section_path": f"{title} > {section}",
            "section_url": f"https://en.wikipedia.org/wiki/{title}#{section}",
            "url": f"https://en.wikipedia.org/wiki/{title}",
            "lang": "en",
            "body": "Body text.",
        },
    )


def _state(**overrides):
    base = {
        "question": "who is albert einstein",
        "session_id": "s1",
        "lang": "en",
        "intent": "new_question",
        "messages": [],
        "node_timings": [],
        "hops": 0,
        "started_at": time.perf_counter(),
        "hop_deadline_ms": 1200,
        "max_hops": 2,
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------- node 1


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
async def test_gate_copies_the_hop_budget_into_state():
    # A conditional edge sees only state, so the budget has to travel in it.
    services = StubServices()

    result = await gate(_state(), services)

    assert result["hop_deadline_ms"] == 1200
    assert result["max_hops"] == 2


# ------------------------------------------------------------------- node 2


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


# ------------------------------------------------------------------- node 3


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


# ------------------------------------------------------------------- node 4


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


@pytest.mark.asyncio
async def test_executor_maps_hits_to_serialisable_chunks():
    services = StubServices(result=RetrievalResult(hits=[_hit(), _hit("Optics")]))

    result = await tool_executor(
        _state(pending_tool="search_knowledge_base", tool_query="physics"), services
    )

    chunk = result["retrieved_chunks"][0]
    assert chunk["article_title"] == "Physics"
    assert chunk["url"].endswith("#Introduction")
    assert set(chunk) >= {"text", "article_title", "url", "section", "score"}


@pytest.mark.asyncio
async def test_live_search_tool_grounds_on_fetched_articles():
    services = StubServices(
        live_result=RetrievalResult(hits=[_hit("Recent event")], used_live_search=True)
    )

    result = await tool_executor(
        _state(pending_tool="search_wikipedia", tool_query="2026 election"), services
    )

    assert services.live_search_calls == ["2026 election"]
    assert services.retrieve_calls == []
    assert result["used_live_search"] is True
    assert result["hits"]


@pytest.mark.asyncio
async def test_live_hits_are_never_treated_as_ambiguous():
    """Live hits carry a nominal score, so every pair of them reads as tied."""
    services = StubServices(
        live_result=RetrievalResult(
            hits=[_hit("Venus", 0.5), _hit("Venus (mythology)", 0.5)],
            used_live_search=True,
        )
    )

    result = await tool_executor(
        _state(question="Venus", pending_tool="search_wikipedia", tool_query="Venus"),
        services,
    )

    assert result.get("intent") != "ambiguous"
    assert services.live_disambiguation_calls == []


@pytest.mark.asyncio
async def test_disambiguation_page_sets_ambiguous_intent():
    services = StubServices(
        result=RetrievalResult(
            disambiguation=[
                DisambiguationOption("Mercury (planet)", "https://en.wikipedia.org/wiki/Mercury_(planet)"),
                DisambiguationOption("Mercury (element)", "https://en.wikipedia.org/wiki/Mercury_(element)"),
            ]
        )
    )

    result = await tool_executor(
        _state(question="Mercury", pending_tool="search_knowledge_base", tool_query="Mercury"),
        services,
    )

    assert result["intent"] == "ambiguous"
    assert len(result["disambiguation_candidates"]) == 2


@pytest.mark.asyncio
async def test_bare_entity_with_tied_scores_is_ambiguous():
    services = StubServices(
        result=RetrievalResult(hits=[_hit("Java", 0.700), _hit("Java coffee", 0.695)])
    )

    result = await tool_executor(
        _state(question="Java", pending_tool="search_knowledge_base", tool_query="Java"),
        services,
    )

    assert result["intent"] == "ambiguous"


@pytest.mark.asyncio
async def test_score_gap_prefers_real_disambiguation_options_from_wikipedia():
    # The index has no disambiguation page for the term, so the options must
    # come from the live API rather than from whatever ranked nearby.
    services = StubServices(
        result=RetrievalResult(hits=[_hit("Venus", 0.700), _hit("Synodic day", 0.695)]),
        live_disambiguation=[
            {"title": "Venus", "url": "u1"},
            {"title": "Venus (mythology)", "url": "u2"},
        ],
    )

    result = await tool_executor(
        _state(question="Venus", pending_tool="search_knowledge_base", tool_query="Venus"),
        services,
    )

    assert services.live_disambiguation_calls == ["Venus"]
    assert result["disambiguation_kind"] == "page"
    assert [c["title"] for c in result["disambiguation_candidates"]] == [
        "Venus",
        "Venus (mythology)",
    ]


@pytest.mark.asyncio
async def test_score_gap_falls_back_to_related_topics_when_wikipedia_has_none():
    services = StubServices(
        result=RetrievalResult(hits=[_hit("Venus", 0.700), _hit("Synodic day", 0.695)]),
        live_disambiguation=[],
    )

    result = await tool_executor(
        _state(question="Venus", pending_tool="search_knowledge_base", tool_query="Venus"),
        services,
    )

    assert result["disambiguation_kind"] == "related"
    assert [c["title"] for c in result["disambiguation_candidates"]] == [
        "Venus",
        "Synodic day",
    ]


@pytest.mark.asyncio
async def test_indexed_disambiguation_page_skips_the_live_lookup():
    services = StubServices(
        result=RetrievalResult(
            disambiguation=[DisambiguationOption(title="Mercury (planet)", url="u1")]
        )
    )

    result = await tool_executor(
        _state(question="Mercury", pending_tool="search_knowledge_base", tool_query="Mercury"),
        services,
    )

    assert services.live_disambiguation_calls == []
    assert result["disambiguation_kind"] == "page"


@pytest.mark.asyncio
async def test_weak_tied_scores_are_a_miss_not_an_ambiguity():
    # Nothing relevant was found, so every hit scores about the same. Offering
    # them as "did you mean?" produced "MS Dhoni could refer to India, Mahatma
    # Gandhi, Hindi cinema" — a refusal plus live search is the honest answer.
    services = StubServices(
        result=RetrievalResult(hits=[_hit("India", 0.31), _hit("Hindi cinema", 0.305)])
    )

    result = await tool_executor(
        _state(question="MS Dhoni", pending_tool="search_knowledge_base", tool_query="MS Dhoni"),
        services,
    )

    assert result.get("intent") != "ambiguous"


@pytest.mark.asyncio
async def test_an_agent_compressed_question_is_not_mistaken_for_a_bare_entity():
    """The word-count gate describes what a *user* typed, not what the agent built.

    "What does photosynthesis produce?" came back as "photosynthesis products" —
    two words, tied scores — and was served a disambiguation prompt for a
    question the index answers outright.
    """
    services = StubServices(
        result=RetrievalResult(
            hits=[_hit("Photosynthesis", 0.700), _hit("Chloroplast", 0.695)]
        )
    )

    result = await tool_executor(
        _state(
            question="What does photosynthesis produce?",
            pending_tool="search_knowledge_base",
            tool_query="photosynthesis products",
        ),
        services,
    )

    assert result.get("intent") != "ambiguous"
    assert result["hits"]


@pytest.mark.asyncio
async def test_a_bare_entity_the_user_typed_is_still_ambiguous_after_rewriting():
    # The mirror case: the agent may expand "Java" into something longer, but
    # the user still typed one bare word and still deserves the choice.
    services = StubServices(
        result=RetrievalResult(hits=[_hit("Java", 0.700), _hit("Java coffee", 0.695)]),
        live_disambiguation=[{"title": "Java (programming language)", "url": "u1"}],
    )

    result = await tool_executor(
        _state(
            question="Java",
            pending_tool="search_knowledge_base",
            tool_query="Java programming language island",
        ),
        services,
    )

    assert result["intent"] == "ambiguous"


@pytest.mark.asyncio
async def test_a_real_question_spanning_articles_is_not_ambiguous():
    # Healthy retrieval routinely spans articles; only bare mentions qualify.
    services = StubServices(
        result=RetrievalResult(hits=[_hit("Physics", 0.700), _hit("Optics", 0.695)])
    )

    result = await tool_executor(
        _state(
            pending_tool="search_knowledge_base",
            tool_query="what is the periodic table organised by",
        ),
        services,
    )

    assert result.get("intent") != "ambiguous"


@pytest.mark.asyncio
async def test_an_empty_result_is_reported_back_to_the_agent():
    services = StubServices(result=RetrievalResult(hits=[]))

    result = await tool_executor(
        _state(pending_tool="search_knowledge_base", tool_query="quantum widgets"), services
    )

    assert result["observations"] == [
        {
            "tool": "search_knowledge_base",
            "query": "quantum widgets",
            "outcome": "no relevant results",
        }
    ]


@pytest.mark.asyncio
async def test_a_successful_result_records_no_observation():
    # Nothing re-enters the agent after a hit, so describing it is dead tokens.
    services = StubServices(result=RetrievalResult(hits=[_hit()]))

    result = await tool_executor(
        _state(pending_tool="search_knowledge_base", tool_query="physics"), services
    )

    assert result["observations"] == []


# ------------------------------------------------------------- the hop gate


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


# ------------------------------------------------------------------- node 5


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


# ------------------------------------------------------------------- node 6


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


# ------------------------------------------------------------------- node 7


@pytest.mark.asyncio
async def test_generator_is_asked_the_users_question_not_the_search_string():
    """"Mona Lisa painter" drew a copular answer the citation check then dropped."""
    services = StubServices(answer="Leonardo painted it [1].")
    captured: list = []

    async def capture(messages):
        captured.append(messages[-1]["content"])
        services.generate_calls += 1
        return "Leonardo painted it [1]."

    services.generate_stream = capture

    await generator(
        _state(
            question="Who painted the Mona Lisa?",
            standalone_query="Mona Lisa painter",
            hits=[_hit("Mona Lisa")],
        ),
        services,
    )

    assert "Who painted the Mona Lisa?" in captured[0]
    assert "Mona Lisa painter" not in captured[0]


@pytest.mark.asyncio
async def test_a_referential_followup_still_gets_the_resolved_question():
    # The one case the rewrite exists for: "his wife" means nothing alone.
    services = StubServices(answer="She was a physicist [1].")
    captured: list = []

    async def capture(messages):
        captured.append(messages[-1]["content"])
        services.generate_calls += 1
        return "She was a physicist [1]."

    services.generate_stream = capture

    await generator(
        _state(
            question="what about his wife?",
            standalone_query="Albert Einstein wife",
            hits=[_hit("Mileva Maric")],
        ),
        services,
    )

    assert "Albert Einstein wife" in captured[0]


@pytest.mark.asyncio
async def test_generator_cites_only_what_the_answer_used():
    services = StubServices(answer="Einstein was a physicist [1].")

    result = await generator(
        _state(hits=[_hit("Albert Einstein"), _hit("Isaac Newton")]), services
    )

    assert [s["title"] for s in result["sources"]] == ["Albert Einstein"]
    assert result["used_live_search"] is False


@pytest.mark.asyncio
async def test_generator_keeps_the_live_flag_when_the_evidence_came_from_live_search():
    services = StubServices(answer="Einstein was a physicist [1].")

    result = await generator(
        _state(hits=[_hit("Albert Einstein")], used_live_search=True), services
    )

    assert result["sources"]
    assert result["used_live_search"] is True


@pytest.mark.asyncio
async def test_generator_refusal_routes_to_live_search():
    # An uncited answer means the chunks were noise; linking them would point
    # the user at articles unrelated to what they asked.
    services = StubServices(answer="The indexed Wikipedia articles don't cover that.")

    result = await generator(_state(hits=[_hit("India")]), services)

    assert result["sources"] == []
    assert result["used_live_search"] is True
    assert result["articles"][0]["title"] == "Live result"
    # The model's own dead-end wording is replaced by something actionable.
    assert "excerpt" not in result["answer"].lower()
    assert "Live result" in result["answer"]


@pytest.mark.asyncio
async def test_the_fallback_search_gets_the_agents_query_not_the_users_sentence():
    """A search string and a question are different things, and this path wanted a search.

    Measured against the live API: "can you tell me about photosynthesis please"
    returns *Robert Anton Wilson, Thure E. Cerling, Isaac Asimov*, while
    "photosynthesis" returns *Photosynthesis*. English fails this way — plausible
    but unrelated suggestions — and smaller editions simply return nothing, which
    is how a Marathi question for प्रकाशसंश्लेषण ended up with no articles at all.
    """
    services = StubServices()

    await generator(
        _state(
            question="can you tell me about photosynthesis please",
            standalone_query="photosynthesis",
            tool_query="photosynthesis",
            hits=[],
        ),
        services,
    )

    assert services.fallback_calls == ["photosynthesis"]


@pytest.mark.asyncio
async def test_the_refusal_names_the_subject_that_was_searched_for():
    """The user should be told what was looked up, not have their sentence quoted back."""
    services = StubServices()

    result = await generator(
        _state(
            question="प्रकाशसंश्लेषणाबद्दल मला सांगा.",
            standalone_query="प्रकाशसंश्लेषण",
            tool_query="प्रकाशसंश्लेषण",
            lang="mr",
            hits=[],
        ),
        services,
    )

    subject, titles, lang = services.refusal_calls[0]
    assert subject == "प्रकाशसंश्लेषण"
    assert titles == ["Live result"]
    # The language travels with it, which is the point: the refusal is written
    # in the user's language rather than picked from a table of six that never
    # included Marathi.
    assert lang == "mr"
    assert "मला सांगा" not in result["answer"]


@pytest.mark.asyncio
async def test_an_uncited_answer_also_falls_back_on_the_search_query():
    """The refusal path reached through a model refusal has the same requirement."""
    services = StubServices(answer="The indexed Wikipedia articles don't cover that.")

    # Deliberately not opening with "so"/"and"/"what about": those are elliptical
    # openers, and for them `_question_for_generator` already returns the resolved
    # query — which would make this pass without the fix.
    await generator(
        _state(
            question="what exactly is a quantum widget anyway",
            standalone_query="quantum widget",
            tool_query="quantum widget",
            hits=[_hit("India")],
        ),
        services,
    )

    assert services.fallback_calls == ["quantum widget"]


@pytest.mark.asyncio
async def test_generator_without_hits_never_asks_the_model_to_answer():
    """The model words the refusal; it is never asked to supply the missing facts.

    The distinction is the whole safety property of this path. `compose_refusal`
    is given a subject and some article titles and told to state no facts about
    them; `generate_stream` is the call that writes grounded answers, and it must
    not run when there is nothing to ground.
    """
    services = StubServices()

    result = await generator(_state(hits=[]), services)

    assert services.generate_calls == 0
    assert len(services.refusal_calls) == 1
    assert result["used_live_search"] is True


# --------------------------------------------------------- compiled routing


def _graph_state(**overrides):
    """Initial state for a full graph run, mirroring the runner's."""
    base = _state(**overrides)
    base.setdefault("tool_calls", [])
    base.setdefault("observations", [])
    base.setdefault("speculative", None)
    return base


@pytest.mark.asyncio
async def test_a_greeting_reaches_end_without_retrieval_or_generation():
    """No vector search and no generation — but the model did decide.

    That is the trade this design makes: a greeting pays one decision call
    instead of zero, and in exchange the set of messages that get a
    conversational answer is whatever the model recognises rather than whatever
    a phrase table lists.
    """
    services = StubServices(
        tool_call=ToolCall("respond_directly", "", "Hello! What would you like to know?")
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question="hello"))

    assert final["intent"] == "chitchat"
    assert final["answer"] == "Hello! What would you like to know?"
    assert services.generate_calls == 0
    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited == ["gate", "agent", "direct_responder"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        # The message that started this: it matched no greeting pattern, so it
        # was searched for in a Wikipedia index and the user was told their own
        # question was not covered by the corpus.
        "do you know me",
        "who are you",
        "thanks a million, that was helpful",
        "are you an AI?",
    ],
)
async def test_a_message_about_the_assistant_never_reaches_the_index(message):
    services = StubServices(
        result=RetrievalResult(hits=[_hit("Albert Einstein")]),
        tool_call=ToolCall("respond_directly", "", "I answer from Wikipedia articles."),
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question=message))

    assert final["answer"] == "I answer from Wikipedia articles."
    # A speculative search does fire under the decision call and is thrown
    # away — that is the cost of racing retrieval rather than sequencing it.
    # What must never happen is any of it reaching the user: no chunks, no
    # citations, and no trip through the executor or the generator.
    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited == ["gate", "agent", "direct_responder"]
    assert final["sources"] == []
    assert final["retrieved_chunks"] == []


@pytest.mark.asyncio
async def test_an_opening_question_still_retrieves_and_answers():
    services = StubServices(
        history=[],
        result=RetrievalResult(hits=[_hit()]),
        tool_call=ToolCall("search_knowledge_base", "what is physics"),
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question="what is physics"))

    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited == ["gate", "agent", "tool_executor", "generator"]
    # The search ran underneath the decision call rather than after it, so the
    # question paid for the decision and got its retrieval free.
    assert final["speculation_hit"] is True
    assert services.retrieve_calls == ["what is physics"]


@pytest.mark.asyncio
async def test_a_followup_retrieves_on_the_agents_resolved_query():
    services = StubServices(
        history=[Turn("user", "Tell me about India")],
        result=RetrievalResult(hits=[_hit("India")]),
        tool_call=ToolCall("search_knowledge_base", "India population"),
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question="what's its population?"))

    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited == ["gate", "agent", "tool_executor", "generator"]
    # The pronoun-laden original never reached the index on its own: the
    # speculation stitched it to its subject, and the agent's resolved query
    # ("India population") is contained in that, so the parked search was
    # reused rather than repeated.
    assert services.retrieve_calls == ["Tell me about India what's its population?"]
    assert final["speculation_hit"] is True


@pytest.mark.asyncio
async def test_an_empty_index_hops_to_live_wikipedia_and_answers():
    """The case the old graph could not express: retry against a different source."""
    services = StubServices(
        history=[],
        result=RetrievalResult(hits=[]),
        live_result=RetrievalResult(hits=[_hit("Recent event")], used_live_search=True),
        tool_call=[
            ToolCall("search_knowledge_base", "saturn rings"),
            # Told the index came back empty, the second hop looks elsewhere.
            ToolCall("search_wikipedia", "saturn rings"),
        ],
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question="what are saturn's rings"))

    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited == [
        "gate",
        "agent",
        "tool_executor",
        "agent",
        "tool_executor",
        "generator",
    ]
    assert services.live_search_calls == ["saturn rings"]
    # The second decision saw what the first search found.
    assert services.decide_calls[1] == [
        {
            "tool": "search_knowledge_base",
            "query": "saturn rings",
            "outcome": "no relevant results",
        }
    ]


@pytest.mark.asyncio
async def test_the_hop_loop_terminates_when_every_source_comes_back_empty():
    services = StubServices(
        history=[],
        result=RetrievalResult(hits=[]),
        live_result=RetrievalResult(hits=[], used_live_search=True),
        tool_call=[
            ToolCall("search_knowledge_base", "nonsense"),
            ToolCall("search_wikipedia", "nonsense"),
        ],
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question="what is a quantum widget"))

    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited.count("agent") == 2  # max_hops, then it stops
    assert visited[-1] == "generator"
    assert final["answer"]


@pytest.mark.asyncio
async def test_ambiguous_retrieval_diverts_to_clarification_not_generation():
    services = StubServices(
        history=[],
        result=RetrievalResult(
            disambiguation=[DisambiguationOption("Mercury (planet)", "u1")]
        ),
        tool_call=ToolCall("search_knowledge_base", "Mercury"),
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question="Mercury"))

    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited == ["gate", "agent", "tool_executor", "clarification"]
    assert services.generate_calls == 0


@pytest.mark.asyncio
async def test_answer_from_history_skips_retrieval_entirely():
    services = StubServices(
        history=[Turn("user", "Who is Einstein"), Turn("assistant", "A physicist.")],
        tool_call=ToolCall("answer_from_history", ""),
    )
    graph = build_graph(services)

    final = await graph.ainvoke(_graph_state(question="what did you just tell me?"))

    visited = [entry["node"] for entry in final["node_timings"]]
    assert visited == ["gate", "agent", "history_answerer"]
    # No *tool* ran, and nothing was fetched from Wikipedia. The speculation
    # did fire and was thrown away — the one path where racing the decision
    # call costs an embedding for nothing. It is rare enough to be worth the
    # win on every other follow-up, and cheap enough not to show in the turn,
    # since it never blocked anything.
    assert services.live_search_calls == []
    assert final["answer"]


# ------------------------------------------------------------- the tool menu


class _FakeCompletions:
    def __init__(self, sink):
        self.sink = sink

    async def create(self, **payload):
        self.sink.append(payload)
        return _Response()


class _Response:
    def __init__(self):
        self.choices = [type("C", (), {"message": _Msg([])})()]


class _FakeClient:
    def __init__(self, sink):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(sink)})()


def _services_with(sink):
    from app.graph.services import GraphServices

    services = GraphServices.__new__(GraphServices)
    services.settings = StubSettings()
    services.settings.agent_model = "gpt-4.1-nano"
    services.client = _FakeClient(sink)
    return services


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


# --------------------------------------------------------------- tool parsing


class _Fn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _Call:
    def __init__(self, name, arguments):
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, calls):
        self.tool_calls = calls


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


# ------------------------------------------------------- speculation heuristic


@pytest.mark.parametrize(
    "message",
    [
        "What did Marie Curie discover?",
        "When did World War II end?",
        "Who painted the Mona Lisa?",
    ],
)
def test_named_subject_is_worth_speculating_on(message):
    assert looks_self_contained(message)


@pytest.mark.parametrize(
    "message",
    [
        # Pronouns and demonstratives: meaningless without the previous turn.
        "What about his wife?",
        "Who built it and why?",
        "What happens at its event horizon?",
        "What is he best known for?",
        "why did that happen",
        # Elliptical without any pronoun at all — the harder half, and the
        # reason a bare "no pronouns" test would not be safe on its own.
        "Tell me more",
        "And the population?",
        "What is the capital?",
        # No named subject, so speculation is not worth the embedding call.
        "What is the speed of light?",
    ],
)
def test_context_dependent_message_is_not_speculated_on(message):
    assert not looks_self_contained(message)


# --------------------------------------------------------- conversational replies


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
