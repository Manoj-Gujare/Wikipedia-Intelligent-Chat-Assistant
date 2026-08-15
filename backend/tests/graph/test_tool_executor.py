"""Node 4: running the chosen tool, shaping its result, and spotting ambiguity."""

from __future__ import annotations

import asyncio

import pytest

from app.graph.nodes import tool_executor
from app.core.retriever import DisambiguationOption, RetrievalResult
from .stubs import StubServices, _hit, _state


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
