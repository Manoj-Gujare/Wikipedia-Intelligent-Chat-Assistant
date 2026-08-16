"""Node 7: grounded generation, citation binding, and refusal routing."""

from __future__ import annotations

import asyncio

import pytest

from app.graph.nodes import generator
from .stubs import StubServices, _hit, _state


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
    # Routing is carried by `articles`, which is what the UI renders as links.
    # The wording no longer names them: it is written concurrently with the
    # search that finds them, so it cannot know their titles yet — see
    # `test_refuse_and_route`. What still matters is that the user is routed
    # somewhere real and not left with the model's dead-end phrasing.
    assert result["articles"][0]["title"] == "Live result"
    assert "excerpt" not in result["answer"].lower()
    assert result["answer"] != "The indexed Wikipedia articles don't cover that."


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
    # No titles: this wording races the search that produces them.
    assert titles is None
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
