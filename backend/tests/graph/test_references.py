"""`looks_self_contained`: whether a message stands on its own."""

from __future__ import annotations


import pytest

from app.graph.nodes import looks_self_contained


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
        # How people actually type. Requiring a *capitalised* proper noun made
        # speculation an artefact of the shift key: measured in the running
        # app, these all fell through to serial retrieval at 0.36-1.29s a turn,
        # while the identical questions in the eval suite — which are written
        # in textbook capitalisation — took the fast path.
        "i want to know about the albert einstein",
        "tell me about albert einstein",
        "who developed the theory of general relativity",
        # No proper noun anywhere, capitalised or not, and still the best
        # possible query for itself: the agent asks for "machine learning",
        # which the message already contains.
        "what is machine learning",
    ],
)
def test_a_lowercase_question_is_still_worth_speculating_on(message):
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
    ],
)
def test_context_dependent_message_is_not_speculated_on(message):
    assert not looks_self_contained(message)


@pytest.mark.parametrize(
    "message",
    [
        # Was listed as "no named subject, not worth the embedding call". It is
        # worth it: the agent's query for this is "speed of light", which the
        # message already contains, so the parked search is reused outright.
        "What is the speed of light?",
    ],
)
def test_a_question_that_is_its_own_best_query_is_speculated_on(message):
    assert looks_self_contained(message)


def test_a_short_elliptical_question_is_an_accepted_false_positive():
    """Documents what dropping the proper-noun proxy costs.

    "What is the capital?" does not stand alone — capital of *what* — and it no
    longer fails the gate, because nothing left in the gate can tell it apart
    from "What is machine learning". So it speculates, and the embedding is
    wasted.

    That is the whole cost, and it is bounded by
    `test_an_accepted_false_positive_is_still_discarded_downstream`: the parked
    result is rejected by `_similar` when the agent resolves the subject, so a
    wasted call is all it can ever become.
    """
    assert looks_self_contained("What is the capital?")
