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
