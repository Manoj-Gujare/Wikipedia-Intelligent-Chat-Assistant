"""Subject extraction and the not-covered wording."""

from __future__ import annotations


import pytest

from app.core.refusals import compose_not_covered, extract_subject


def test_not_covered_names_the_articles_to_read_instead():
    text = compose_not_covered("mitochondria", ["Mitochondria", "Mitochondrial DNA", "Eukaryote"])

    assert "mitochondria" in text
    assert "“Mitochondria”" in text and "“Eukaryote”" in text
    assert "Sources" in text


def test_not_covered_lists_at_most_three_articles():
    text = compose_not_covered("x", ["A", "B", "C", "D", "E"])

    assert "“D”" not in text and "“E”" not in text


def test_not_covered_reads_naturally_with_a_single_article():
    text = compose_not_covered("MS Dhoni", ["MS Dhoni"])

    assert "“MS Dhoni”" in text
    assert " and " not in text.split("Have a look at")[1].split(",")[0]


def test_not_covered_without_articles_says_so_plainly():
    text = compose_not_covered("zxqv", [])

    assert "couldn't find" in text
    assert "Sources" not in text


def test_not_covered_never_leaks_internal_vocabulary():
    # The model's own refusals said things like "the excerpts do not cover
    # mitochondria", which means nothing to a user.
    for text in (compose_not_covered("t", ["A"]), compose_not_covered("t", [])):
        lowered = text.lower()
        assert "excerpt" not in lowered
        assert "chunk" not in lowered
        assert "context" not in lowered


def test_not_covered_handles_an_empty_topic():
    assert compose_not_covered("", ["A"]).strip()


@pytest.mark.parametrize(
    "topic,expected",
    [
        ("and about his wife", "his wife"),
        ("And about mitochondria", "mitochondria"),
        ("what about his wife", "his wife"),
        ("tell me about quasars", "quasars"),
        ("mitochondria", "mitochondria"),
    ],
)
def test_not_covered_strips_conversational_scaffolding(topic, expected):
    # A raw follow-up quoted back verbatim reads as broken English:
    # "I don't have anything about and about his wife".
    text = compose_not_covered(topic, ["A"])

    assert f"about {expected} in your knowledge base" in text


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is the current stock price of Tesla?", "the current stock price of Tesla"),
        ("Who is the CEO of OpenAI?", "the CEO of OpenAI"),
        ("Tell me about the Nazca lines", "the Nazca lines"),
        ("Mercury", "Mercury"),
        ("the Silk Road", "the Silk Road"),
    ],
)
def test_extract_subject_returns_a_quotable_noun_phrase(question, expected):
    assert extract_subject(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "When did the Berlin Wall fall?",
        "How does a nuclear reactor work?",
        "Why is the sky blue?",
        "Where can I buy bitcoin today?",
        "Give me the latest football scores",
    ],
)
def test_extract_subject_declines_when_it_would_read_as_nonsense(question):
    # Stripping "When did " leaves "the Berlin Wall fall", which is
    # ungrammatical after "anything about". Better to say nothing.
    assert extract_subject(question) is None


def test_not_covered_never_echoes_a_whole_question_back():
    text = compose_not_covered("What is the current stock price of Tesla?", ["Tesla, Inc."])

    assert "about What is" not in text
    assert "the current stock price of Tesla" in text


def test_not_covered_falls_back_to_a_generic_subject():
    text = compose_not_covered("When did the Berlin Wall fall?", ["Berlin Wall"])

    assert "about that in your knowledge base" in text
    assert "When did" not in text


@pytest.mark.parametrize("lang,marker", [("es", "Todavía no tengo"), ("fr", "Je n'ai encore rien")])
def test_not_covered_answers_in_the_caller_language(lang, marker):
    # A Spanish question answered with an English apology is its own miss.
    assert compose_not_covered("penicilina", ["Penicilina"], lang).startswith(marker)


def test_not_covered_falls_back_to_english_for_unlisted_languages():
    assert compose_not_covered("x", ["A"], "ja").startswith("I don't have anything")
