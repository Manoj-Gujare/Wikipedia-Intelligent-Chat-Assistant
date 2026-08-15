"""Unit tests for citation handling, routing, memory and cache."""

from __future__ import annotations

import time

from app.config import Settings
from app.core.cache import TTLCache
from app.core.conversation import ConversationStore
import pytest

from app.core.rag import (
    build_article_links,
    build_sources,
    cited_indices,
    compose_not_covered,
    extract_subject,
    verify_citations,
)
from app.core.retriever import _mmr
from app.core.vector_store import SearchHit


def _hit(title: str, section: str, score: float, body: str = "Body text.") -> SearchHit:
    return SearchHit(
        id=f"{title}-{section}",
        text=f"{title} > {section}\n\n{body}",
        score=score,
        metadata={
            "title": title,
            "section_title": section,
            "section_path": f"{title} > {section}",
            "section_url": f"https://en.wikipedia.org/wiki/{title}#{section}",
            "url": f"https://en.wikipedia.org/wiki/{title}",
            "lang": "en",
            "body": body,
        },
    )


# ----------------------------------------------------------------- citations


def test_cited_indices_are_deduped_and_ordered():
    assert cited_indices("A [2] then B [1][2] and C [1].", 3) == [2, 1]


def test_out_of_range_citations_are_ignored():
    assert cited_indices("Claim [7] and [1].", 2) == [1]


def test_only_cited_sources_are_returned():
    hits = [_hit("Physics", "Intro", 0.9), _hit("Optics", "Intro", 0.8), _hit("Waves", "Intro", 0.7)]

    sources = build_sources(hits, "Light travels fast [1] and refracts [3].")

    assert [s.index for s in sources] == [1, 3]
    assert [s.title for s in sources] == ["Physics", "Waves"]


def test_source_url_is_a_section_deep_link():
    hits = [_hit("Physics", "Optics", 0.9)]

    source = build_sources(hits, "Fact [1].")[0]

    assert source.url == "https://en.wikipedia.org/wiki/Physics#Optics"
    assert source.article_url == "https://en.wikipedia.org/wiki/Physics"


def test_uncited_answer_yields_no_sources():
    hits = [_hit("Physics", "Intro", 0.9)]

    assert build_sources(hits, "I don't have that information.") == []


# -------------------------------------------------- deterministic small talk


# Small talk used to be classified here, by a phrase table these tests pinned.
# The agent classifies it now, so the behaviour they covered is a property of a
# model call and belongs in the evaluation suite rather than a unit test — see
# the routing cases in `scripts/evaluate.py`.


# ------------------------------------------------------------- not-covered


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


# ---------------------------------------------------------- citation checking


def _java_island_hit() -> SearchHit:
    body = (
        "Java is one of the Greater Sunda Islands in the South East Asian country "
        "of Indonesia. Four main languages are spoken on the island: Javanese, "
        "Sundanese, Madurese, and Betawi."
    )
    return _hit("Java", "Introduction", 0.6, body=body)


def test_verify_citations_drops_a_claim_the_cited_chunk_contradicts():
    # The real failure: the island article cited for a programming-language
    # claim. Every other word in the sentence does come from the chunk, so
    # plain term overlap cannot see it.
    hits = [_java_island_hit()]
    answer = (
        "Java is a programming language with four main spoken languages on the "
        "island of Java, including Javanese and Sundanese [1]."
    )

    cleaned, rejected = verify_citations(answer, hits)

    assert cleaned == ""
    assert len(rejected) == 1


def test_verify_citations_keeps_a_correctly_sourced_claim():
    hits = [_java_island_hit()]
    answer = "Java is an island of Indonesia in the Greater Sunda Islands [1]."

    cleaned, rejected = verify_citations(answer, hits)

    assert cleaned == answer
    assert rejected == []


def test_verify_citations_leaves_uncited_sentences_alone():
    # Only citations are held to a citation's standard; connective text is not
    # a sourced claim.
    hits = [_java_island_hit()]
    answer = "Here is what I found. Java is an island of Indonesia [1]."

    cleaned, _ = verify_citations(answer, hits)

    assert "Here is what I found." in cleaned


def test_verify_citations_drops_a_citation_to_unrelated_material():
    hits = [_hit("Photosynthesis", "Introduction", 0.6, body="Plants convert light into sugars.")]
    answer = "The Treaty of Westphalia ended the Thirty Years War in 1648 [1]."

    cleaned, rejected = verify_citations(answer, hits)

    assert cleaned == ""
    assert "term overlap" in rejected[0][1]


def test_verify_citations_removes_the_whole_bullet_not_just_its_text():
    hits = [_java_island_hit()]
    answer = "- Java is a programming language used for Android apps [1].\n- Java is an island of Indonesia [1]."

    cleaned, _ = verify_citations(answer, hits)

    assert "programming language" not in cleaned
    assert cleaned.startswith("- Java is an island")
    assert "- \n" not in cleaned


def test_dropped_sentences_also_lose_their_citation_from_the_source_list():
    # The ordering guarantee: verification runs before sources are built, so a
    # removed claim never leaves evidence behind for a claim that is gone.
    hits = [_java_island_hit()]
    answer = "Java is a programming language with four spoken languages [1]."

    cleaned, _ = verify_citations(answer, hits)

    assert build_sources(hits, cleaned) == []


# -------------------------------------------------------------- smart routing


def test_article_links_are_deduplicated_per_article():
    hits = [
        _hit("Physics", "Intro", 0.9),
        _hit("Physics", "History", 0.85),
        _hit("Optics", "Intro", 0.7),
    ]

    links = build_article_links(hits, [])

    assert [l.title for l in links] == ["Physics", "Optics"]


def test_cited_articles_are_routed_first():
    hits = [_hit("Optics", "Intro", 0.95), _hit("Physics", "Intro", 0.5)]
    sources = build_sources(hits, "Fact [2].")  # cites Physics, the lower-scoring hit

    links = build_article_links(hits, sources)

    assert links[0].title == "Physics"


# ------------------------------------------------------------------------ MMR


def _mmr_fixture() -> list[SearchHit]:
    # `a` and `b` are near-duplicates of each other; `c` is less relevant but
    # covers different material entirely.
    return [
        SearchHit("a", "a", 0.90, {"title": "A"}, embedding=[1.0, 0.0, 0.0]),
        SearchHit("b", "b", 0.89, {"title": "A"}, embedding=[0.99, 0.14, 0.0]),
        SearchHit("c", "c", 0.45, {"title": "B"}, embedding=[0.0, 0.0, 1.0]),
    ]


def test_mmr_prefers_diverse_chunks_over_near_duplicates():
    selected = _mmr(_mmr_fixture(), k=2, lambda_mult=0.5)

    assert {h.id for h in selected} == {"a", "c"}


def test_mmr_with_lambda_one_is_plain_relevance_ranking():
    selected = _mmr(_mmr_fixture(), k=2, lambda_mult=1.0)

    assert {h.id for h in selected} == {"a", "b"}


def test_mmr_relevance_honours_boosted_scores():
    # A lead-section chunk whose stored score was boosted upstream must be able
    # to win a slot on that boosted score, not its raw similarity.
    hits = _mmr_fixture()
    hits[2].score = 0.95  # as if intro_boost promoted it past the duplicates

    selected = _mmr(hits, k=2, lambda_mult=1.0)

    assert {h.id for h in selected} == {"a", "c"}


def test_mmr_passes_through_when_candidates_fit():
    hits = [SearchHit("a", "a", 0.9, {}, embedding=[1.0, 0.0])]

    assert _mmr(hits, k=5, lambda_mult=0.5) == hits


# ------------------------------------------------------------- conversations


def _store(tmp_path, **overrides) -> ConversationStore:
    settings = Settings(
        openai_api_key="test",
        conversations_db=str(tmp_path / "conversations.db"),
        **overrides,
    )
    return ConversationStore(settings)


def test_conversation_ids_persist_across_turns(tmp_path):
    store = _store(tmp_path)
    first = store.get_or_create(None)
    store.append(first.id, "user", "Who was Einstein?")

    again = store.get_or_create(first.id)

    assert again.id == first.id
    assert [t.content for t in store.history(first.id)] == ["Who was Einstein?"]


def test_history_survives_a_new_store_instance(tmp_path):
    # The point of SQLite: a backend restart must not lose conversations.
    store = _store(tmp_path)
    conversation = store.get_or_create(None)
    store.append(conversation.id, "user", "hello")
    store.close()

    reopened = _store(tmp_path)

    assert [t.content for t in reopened.history(conversation.id)] == ["hello"]


def test_assistant_meta_round_trips(tmp_path):
    store = _store(tmp_path)
    conversation = store.get_or_create(None)
    meta = {"sources": [{"index": 1, "title": "Physics"}]}
    store.append(conversation.id, "assistant", "Fact [1].", meta=meta)

    restored = store.history(conversation.id)[0]

    assert restored.meta == meta


def test_peek_never_creates(tmp_path):
    store = _store(tmp_path)

    assert store.peek("nonexistent") is None

    conversation = store.get_or_create(None, lang="es")
    peeked = store.peek(conversation.id)

    assert peeked is not None
    assert peeked.lang == "es"


def test_history_is_trimmed_to_the_configured_window(tmp_path):
    store = _store(tmp_path, conversation_max_turns=4)
    conversation = store.get_or_create(None)
    for i in range(10):
        store.append(conversation.id, "user", f"message {i}")

    history = store.history(conversation.id)

    assert len(history) == 4
    assert history[-1].content == "message 9"


def test_expired_conversations_are_evicted(tmp_path):
    store = _store(tmp_path, conversation_ttl_seconds=0)
    conversation = store.get_or_create(None)
    store.append(conversation.id, "user", "hello")
    time.sleep(0.01)

    revived = store.get_or_create(conversation.id)

    assert revived.turns == []


def test_clear_removes_a_conversation(tmp_path):
    store = _store(tmp_path)
    conversation = store.get_or_create(None)

    assert store.clear(conversation.id) is True
    assert store.clear(conversation.id) is False


# -------------------------------------------------------------------- cache


def _cache() -> TTLCache:
    return TTLCache(Settings(openai_api_key="test", cache_ttl_seconds=60))


def test_cache_key_ignores_case_and_whitespace():
    cache = _cache()

    assert cache.key("  What   is DNA? ", "en") == cache.key("what is dna?", "en")


def test_cache_key_is_language_scoped():
    cache = _cache()

    assert cache.key("What is DNA?", "en") != cache.key("What is DNA?", "es")


def test_cache_key_is_account_scoped():
    # Two accounts search different corpora (shared + their own additions), so
    # the same question can legitimately have different answers.
    cache = _cache()

    assert cache.key("What is DNA?", "en", "acct-a") != cache.key(
        "What is DNA?", "en", "acct-b"
    )


def test_invalidating_an_account_retires_its_cached_answers():
    # After a knowledge-base build the same question must be re-answered.
    cache = _cache()
    before = cache.key("who is MS Dhoni", "en", "acct-a")
    cache.set(before, "stale")

    cache.invalidate("acct-a")

    assert cache.key("who is MS Dhoni", "en", "acct-a") != before
    assert cache.get(cache.key("who is MS Dhoni", "en", "acct-a")) is None


def test_invalidation_does_not_affect_other_accounts():
    cache = _cache()
    other = cache.key("what is DNA", "en", "acct-b")
    cache.set(other, "still good")

    cache.invalidate("acct-a")

    assert cache.get(cache.key("what is DNA", "en", "acct-b")) == "still good"


def test_cache_returns_stored_values_and_counts_hits():
    cache = TTLCache(Settings(openai_api_key="test", cache_ttl_seconds=60))
    cache.set("k", {"answer": "42"})

    assert cache.get("k") == {"answer": "42"}
    assert cache.get("missing") is None
    assert cache.stats() == {"entries": 1, "hits": 1, "misses": 1}


def test_expired_entries_are_dropped():
    cache = TTLCache(Settings(openai_api_key="test", cache_ttl_seconds=0))
    cache.set("k", "v")

    assert cache.get("k") is None
