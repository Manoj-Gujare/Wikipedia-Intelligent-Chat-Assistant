"""Citation extraction, and holding a sentence to the chunk it cites."""

from __future__ import annotations



from app.core.citations import cited_indices, verify_citations
from app.core.sources import build_sources
from app.core.vector_store import SearchHit
from .factories import _hit


def test_cited_indices_are_deduped_and_ordered():
    assert cited_indices("A [2] then B [1][2] and C [1].", 3) == [2, 1]


def test_out_of_range_citations_are_ignored():
    assert cited_indices("Claim [7] and [1].", 2) == [1]


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
