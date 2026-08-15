"""Chunking and section-parsing tests. No network or API key required."""

from __future__ import annotations

from app.core.chunker import chunk_article, count_tokens
from app.core.wikipedia import WikiArticle, parse_sections

EXTRACT = """Albert Einstein was a German-born theoretical physicist who developed \
the theory of relativity, one of the two pillars of modern physics.

His work is also known for its influence on the philosophy of science.

== Early life ==
Einstein was born in Ulm in 1879 to a Jewish family.

=== Education ===
He attended the Swiss Federal Polytechnic in Zurich.

== Scientific career ==
In 1905 he published four groundbreaking papers.

== References ==
Some citation noise that should never be indexed.
"""


def _article(extract: str = EXTRACT) -> WikiArticle:
    return WikiArticle(
        page_id=736,
        title="Albert Einstein",
        lang="en",
        url="https://en.wikipedia.org/wiki/Albert_Einstein",
        summary="German-born theoretical physicist.",
        sections=parse_sections(extract),
    )


def test_parse_sections_splits_on_headings():
    sections = parse_sections(EXTRACT)
    titles = [s.title for s in sections]

    assert titles == ["Introduction", "Early life", "Education", "Scientific career"]


def test_parse_sections_drops_boilerplate():
    assert all(s.title != "References" for s in parse_sections(EXTRACT))


def test_boilerplate_subsections_are_dropped_too():
    # A subsection of "Further reading" is still a bibliography. Keeping it lets
    # citation lists win retrieval slots away from real content.
    extract = (
        "Intro text that is long enough to survive the minimum length filter here.\n"
        "== Further reading ==\n"
        "=== Articles ===\n"
        "Hodges, Andrew (1983). Alan Turing: The Enigma.\n"
        "== Legacy ==\n"
        "His name lives on in the Turing Award.\n"
    )

    titles = [s.title for s in parse_sections(extract)]

    assert "Articles" not in titles
    assert "Legacy" in titles


def test_subsection_records_its_parent_path():
    education = next(s for s in parse_sections(EXTRACT) if s.title == "Education")

    assert education.path == ["Early life", "Education"]
    assert education.level == 2


def test_section_anchor_is_url_safe():
    sections = parse_sections("== Early life and career ==\nSome text here about things.\n")

    assert sections[0].anchor == "Early_life_and_career"


def test_chunks_carry_deep_links_to_their_section():
    chunks = chunk_article(_article())
    career = next(c for c in chunks if c.metadata["section_title"] == "Scientific career")

    assert career.metadata["section_url"].endswith("#Scientific_career")
    assert career.metadata["title"] == "Albert Einstein"
    assert career.metadata["lang"] == "en"


def test_intro_chunk_links_to_the_article_not_an_anchor():
    chunks = chunk_article(_article())
    intro = next(c for c in chunks if c.metadata["section_title"] == "Introduction")

    assert "#" not in intro.metadata["section_url"]


def test_chunk_text_is_prefixed_with_a_breadcrumb():
    chunks = chunk_article(_article())
    education = next(c for c in chunks if c.metadata["section_title"] == "Education")

    assert education.text.startswith("Albert Einstein > Early life > Education")


def test_long_sections_are_split_with_overlap(monkeypatch):
    body = "\n".join(
        f"Paragraph {i} contains a reasonable amount of prose about the subject "
        f"at hand, enough to consume a meaningful number of tokens." for i in range(40)
    )
    article = _article(f"== Long section ==\n{body}\n")

    chunks = chunk_article(article)
    long_chunks = [c for c in chunks if c.metadata["section_title"] == "Long section"]

    assert len(long_chunks) > 1
    # Overlap means consecutive chunks share trailing/leading content.
    assert long_chunks[0].metadata["body"].split("\n\n")[-1] in long_chunks[1].metadata["body"]


def test_chunks_respect_the_token_budget():
    body = " ".join("token" for _ in range(4000))
    chunks = chunk_article(_article(f"== Dense ==\n{body}\n"))

    assert chunks
    assert all(count_tokens(c.metadata["body"]) <= 400 for c in chunks)


def test_chunk_ids_are_unique():
    chunks = chunk_article(_article())

    assert len({c.id for c in chunks}) == len(chunks)
