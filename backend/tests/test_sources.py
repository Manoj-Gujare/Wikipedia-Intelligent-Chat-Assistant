"""Building the evidence list, and the smart-routing article links."""

from __future__ import annotations



from app.core.sources import build_article_links, build_sources
from .factories import _hit


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
