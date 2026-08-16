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


def test_an_uncited_article_weaker_than_the_evidence_used_is_not_suggested():
    """The bug from the running UI, in miniature.

    "who is albert einstein" answered from four Einstein chunks and then
    offered *Michael Faraday* and *Sam Altman* as further reading — both
    retrieved, neither cited, and one of them from a private knowledge base
    about something else entirely. Retrieval keeps everything above a
    permissive floor and MMR diversifies what survives, so the tail of a good
    result set is routinely off topic.
    """
    hits = [
        _hit("Albert Einstein", "Intro", 0.72),
        _hit("Albert Einstein", "Career", 0.61),
        _hit("Michael Faraday", "Intro", 0.44),
        _hit("Sam Altman", "Intro", 0.31),
    ]
    sources = build_sources(hits, "A physicist [1]. He worked on quanta [2].")

    links = build_article_links(hits, sources)

    assert [l.title for l in links] == ["Albert Einstein"]


def test_an_uncited_article_as_relevant_as_the_evidence_used_is_kept():
    """The bar is 'as relevant as what the answer used', not 'cited'.

    A genuinely related article that the model simply did not need is still
    worth routing to — that is the difference between filtering noise and
    refusing to suggest anything.
    """
    hits = [
        _hit("Black hole", "Intro", 0.80),
        _hit("Event horizon", "Intro", 0.74),
        _hit("Tidal downsizing", "Intro", 0.10),
    ]
    sources = build_sources(hits, "Nothing escapes [1].")  # floor = 0.80

    links = build_article_links(hits, sources)

    assert "Tidal downsizing" not in [l.title for l in links]

    # Same corpus, but the answer leaned on weaker evidence, so the bar drops
    # and the related article clears it.
    sources = build_sources(hits, "Nothing escapes [1]. Its boundary [2].")
    links = build_article_links(hits, sources)

    assert [l.title for l in links] == ["Black hole", "Event horizon"]


def test_cited_articles_are_routed_first():
    hits = [_hit("Optics", "Intro", 0.95), _hit("Physics", "Intro", 0.5)]
    sources = build_sources(hits, "Fact [2].")  # cites Physics, the lower-scoring hit

    links = build_article_links(hits, sources)

    assert links[0].title == "Physics"
