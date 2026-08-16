"""The not-covered path, and why its two calls overlap.

This was the only benchmark case that missed the 3s budget *reliably* rather
than randomly — 3.50s, 3.71s and 3.82s on three separate runs, while every
other case moved with API variance. The cause was a real dependency: the
refusal named the articles the live search returned, so it could not start
until that search finished. These tests pin the overlap, and pin the two edges
that make breaking the dependency safe.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.graph.services import GraphServices

_DELAY = 0.15


class _Recorder(GraphServices):
    """A services object whose two remote calls are timed, scripted stubs."""

    def __init__(self, articles):
        self._articles = articles
        self.refusal_titles: list[list[str] | None] = []
        self.windows: list[tuple[float, float]] = []

    async def fallback_articles(self, subject, lang="en"):
        start = time.perf_counter()
        await asyncio.sleep(_DELAY)
        self.windows.append((start, time.perf_counter()))
        return list(self._articles)

    async def compose_refusal(self, subject, titles, lang="en"):
        start = time.perf_counter()
        self.refusal_titles.append(None if titles is None else list(titles))
        await asyncio.sleep(_DELAY)
        self.windows.append((start, time.perf_counter()))
        return f"refusal(titles={titles})"


def _article(title="Nikola Tesla"):
    return {"title": title, "url": f"https://en.wikipedia.org/wiki/{title}",
            "lang": "en", "summary": ""}


@pytest.mark.asyncio
async def test_the_search_and_the_wording_overlap():
    """The two calls must run together, which is the entire point of the change.

    Asserted on wall clock rather than call order, because call order is
    identical whether they overlap or not — serial code would still call both.
    """
    services = _Recorder([_article()])

    started = time.perf_counter()
    articles, answer = await services.refuse_and_route("Tesla stock price")
    elapsed = time.perf_counter() - started

    assert elapsed < _DELAY * 1.8, (
        f"expected the two {_DELAY}s calls to overlap, took {elapsed:.3f}s"
    )
    assert articles == [_article()]
    assert answer == "refusal(titles=None)"


@pytest.mark.asyncio
async def test_the_racing_wording_is_never_given_titles():
    """It cannot be: the search that produces them has not returned yet.

    A future refactor that passes titles here would silently re-serialise the
    path, so the contract is pinned rather than left to the timing assertion.
    """
    services = _Recorder([_article()])

    await services.refuse_and_route("Tesla stock price")

    assert services.refusal_titles == [None]


@pytest.mark.asyncio
async def test_no_articles_found_is_re_worded_rather_than_left_promising_a_list():
    """The one case the optimistic wording gets wrong.

    Racing means committing to "the articles are listed under Sources" before
    knowing whether any exist. When none do, that sentence points at an empty
    list, so this case pays the serial cost it used to pay every time.
    """
    services = _Recorder([])

    articles, answer = await services.refuse_and_route("asdfqwerzxcv")

    assert articles == []
    assert services.refusal_titles == [None, []], "should re-word once, with no titles"
    assert answer == "refusal(titles=[])"


@pytest.mark.asyncio
async def test_articles_are_returned_for_routing_even_though_unnamed():
    """The user must still be routed somewhere.

    Dropping the titles from the *wording* is only safe because they survive in
    `articles`, which is what the UI renders as links.
    """
    services = _Recorder([_article("Tesla, Inc."), _article("Nikola Tesla")])

    articles, _ = await services.refuse_and_route("Tesla stock price")

    assert [a["title"] for a in articles] == ["Tesla, Inc.", "Nikola Tesla"]
