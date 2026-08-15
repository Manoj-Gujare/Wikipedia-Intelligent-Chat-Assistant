"""MediaWiki client tests against a mocked API (no live requests)."""

from __future__ import annotations

import httpx
import pytest
import respx

from app.config import Settings
from app.core.wikipedia import WikipediaClient, WikipediaError

API = "https://en.wikipedia.org/w/api.php"


def _settings() -> Settings:
    return Settings(
        openai_api_key="test",
        wiki_requests_per_second=1000,  # keep tests fast
        wiki_user_agent="TestAgent/1.0 (test@example.com)",
    )


def _page(**overrides) -> dict:
    page = {
        "pageid": 736,
        "title": "Albert Einstein",
        "fullurl": "https://en.wikipedia.org/wiki/Albert_Einstein",
        "extract": (
            "Albert Einstein was a German-born theoretical physicist who developed "
            "the theory of relativity, one of the two pillars of modern physics "
            "alongside quantum mechanics. He is best known to the general public "
            "for the mass-energy equivalence formula, which has been dubbed the "
            "world's most famous equation. He received the 1921 Nobel Prize in "
            "Physics for his services to theoretical physics, and especially for "
            "his discovery of the law of the photoelectric effect."
            "\n\n== Career ==\nHe published four groundbreaking papers in 1905, "
            "during what is now called his annus mirabilis."
        ),
        "langlinks": [{"lang": "es", "title": "Albert Einstein"}],
    }
    page.update(overrides)
    return page


@pytest.mark.asyncio
@respx.mock
async def test_fetch_article_parses_sections_and_langlinks():
    respx.get(API).mock(return_value=httpx.Response(200, json={"query": {"pages": [_page()]}}))

    async with WikipediaClient(_settings()) as client:
        article = await client.fetch_article("Albert Einstein")

    assert article is not None
    assert article.page_id == 736
    assert [s.title for s in article.sections] == ["Introduction", "Career"]
    assert article.langlinks == {"es": "Albert Einstein"}
    assert article.is_disambiguation is False


@pytest.mark.asyncio
@respx.mock
async def test_requests_carry_a_descriptive_user_agent_and_maxlag():
    route = respx.get(API).mock(
        return_value=httpx.Response(200, json={"query": {"pages": [_page()]}})
    )

    async with WikipediaClient(_settings()) as client:
        await client.fetch_article("Albert Einstein")

    request = route.calls[0].request
    assert "TestAgent/1.0" in request.headers["user-agent"]
    # maxlag makes the API shed our load when replicas fall behind.
    assert request.url.params["maxlag"] == "5"


@pytest.mark.asyncio
@respx.mock
async def test_missing_pages_return_none():
    respx.get(API).mock(
        return_value=httpx.Response(200, json={"query": {"pages": [{"missing": True}]}})
    )

    async with WikipediaClient(_settings()) as client:
        assert await client.fetch_article("Nonexistent Page") is None


@pytest.mark.asyncio
@respx.mock
async def test_stub_articles_are_skipped():
    respx.get(API).mock(
        return_value=httpx.Response(200, json={"query": {"pages": [_page(extract="Too short.")]}})
    )

    async with WikipediaClient(_settings()) as client:
        assert await client.fetch_article("Stub") is None


@pytest.mark.asyncio
@respx.mock
async def test_short_disambiguation_pages_are_still_indexed():
    page = _page(
        title="Mercury",
        extract=(
            "Mercury may refer to:\n\nMercury (planet), the smallest planet in "
            "the Solar System\nMercury (element), a chemical element\n"
            "Mercury (mythology), a Roman god"
        ),
        pageprops={"disambiguation": ""},
    )
    respx.get(API).mock(return_value=httpx.Response(200, json={"query": {"pages": [page]}}))

    async with WikipediaClient(_settings()) as client:
        article = await client.fetch_article("Mercury")

    assert article is not None
    assert article.is_disambiguation is True


@pytest.mark.asyncio
@respx.mock
async def test_maxlag_error_is_retried_then_succeeds():
    responses = [
        httpx.Response(
            200,
            json={"error": {"code": "maxlag", "info": "Waiting for replica"}},
            headers={"Retry-After": "0"},
        ),
        httpx.Response(200, json={"query": {"pages": [_page()]}}),
    ]
    route = respx.get(API).mock(side_effect=responses)

    async with WikipediaClient(_settings()) as client:
        article = await client.fetch_article("Albert Einstein")

    assert article is not None
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_rate_limit_responses_are_retried_with_retry_after():
    responses = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json={"query": {"pages": [_page()]}}),
    ]
    route = respx.get(API).mock(side_effect=responses)

    async with WikipediaClient(_settings()) as client:
        await client.fetch_article("Albert Einstein")

    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_unrecoverable_api_errors_raise():
    respx.get(API).mock(
        return_value=httpx.Response(200, json={"error": {"code": "badvalue", "info": "nope"}})
    )

    async with WikipediaClient(_settings()) as client:
        with pytest.raises(WikipediaError, match="badvalue"):
            await client.fetch_article("Whatever")


@pytest.mark.asyncio
@respx.mock
async def test_disambiguation_options_are_linked():
    respx.get(API).mock(
        return_value=httpx.Response(
            200,
            json={
                "query": {
                    "pages": [
                        {
                            "pageid": 1,
                            "title": "Mercury",
                            "links": [
                                {"title": "Mercury (planet)"},
                                {"title": "Mercury (element)"},
                            ],
                        }
                    ]
                }
            },
        )
    )

    async with WikipediaClient(_settings()) as client:
        options = await client.disambiguation_options("Mercury")

    assert options[0]["title"] == "Mercury (planet)"
    assert options[0]["url"] == "https://en.wikipedia.org/wiki/Mercury_(planet)"


def test_article_url_encodes_titles_and_anchors():
    url = WikipediaClient.article_url("es", "Café de Colombia", anchor="Historia")

    assert url.startswith("https://es.wikipedia.org/wiki/")
    assert url.endswith("#Historia")
    assert " " not in url
