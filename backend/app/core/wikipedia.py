"""Async MediaWiki API client.

Uses only Wikipedia's official public API (``/w/api.php``). Compliance notes:

* Sends a descriptive ``User-Agent`` with contact info, per the Wikimedia
  User-Agent policy (https://meta.wikimedia.org/wiki/User-Agent_policy).
* Sends ``maxlag`` so we back off automatically when the replica lag is high,
  as recommended for any automated client.
* Self-throttles with a token bucket and honours ``Retry-After`` on 429/503.
* Requests gzip and reuses one connection pool.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import quote

import httpx

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

# A page whose extract is shorter than this is a stub/redirect artefact and not
# worth indexing.
MIN_ARTICLE_CHARS = 400


class WikipediaError(RuntimeError):
    """Raised when the MediaWiki API returns an error we cannot recover from."""


@dataclass(slots=True)
class WikiSection:
    """A ``== heading ==`` block of an article, with its anchor."""

    title: str
    level: int
    text: str
    path: list[str] = field(default_factory=list)

    @property
    def anchor(self) -> str:
        """URL fragment MediaWiki uses for this section heading."""
        # Parentheses are left literal, matching how Wikipedia renders its own links.
        return quote(self.title.replace(" ", "_"), safe="()")


@dataclass(slots=True)
class WikiArticle:
    page_id: int
    title: str
    lang: str
    url: str
    summary: str
    sections: list[WikiSection]
    is_disambiguation: bool = False
    langlinks: dict[str, str] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)


class RateLimiter:
    """Simple async token bucket shared by every request from this client."""

    def __init__(self, rate_per_second: float) -> None:
        self._rate = max(rate_per_second, 0.5)
        self._interval = 1.0 / self._rate
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait_for = self._next_slot - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
                now = time.monotonic()
            self._next_slot = now + self._interval


class WikipediaClient:
    """Thin async wrapper over the MediaWiki action API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._limiter = RateLimiter(self.settings.wiki_requests_per_second)
        self._semaphore = asyncio.Semaphore(self.settings.wiki_max_concurrency)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=10.0),
            headers={
                "User-Agent": self.settings.wiki_user_agent,
                "Accept-Encoding": "gzip",
                "Api-User-Agent": self.settings.wiki_user_agent,
            },
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "WikipediaClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ core

    @staticmethod
    def api_url(lang: str) -> str:
        return f"https://{lang}.wikipedia.org/w/api.php"

    @staticmethod
    def article_url(lang: str, title: str, anchor: str | None = None) -> str:
        base = f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='()')}"
        return f"{base}#{anchor}" if anchor else base

    async def _get(self, lang: str, params: dict[str, Any], *, retries: int = 4) -> dict:
        query = {
            "format": "json",
            "formatversion": "2",
            "maxlag": self.settings.wiki_maxlag,
            **params,
        }
        last_error: Exception | None = None

        for attempt in range(retries):
            await self._limiter.acquire()
            async with self._semaphore:
                try:
                    response = await self._client.get(self.api_url(lang), params=query)
                except httpx.HTTPError as exc:  # network blip -> retry
                    last_error = exc
                    await asyncio.sleep(2**attempt)
                    continue

            if response.status_code in (429, 503):
                # Wikimedia tells us how long to wait; respect it.
                delay = float(response.headers.get("Retry-After", 2**attempt))
                logger.warning("Wikipedia throttled us, sleeping %.1fs", delay)
                await asyncio.sleep(delay)
                continue

            if response.status_code >= 500:
                await asyncio.sleep(2**attempt)
                continue

            response.raise_for_status()
            payload = response.json()

            error = payload.get("error")
            if error:
                if error.get("code") == "maxlag":
                    delay = float(response.headers.get("Retry-After", 5))
                    logger.warning("Replica lag high, sleeping %.1fs", delay)
                    await asyncio.sleep(delay)
                    continue
                raise WikipediaError(f"{error.get('code')}: {error.get('info')}")

            return payload

        raise WikipediaError(
            f"Gave up on {params.get('list') or params.get('prop') or 'request'} "
            f"after {retries} attempts"
        ) from last_error

    # --------------------------------------------------------------- queries

    async def search(self, query: str, lang: str = "en", limit: int = 5) -> list[dict]:
        """Full-text search; used as a live fallback when the index misses."""
        payload = await self._get(
            lang,
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srnamespace": 0,
                "srprop": "snippet",
            },
        )
        return payload.get("query", {}).get("search", [])

    async def category_members(
        self, category: str, lang: str = "en", limit: int = 200
    ) -> list[str]:
        """Article titles directly inside ``Category:<category>``."""
        titles: list[str] = []
        cont: dict[str, Any] = {}
        title = category if category.lower().startswith("category:") else f"Category:{category}"

        while len(titles) < limit:
            payload = await self._get(
                lang,
                {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": title,
                    "cmtype": "page",
                    "cmnamespace": 0,
                    "cmlimit": min(500, limit - len(titles)),
                    **cont,
                },
            )
            titles.extend(m["title"] for m in payload.get("query", {}).get("categorymembers", []))
            cont = payload.get("continue", {})
            if not cont:
                break

        return titles[:limit]

    async def page_links(
        self, page: str, lang: str = "en", limit: int = 2000
    ) -> list[str]:
        """Article titles linked from ``page``.

        Used to read Wikipedia's own curation lists (``Wikipedia:Vital
        articles/…``), which are ordinary pages whose links *are* the list.
        ``plnamespace=0`` keeps the result to real articles, dropping the
        template, project and portal links that make up most of such a page's
        markup.
        """
        titles: list[str] = []
        cont: dict[str, Any] = {}

        while len(titles) < limit:
            payload = await self._get(
                lang,
                {
                    "action": "query",
                    "titles": page,
                    "prop": "links",
                    "plnamespace": 0,
                    "pllimit": "max",
                    **cont,
                },
            )
            # formatversion=2 returns `pages` as a list; the legacy shape is a
            # dict keyed by page id. Accept both so this does not break if the
            # format ever changes underneath us.
            pages = payload.get("query", {}).get("pages", [])
            for entry in pages.values() if isinstance(pages, dict) else pages:
                titles.extend(link["title"] for link in entry.get("links", []))
            cont = payload.get("continue", {})
            if not cont:
                break

        return titles[:limit]

    async def subpages(
        self, prefix: str, lang: str = "en", namespace: int = 4, limit: int = 500
    ) -> list[str]:
        """Titles under ``prefix`` in ``namespace`` (4 = ``Wikipedia:``).

        Discovering these at runtime rather than hard-coding them matters: the
        Vital articles project reorganises its own subpages, and the obvious
        guesses are wrong in a way that fails silently. ``…/Level/4/Technology``
        exists but is an empty stub, while the live list is at
        ``…/Level 4/Technology`` — both return HTTP 200, so a hard-coded path
        yields zero titles and no error.
        """
        titles: list[str] = []
        cont: dict[str, Any] = {}

        while len(titles) < limit:
            payload = await self._get(
                lang,
                {
                    "action": "query",
                    "list": "allpages",
                    "apnamespace": namespace,
                    "apprefix": prefix,
                    "aplimit": "max",
                    **cont,
                },
            )
            titles.extend(p["title"] for p in payload.get("query", {}).get("allpages", []))
            cont = payload.get("continue", {})
            if not cont:
                break

        return titles[:limit]

    async def fetch_article(self, title: str, lang: str = "en") -> WikiArticle | None:
        """Fetch one article's full plain-text extract, metadata and langlinks.

        ``exsectionformat=wiki`` keeps ``== Heading ==`` markers in the plain
        text, which is what lets us chunk along real section boundaries and
        build deep links.
        """
        payload = await self._get(
            lang,
            {
                "action": "query",
                "prop": "extracts|info|pageprops|langlinks",
                "titles": title,
                "explaintext": 1,
                "exsectionformat": "wiki",
                "exlimit": 1,
                "inprop": "url",
                "redirects": 1,
                "lllimit": 100,
            },
        )

        pages = payload.get("query", {}).get("pages", [])
        if not pages:
            return None

        page = pages[0]
        if page.get("missing") or page.get("invalid"):
            return None

        extract = page.get("extract") or ""
        page_props = page.get("pageprops") or {}
        is_disambig = "disambiguation" in page_props

        # Disambiguation pages are short by nature but we want them indexed, so
        # the assistant can recognise an ambiguous question and offer choices.
        min_chars = 120 if is_disambig else MIN_ARTICLE_CHARS
        if len(extract) < min_chars:
            return None

        return WikiArticle(
            page_id=page["pageid"],
            title=page["title"],
            lang=lang,
            url=page.get("fullurl") or self.article_url(lang, page["title"]),
            summary=_lead_paragraph(extract),
            sections=parse_sections(extract),
            is_disambiguation=is_disambig,
            langlinks={ll["lang"]: ll["title"] for ll in page.get("langlinks", [])},
        )

    async def fetch_articles(
        self, titles: Iterable[str], lang: str = "en"
    ) -> list[WikiArticle]:
        """Fetch many articles concurrently (bounded by the client semaphore)."""

        async def one(title: str) -> WikiArticle | None:
            try:
                return await self.fetch_article(title, lang=lang)
            except WikipediaError:
                logger.warning("Skipping %r: API error", title)
                return None

        results = await asyncio.gather(*(one(t) for t in titles))
        return [a for a in results if a is not None]

    async def disambiguation_options(
        self, title: str, lang: str = "en", limit: int = 8
    ) -> list[dict[str, str]]:
        """The candidate articles a disambiguation page points at, in page order.

        ``prop=links`` alone returns titles alphabetically, which for "Mercury"
        surfaces *Anna Kavan* and *Blackburn Mercury* while burying the planet
        and the element. So we take two things in one request: the link set
        (authoritative article titles) and the plain-text extract (which
        preserves the order a reader sees), then walk the extract and emit the
        links in the order they are actually listed.
        """
        payload = await self._get(
            lang,
            {
                "action": "query",
                "prop": "links|extracts",
                "titles": title,
                "plnamespace": 0,
                "pllimit": 500,
                "explaintext": 1,
                "exlimit": 1,
                "redirects": 1,
            },
        )
        pages = payload.get("query", {}).get("pages", [])
        if not pages:
            return []

        page = pages[0]
        link_titles = [link["title"] for link in page.get("links", [])]
        if not link_titles:
            return []

        ordered = self._order_by_extract(page.get("extract") or "", link_titles)
        return [
            {"title": t, "url": self.article_url(lang, t)} for t in ordered[:limit]
        ]

    @staticmethod
    def _order_by_extract(extract: str, link_titles: list[str]) -> list[str]:
        """Sort link titles by where they appear in the page's plain text."""
        if not extract:
            return link_titles

        # Longest first, so "Mercury (planet)" wins over a bare "Mercury".
        by_length = sorted(link_titles, key=len, reverse=True)
        positions: dict[str, int] = {}

        for line in extract.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            for candidate in by_length:
                if candidate in positions:
                    continue
                # A disambiguation entry names its target at the line start.
                if stripped.startswith(candidate):
                    positions[candidate] = len(positions)
                    break

        if not positions:
            return link_titles

        ranked = sorted(positions, key=positions.get)  # type: ignore[arg-type]
        # Anything the extract never mentioned goes last, order preserved.
        return ranked + [t for t in link_titles if t not in positions]


def _lead_paragraph(extract: str) -> str:
    """First paragraph before any ``== Heading ==``."""
    head = extract.split("\n==", 1)[0].strip()
    for para in head.split("\n"):
        para = para.strip()
        if len(para) > 80:
            return para
    return head[:600]


def parse_sections(extract: str) -> list[WikiSection]:
    """Split a plain-text extract into sections using its ``==`` headings.

    The text before the first heading becomes an implicit "Introduction"
    section, which is where most short factual answers live.
    """
    sections: list[WikiSection] = []
    current = {"title": "Introduction", "level": 1, "ancestors": []}
    current_lines: list[str] = []
    # Tracks the heading hierarchy so each section knows its parent path.
    stack: list[tuple[int, str]] = []

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if not text:
            return
        sections.append(
            WikiSection(
                title=str(current["title"]),
                level=int(current["level"]),
                text=text,
                path=[*current["ancestors"], str(current["title"])],
            )
        )

    for raw_line in extract.splitlines():
        line = raw_line.strip()
        if line.startswith("==") and line.endswith("==") and len(line) > 4:
            flush()
            equals = len(line) - len(line.lstrip("="))
            level = max(equals - 1, 1)
            while stack and stack[-1][0] >= level:
                stack.pop()
            current = {
                "title": line.strip("=").strip(),
                "level": level,
                "ancestors": [t for _, t in stack],
            }
            current_lines = []
            stack.append((level, str(current["title"])))
        else:
            current_lines.append(raw_line)

    flush()

    # Drop boilerplate tail sections that carry no answerable content. The check
    # walks the whole heading path, not just the leaf: "Further reading >
    # Articles" is still a bibliography, and left in it wastes retrieval slots
    # on citation lists.
    noise = {
        "references",
        "external links",
        "see also",
        "further reading",
        "notes",
        "bibliography",
        "sources",
        "citations",
        "footnotes",
        "works cited",
    }
    return [
        s
        for s in sections
        if not any(part.strip().lower() in noise for part in s.path)
    ]
