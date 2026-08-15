"""Turning cited chunks into the evidence the UI shows.

Only chunks the answer actually cited become sources -- listing a retrieved
chunk the model never used would imply evidence for a claim nobody made. The
citation numbers are never renumbered, so the markers in the text stay valid.
"""

from __future__ import annotations

from ..models import ArticleLink, ChatResponse, Source
from .citations import cited_indices
from .vector_store import SearchHit

def _snippet(hit: SearchHit, limit: int = 260) -> str:
    body = (hit.metadata.get("body") or hit.text).strip().replace("\n", " ")
    return body if len(body) <= limit else body[: limit - 1].rsplit(" ", 1)[0] + "…"

def build_sources(hits: list[SearchHit], answer: str) -> list[Source]:
    """Keep only cited chunks; markers stay valid because we do not renumber."""
    used = cited_indices(answer, len(hits))
    # If the model cited nothing (e.g. a refusal), show nothing rather than
    # implying evidence that was not used.
    return [
        Source(
            index=n,
            title=hits[n - 1].title,
            section=str(hits[n - 1].metadata.get("section_title", "")),
            url=hits[n - 1].section_url,
            article_url=str(hits[n - 1].metadata.get("url", "")),
            lang=str(hits[n - 1].metadata.get("lang", "en")),
            snippet=_snippet(hits[n - 1]),
            score=round(hits[n - 1].score, 4),
        )
        for n in used
    ]

def turn_meta(response: ChatResponse) -> dict | None:
    """Citation payload stored with an assistant turn, so a restored
    conversation keeps its sources and article links, not just its text."""
    if not response.sources and not response.articles:
        return None
    return {
        "sources": [s.model_dump() for s in response.sources],
        "articles": [a.model_dump() for a in response.articles],
    }

def build_article_links(hits: list[SearchHit], sources: list[Source]) -> list[ArticleLink]:
    """Smart routing: one deduplicated link per distinct article, cited first."""
    cited_titles = {s.title for s in sources}
    ordered: list[ArticleLink] = []
    seen: set[str] = set()

    for hit in sorted(hits, key=lambda h: (h.title not in cited_titles, -h.score)):
        if hit.title in seen:
            continue
        seen.add(hit.title)
        ordered.append(
            ArticleLink(
                title=hit.title,
                url=str(hit.metadata.get("url", hit.section_url)),
                lang=str(hit.metadata.get("lang", "en")),
                summary=_snippet(hit, limit=160),
            )
        )
    return ordered[:5]
