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
    """Smart routing: one deduplicated link per distinct article, cited first.

    An article the answer never cited is only worth suggesting if it is at
    least as relevant as the weakest chunk the answer actually relied on. That
    bar is self-calibrating — it needs no threshold of its own, because the
    answer has already demonstrated what "relevant enough to use" means for
    this particular question.

    Without it, every retrieved hit became a suggestion. Retrieval keeps
    anything above `min_relevance_score` (0.18) and MMR deliberately
    *diversifies* what survives, so the tail of a good result set is often only
    loosely on topic — and a private knowledge base widens it further. Observed
    in the running UI: "who is albert einstein" returned four cited Einstein
    chunks and then offered *Michael Faraday* and *Sam Altman* as further
    reading. The answer was correct and well sourced; the routing beside it
    read as broken, which costs the same trust.
    """
    cited_titles = {s.title for s in sources}
    # The weakest evidence the answer leaned on. With no citations there is no
    # bar to set, and the caller is on the not-covered path anyway.
    floor = min((s.score for s in sources), default=None)

    ordered: list[ArticleLink] = []
    seen: set[str] = set()

    for hit in sorted(hits, key=lambda h: (h.title not in cited_titles, -h.score)):
        if hit.title in seen:
            continue
        if (
            floor is not None
            and hit.title not in cited_titles
            and hit.score < floor
        ):
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
