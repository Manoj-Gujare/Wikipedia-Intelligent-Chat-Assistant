"""Builders shared by the unit tests."""

from __future__ import annotations



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
