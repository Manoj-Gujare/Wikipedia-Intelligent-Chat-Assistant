"""The article shapes the rest of the application reads.

Kept apart from the client so that chunking, indexing and tests can name an
article without importing an HTTP stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import quote

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
