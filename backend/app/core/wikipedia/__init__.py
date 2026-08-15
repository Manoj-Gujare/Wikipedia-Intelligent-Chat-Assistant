"""Wikipedia access: the API client, the article shapes, and extract parsing.

Re-exported here so callers keep importing ``app.core.wikipedia`` without
caring which module inside the package a name lives in.
"""

from .client import WikipediaClient
from .models import (
    MIN_ARTICLE_CHARS,
    WikiArticle,
    WikipediaError,
    WikiSection,
)
from .parsing import lead_paragraph, parse_sections
from .rate_limit import RateLimiter

__all__ = [
    "MIN_ARTICLE_CHARS",
    "RateLimiter",
    "WikiArticle",
    "WikiSection",
    "WikipediaClient",
    "WikipediaError",
    "lead_paragraph",
    "parse_sections",
]
