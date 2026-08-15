"""Section-aware chunking for long Wikipedia articles.

Strategy
--------
1. Split the article along its real ``== Section ==`` boundaries (done in
   ``wikipedia.parse_sections``) so a chunk never straddles two topics.
2. Pack whole paragraphs into token-bounded windows; only fall back to
   sentence splitting when a single paragraph exceeds the window.
3. Carry a token overlap between consecutive windows so a fact split across a
   boundary is still retrievable from at least one chunk.
4. Prefix every chunk with a breadcrumb (``Article > Section > Subsection``)
   before embedding. The embedding then encodes *where* the text came from,
   which measurably improves retrieval on short, pronoun-heavy chunks.

Each chunk keeps a deep link (``…/wiki/Title#Section``) so answers can cite a
section, not just an article.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import tiktoken

from ..config import Settings, get_settings
from .wikipedia import WikiArticle, WikipediaClient

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@lru_cache
def _encoder() -> tiktoken.Encoding:
    # cl100k_base covers the text-embedding-3-* and gpt-4o-* families.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _split_paragraph(paragraph: str, max_tokens: int) -> list[str]:
    """Break an oversized paragraph on sentence boundaries."""
    sentences = _SENTENCE_END.split(paragraph)
    pieces: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for sentence in sentences:
        tokens = count_tokens(sentence)
        if buffer and buffer_tokens + tokens > max_tokens:
            pieces.append(" ".join(buffer))
            buffer, buffer_tokens = [], 0
        # A single sentence longer than the window is hard-split on tokens.
        if tokens > max_tokens:
            encoder = _encoder()
            ids = encoder.encode(sentence)
            for start in range(0, len(ids), max_tokens):
                pieces.append(encoder.decode(ids[start : start + max_tokens]))
            continue
        buffer.append(sentence)
        buffer_tokens += tokens

    if buffer:
        pieces.append(" ".join(buffer))
    return [p.strip() for p in pieces if p.strip()]


def _windows(paragraphs: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Pack paragraphs into overlapping token windows."""
    units: list[tuple[str, int]] = []
    for paragraph in paragraphs:
        tokens = count_tokens(paragraph)
        if tokens > max_tokens:
            units.extend((piece, count_tokens(piece)) for piece in _split_paragraph(paragraph, max_tokens))
        else:
            units.append((paragraph, tokens))

    chunks: list[str] = []
    buffer: list[tuple[str, int]] = []
    buffer_tokens = 0

    for unit, tokens in units:
        if buffer and buffer_tokens + tokens > max_tokens:
            chunks.append("\n\n".join(text for text, _ in buffer))
            # Rewind by up to `overlap_tokens` worth of trailing units. Both
            # bounds matter: the carry must fit the overlap budget *and* leave
            # room for the incoming unit, or the next chunk blows the window.
            carry: list[tuple[str, int]] = []
            carried = 0
            for prev in reversed(buffer):
                if (
                    carried + prev[1] > overlap_tokens
                    or carried + prev[1] + tokens > max_tokens
                ):
                    break
                carry.insert(0, prev)
                carried += prev[1]
            buffer = carry
            buffer_tokens = carried
        buffer.append((unit, tokens))
        buffer_tokens += tokens

    if buffer:
        chunks.append("\n\n".join(text for text, _ in buffer))
    return chunks


def chunk_article(article: WikiArticle, settings: Settings | None = None) -> list[Chunk]:
    """Turn one article into embeddable, individually citable chunks."""
    settings = settings or get_settings()
    max_tokens = settings.chunk_size_tokens
    overlap = min(settings.chunk_overlap_tokens, max_tokens // 2)

    chunks: list[Chunk] = []
    index = 0

    for section in article.sections:
        paragraphs = [p.strip() for p in section.text.split("\n") if p.strip()]
        if not paragraphs:
            continue

        breadcrumb = " > ".join([article.title, *section.path])
        section_url = WikipediaClient.article_url(
            article.lang,
            article.title,
            anchor=None if section.title == "Introduction" else section.anchor,
        )

        for window in _windows(paragraphs, max_tokens, overlap):
            if count_tokens(window) < 8:
                # Stray fragment (a leftover caption or heading), not content.
                # Short sections are kept: with their breadcrumb they still
                # answer questions, and dropping them loses real article text.
                continue

            # The breadcrumb is embedded with the body so the vector encodes context.
            embed_text = f"{breadcrumb}\n\n{window}"
            digest = hashlib.sha1(
                f"{article.lang}:{article.page_id}:{index}:{window[:120]}".encode()
            ).hexdigest()[:20]

            chunks.append(
                Chunk(
                    id=f"{article.lang}-{article.page_id}-{index}-{digest}",
                    text=embed_text,
                    metadata={
                        "page_id": article.page_id,
                        "title": article.title,
                        "lang": article.lang,
                        "url": article.url,
                        "section_title": section.title,
                        "section_path": breadcrumb,
                        "section_anchor": section.anchor,
                        "section_url": section_url,
                        "chunk_index": index,
                        "is_disambiguation": article.is_disambiguation,
                        "body": window,
                        "token_count": count_tokens(window),
                    },
                )
            )
            index += 1

    return chunks
