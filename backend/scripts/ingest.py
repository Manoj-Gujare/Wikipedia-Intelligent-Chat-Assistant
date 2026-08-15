"""Crawl Wikipedia via the MediaWiki API, chunk, embed, and index.

Usage (from the ``backend/`` directory):

    python -m scripts.ingest                       # ~300 English articles
    python -m scripts.ingest --limit 50            # quick demo index
    python -m scripts.ingest --lang es --limit 100 # Spanish index
    python -m scripts.ingest --mirror-langs es,fr --limit 60
    python -m scripts.ingest --titles "Black hole,Neutron star"
    python -m scripts.ingest --reset               # rebuild from scratch

The crawler is resumable: articles already in the index are skipped unless
``--reset`` or ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from openai import AuthenticationError

from app.config import Settings, get_settings
from app.core.chunker import Chunk, chunk_article
from app.core.embeddings import Embedder
from app.core.vector_store import VectorStore
from app.core.wikipedia import WikiArticle, WikipediaClient

from .seeds import CATEGORIES, curated_titles, vital_titles

# Non-English editions carry non-Latin titles, and a Windows console defaults
# to cp1252 — which cannot encode them. Force UTF-8 before any logging so a
# crawl cannot die on the character set of the content it is indexing.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # pragma: no cover - non-reconfigurable stream
        pass

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
)
logger = logging.getLogger("ingest")

# Fetch articles in waves so we surface progress and bound memory use.
FETCH_BATCH = 20


async def collect_titles(
    client: WikipediaClient,
    lang: str,
    limit: int,
    use_categories: bool,
    use_vital: bool = True,
) -> list[str]:
    """Curated titles, then Vital articles, then category members.

    Curated stays first so the demo articles and the deliberately ambiguous
    titles are always present regardless of `--limit`. Vital articles supply the
    bulk, and categories only top up whatever is still missing.
    """
    titles: list[str] = []
    seen: set[str] = set()

    for title in curated_titles():
        if title not in seen:
            seen.add(title)
            titles.append(title)
        if len(titles) >= limit:
            return titles[:limit]

    # The Vital articles project is an English-Wikipedia effort, and its lists
    # hold English titles. Other editions are populated by `--mirror-langs`,
    # which maps these same titles through langlinks.
    if use_vital and lang == "en" and len(titles) < limit:
        for title in await vital_titles(client, limit=limit, lang=lang):
            if title in seen:
                continue
            seen.add(title)
            titles.append(title)
            if len(titles) >= limit:
                break
        logger.info("Vital articles brought the pool to %d titles", len(titles))

    if not use_categories:
        return titles[:limit]

    for category in CATEGORIES:
        if len(titles) >= limit:
            break
        try:
            members = await client.category_members(category, lang=lang, limit=100)
        except Exception:  # noqa: BLE001 - a missing category must not stop ingest
            logger.warning("Could not read %s", category)
            continue

        logger.info("%s -> %d candidate articles", category, len(members))
        for title in members:
            if title in seen:
                continue
            seen.add(title)
            titles.append(title)
            if len(titles) >= limit:
                break

    return titles[:limit]


async def index_articles(
    articles: list[WikiArticle],
    embedder: Embedder,
    store: VectorStore,
) -> int:
    """Chunk, embed and upsert one wave of articles. Returns chunks written."""
    chunks: list[Chunk] = []
    for article in articles:
        article_chunks = chunk_article(article)
        if not article_chunks:
            logger.debug("No usable chunks for %r", article.title)
            continue
        chunks.extend(article_chunks)

    if not chunks:
        return 0

    embeddings = await embedder.embed_documents([c.text for c in chunks])
    store.add(chunks, embeddings)
    return len(chunks)


async def ingest_language(
    client: WikipediaClient,
    embedder: Embedder,
    store: VectorStore,
    titles: list[str],
    lang: str,
    force: bool,
) -> tuple[int, int]:
    """Returns (articles indexed, chunks written)."""
    known_ids = set() if force else store.indexed_page_ids(lang)
    total_articles = 0
    total_chunks = 0

    for start in range(0, len(titles), FETCH_BATCH):
        batch = titles[start : start + FETCH_BATCH]
        articles = await client.fetch_articles(batch, lang=lang)

        fresh = [a for a in articles if a.page_id not in known_ids]
        skipped = len(articles) - len(fresh)
        if not fresh:
            logger.info(
                "[%s] %d-%d: nothing new (%d already indexed)",
                lang, start + 1, start + len(batch), skipped,
            )
            continue

        written = await index_articles(fresh, embedder, store)
        known_ids.update(a.page_id for a in fresh)
        total_articles += len(fresh)
        total_chunks += written

        logger.info(
            "[%s] %d-%d: +%d articles, +%d chunks (skipped %d indexed, %d unusable)",
            lang, start + 1, start + len(batch), len(fresh), written, skipped,
            len(batch) - len(articles),
        )

    return total_articles, total_chunks


async def mirror_titles(
    client: WikipediaClient, titles: list[str], source_lang: str, target_lang: str
) -> list[str]:
    """Translate article titles into another language using ``langlinks``."""
    translated: list[str] = []
    for start in range(0, len(titles), FETCH_BATCH):
        batch = titles[start : start + FETCH_BATCH]
        articles = await client.fetch_articles(batch, lang=source_lang)
        translated.extend(
            a.langlinks[target_lang] for a in articles if target_lang in a.langlinks
        )
    logger.info(
        "Mapped %d/%d titles from %s to %s", len(translated), len(titles), source_lang, target_lang
    )
    return translated


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index Wikipedia into ChromaDB")
    parser.add_argument("--limit", type=int, default=300, help="max articles per language")
    parser.add_argument("--lang", default="en", help="Wikipedia language edition")
    parser.add_argument(
        "--mirror-langs",
        default="",
        help="comma-separated languages to also index, mapped via langlinks (e.g. es,fr)",
    )
    parser.add_argument("--titles", default="", help="comma-separated titles instead of seeds")
    parser.add_argument("--no-categories", action="store_true", help="skip category walk")
    parser.add_argument("--no-vital", action="store_true", help="skip the Vital articles lists")
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    parser.add_argument("--force", action="store_true", help="re-index already-indexed pages")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.openai_configured:
        logger.error(
            "OPENAI_API_KEY is missing or still the placeholder. "
            "Copy .env.example to .env and set a real key."
        )
        return 1

    started = time.perf_counter()
    store = VectorStore(settings)
    if args.reset:
        logger.warning("Resetting collection %r", settings.chroma_collection)
        store.reset()

    embedder = Embedder(settings)

    try:
        articles, chunks = await _run(args, settings, store, embedder)
    except AuthenticationError:
        logger.error("OpenAI rejected the API key in .env. Check OPENAI_API_KEY.")
        return 1

    elapsed = time.perf_counter() - started
    logger.info(
        "Done in %.1fs: %d articles, %d chunks. Index now holds %d chunks across %s.",
        elapsed, articles, chunks, store.count(), store.indexed_languages(),
    )
    return 0


async def _run(
    args: argparse.Namespace,
    settings: Settings,
    store: VectorStore,
    embedder: Embedder,
) -> tuple[int, int]:
    async with WikipediaClient(settings) as client:
        if args.titles:
            titles = [t.strip() for t in args.titles.split(",") if t.strip()]
        else:
            titles = await collect_titles(
                client,
                args.lang,
                args.limit,
                use_categories=not args.no_categories,
                use_vital=not args.no_vital,
            )
        logger.info("Collected %d titles for %s.wikipedia.org", len(titles), args.lang)

        articles, chunks = await ingest_language(
            client, embedder, store, titles, args.lang, args.force
        )

        for target in [l.strip() for l in args.mirror_langs.split(",") if l.strip()]:
            logger.info("--- mirroring into %s ---", target)
            target_titles = await mirror_titles(client, titles, args.lang, target)
            more_articles, more_chunks = await ingest_language(
                client, embedder, store, target_titles, target, args.force
            )
            articles += more_articles
            chunks += more_chunks

    return articles, chunks


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
