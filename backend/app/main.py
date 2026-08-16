"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routers import router
from .config import get_settings

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .core.vector_store import get_vector_store

    from .graph.build import get_graph, get_services

    try:
        count = get_vector_store().count()
        if count == 0:
            logger.warning(
                "Vector index is empty. Run: python -m scripts.ingest --limit 300"
            )
        else:
            logger.info("Vector index ready with %d chunks", count)
    except Exception:  # noqa: BLE001
        logger.exception("Could not open the vector store")

    if settings.openai_configured:
        get_graph()  # compile once at startup, not on the first request
        await _warmup(get_services())
        logger.info("Graph compiled; embeddings and search warmed up")

    yield

    await get_services().aclose()


async def _warmup(services) -> None:
    """Pay the one-off costs at startup instead of on a user's first request.

    That means TLS negotiation to OpenAI and to Wikipedia, plus Chroma's lazy
    index load.

    The MediaWiki leg matters more than its one line suggests. Only one kind of
    turn reaches the live API — a question the corpus cannot answer — so that
    turn used to pay a connection handshake no other turn pays, and it is
    already the most expensive path in the app. Measured on the benchmark's
    not-covered case, a cold client cost 4.82s against 2.38-2.56s warm: the
    difference between missing the 3s budget and meeting it, on the one case
    that was missing it every single run.
    """
    try:
        vector = await services.retriever.embedder.embed_query("warmup")
        for lang in services.retriever.store.indexed_languages() or {"en": 0}:
            await services.retriever.store.query(vector, top_k=1, lang=lang)
            await services.retriever.store.query(
                vector, top_k=1, lang=lang, section_title="Introduction"
            )
    except Exception:  # noqa: BLE001 - warmup is best effort
        logger.warning("Warmup failed; the first request will be slower")

    try:
        await services.retriever.wiki.search("Wikipedia", lang="en", limit=1)
    except Exception:  # noqa: BLE001 - Wikipedia being unreachable is not fatal
        logger.warning("Could not reach Wikipedia at startup; live search may be slow")


app = FastAPI(
    title="Wikipedia Intelligent Chat Assistant",
    description=(
        "RAG over Wikipedia content fetched through the official MediaWiki API. "
        "Answers are grounded in indexed article text and cite the exact sections "
        "they came from."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
# The specification names the endpoints `/chat` and `/health`; the frontend and
# existing clients use the `/api` prefix. Both mount the same router.
app.include_router(router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "Wikipedia Intelligent Chat Assistant",
        "docs": "/docs",
        "health": "/api/health",
    }
