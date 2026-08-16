"""Application settings, loaded from environment / .env."""

import logging
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""

    # Measured 2026-08-16 on the 26-case suite (see scripts/evaluate.py):
    #   gpt-4.1-nano  100% accuracy, p50 2.22-2.55s, follow-up p50 2.04-2.67s
    #   gpt-4.1-mini   95% accuracy, p50 2.58s,      follow-up p50 4.85s
    # nano is the default on both counts, which is not what one would guess.
    # mini writes longer answers (52 words at the median against nano's 40) and
    # latency is roughly linear in answer length, so it misses the 3s budget on
    # every follow-up while also scoring lower — the extra words dilute rather
    # than add. Treat the accuracy gap as noise at this suite size; the latency
    # gap is not noise.
    chat_model: str = "gpt-4.1-nano"
    # The agent's tool-decision call. It sits on the critical path of every
    # turn, so it is a small model emitting one tool call — not the model that
    # writes the answer. It also writes small-talk replies directly, which is
    # well within a nano model's range: one or two conversational sentences
    # with no facts in them.
    agent_model: str = "gpt-4.1-nano"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # --- agent budget ---
    # A second retrieval hop is allowed only while the turn still has room for
    # it. The gate is wall-clock, not a hop count, because a hop count lets one
    # slow decision call push the turn past 3s and still authorise more work:
    # 1.2s in, a hop (~0.4s) plus generation (~1.5s) still lands near 3.1s,
    # while at 1.6s in the same hop cannot. Set to 0 to forbid second hops.
    agent_hop_deadline_ms: int = 1200
    # Hard ceiling regardless of the clock, so a pathological loop terminates.
    agent_max_hops: int = 2
    # Race a second decision call once the first has taken this long. The
    # decision call is the biggest serial cost in a turn and the place the 3s
    # budget actually gets missed — not by slow work but by stalled requests
    # (one measured turn spent 10,296ms of its 10,320ms inside this call, while
    # the next message took 1.6s). A plain retry cannot help, because it keeps
    # waiting on the stalled request; hedging stops waiting. Set to 0 to
    # disable. Sits above the ~1.0s median so it fires on the slowest tenth.
    agent_hedge_after_ms: int = 1500
    # Speculative retrieval races the decision call on messages that are
    # already standalone: retrieval is ~340ms against the call's ~1s, so the
    # result is usually parked and waiting by the time the agent asks for it.
    # Disable to trade latency back for the embedding calls it sometimes wastes.
    agent_speculative_retrieval: bool = True
    # Write the answer under the decision call too, not just the search. The
    # two model calls in a turn are only logically serial for the minority of
    # turns that route somewhere other than the knowledge base; for the rest
    # the answer can be written while the decision is still being made, then
    # flushed once it is confirmed. Costs a discarded generation on the turns
    # that route elsewhere, which is the price of not knowing in advance which
    # kind of turn this is. Set to 0/false to go back to strictly serial.
    agent_speculative_generation: bool = True
    # A speculative result is reused only when the message we searched already
    # contained this much of the query the agent asked for. Containment, not
    # symmetric overlap: the agent's usual edit is to strip filler ("What is
    # the event horizon of a black hole?" -> "event horizon of a black hole"),
    # and a symmetric measure counts the removed words against the match.
    # Below this the agent rewrote the query into something materially
    # different and the parked hits would answer a question nobody asked.
    #
    # 0.6 sits in a measured gap rather than on either population's edge.
    # Scored over rewrites that should reuse (filler stripped, a keyword added)
    # and rewrites that must not (a pronoun resolved to a new subject), the
    # first group ran 0.67-1.00 and the second 0.00-0.50 — nothing in between.
    # 0.7 was the first guess and it cut through the safe group, rejecting
    # "Marie Curie discoveries" against "...what did she discover?" purely
    # because the tokeniser does not stem.
    #
    # Do not "fix" that by stemming. It was tried: it reclassifies the one
    # remaining miss on the smoke path from 0.50 to 0.75, and that reuse
    # produces a refusal where the second search produced a cited answer. The
    # slack between the two populations is standing in for retrieval quality,
    # which word overlap cannot measure — see `_similar`.
    agent_speculation_reuse_threshold: float = 0.6
    # Live-search grounding fetches this many articles' lead sections.
    agent_live_articles: int = 2

    chroma_path: str = "./data/chroma"
    chroma_collection: str = "wikipedia"

    retrieval_top_k: int = 24
    retrieval_final_k: int = 6
    mmr_lambda: float = 0.6
    min_relevance_score: float = 0.18
    # Score bonus for lead-section chunks. Leads hold the canonical facts but
    # score modestly against specific queries, losing to deep subsections.
    intro_boost: float = 0.07
    # A larger lead-section boost for short queries was tried here and backed
    # out. "Saturn" answering with the equatorial ridges of its moons is a
    # *coverage* gap, not a ranking one — neither Saturn nor Venus is among the
    # 300 seeded articles, so there is no lead section to promote. All the boost
    # did was lift unrelated articles' leads over the only on-topic passages,
    # which broke ambiguity detection for "Venus" and pushed "Saturn" into a
    # refusal. The fix for those two is indexing them, not reweighting.
    intro_pool_k: int = 6

    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 60

    # Wikimedia's User-Agent policy asks for a descriptive agent with working
    # contact details, so the default carries real ones rather than a
    # placeholder — an unedited checkout is the case most likely to reach
    # production Wikipedia, and it is the one a placeholder fails.
    wiki_user_agent: str = (
        "WikipediaIntelligentChatAssistant/1.0 "
        "(https://github.com/Manoj-Gujare/Wikipedia-Intelligent-Chat-Assistant; "
        "manojgujare726@gmail.com)"
    )
    wiki_requests_per_second: float = 8.0
    wiki_max_concurrency: int = 4
    wiki_maxlag: int = 5

    cors_origins: str = "http://localhost:3000"
    # Signing key for session tokens. Generated per-process when unset, which is
    # fine locally (a restart just forces re-login) but must be set explicitly
    # anywhere running more than one process, or tokens won't verify across them.
    jwt_secret: str = ""
    jwt_ttl_seconds: int = 60 * 60 * 12
    conversations_db: str = "./data/conversations.db"
    conversation_ttl_seconds: int = 3600
    conversation_max_turns: int = 12
    cache_ttl_seconds: int = 900
    log_level: str = "INFO"

    @property
    def openai_configured(self) -> bool:
        """True only for a real key.

        `.env.example` ships a placeholder, and a copied-but-unedited `.env` is
        the most common first-run mistake — treating it as configured turns a
        clear setup error into a confusing 401 deep inside the pipeline.
        """
        key = self.openai_api_key.strip()
        return bool(key) and "your-key" not in key.lower()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def chroma_dir(self) -> Path:
        path = Path(self.chroma_path)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return path

    @property
    def conversations_db_path(self) -> Path:
        path = Path(self.conversations_db)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return path


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if not settings.jwt_secret.strip():
        # Ephemeral secret: never a fixed default, because a hard-coded fallback
        # that ships in the repo lets anyone forge tokens against any install.
        settings.jwt_secret = secrets.token_urlsafe(48)
        logging.getLogger(__name__).warning(
            "JWT_SECRET is not set; generated a temporary one. "
            "Sessions will not survive a restart and cannot be shared across workers."
        )
    return settings
