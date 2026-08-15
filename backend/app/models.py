"""Request/response schemas for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    # Both spellings accepted: `session_id` per the API spec, `conversation_id`
    # for the existing frontend and any client already using it.
    session_id: str | None = None
    conversation_id: str | None = None
    lang: str = Field(default="en", min_length=2, max_length=12)

    @property
    def session_key(self) -> str | None:
        return self.session_id or self.conversation_id


class Source(BaseModel):
    """One cited chunk, deep-linked to the section it came from."""

    index: int = Field(description="Citation marker used in the answer, e.g. 1 for [1]")
    title: str
    section: str
    url: str = Field(description="Deep link including the #Section anchor")
    article_url: str
    lang: str
    snippet: str
    score: float


class ArticleLink(BaseModel):
    """Smart routing: an article the user should read next."""

    title: str
    url: str
    lang: str
    summary: str = ""


class DisambiguationOptionModel(BaseModel):
    title: str
    url: str


class Timings(BaseModel):
    rewrite_ms: int = 0
    retrieval_ms: int = 0
    generation_ms: int = 0
    total_ms: int = 0


class ToolCallTrace(BaseModel):
    tool: str
    query: str = ""
    hop: int = 0


class AgentTrace(BaseModel):
    """What the agent decided, exposed so a turn's plan can be audited.

    The latency argument for this pipeline rests on two claims that are
    invisible from the outside: that most turns take one hop, and that
    speculative retrieval is usually reused rather than thrown away. Both are
    reported here for the same reason `timings` is — a claim nothing measures
    is indistinguishable from one that stopped being true.
    """

    tools: list[ToolCallTrace] = []
    hops: int = 0
    speculation_hit: bool = False


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[Source] = []
    articles: list[ArticleLink] = []
    disambiguation: list[DisambiguationOptionModel] = []
    disambiguation_term: str | None = None
    lang: str = "en"
    resolved_query: str | None = Field(
        default=None,
        description="The standalone query used for retrieval after history rewriting",
    )
    cached: bool = False
    used_live_search: bool = False
    timings: Timings = Timings()
    agent: AgentTrace = AgentTrace()


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=200)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=200)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=200)
    new_password: str = Field(min_length=1, max_length=200)


class SessionResponse(BaseModel):
    account_id: str
    email: str
    role: str = "user"
    shared_chunks: int
    personal_chunks: int
    personal_articles: int


class LoginResponse(SessionResponse):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class AddArticlesRequest(BaseModel):
    titles: list[str] = Field(min_length=1, max_length=5)
    lang: str = Field(default="en", min_length=2, max_length=12)


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    lang: str
    updated_at: float


class IngestRequest(BaseModel):
    topics: list[str] = Field(min_length=1, max_length=10)
    articles_per_topic: int = Field(default=25, ge=1, le=200)
    lang: str = Field(default="en", min_length=2, max_length=12)


class KnowledgeBaseResponse(BaseModel):
    shared_chunks: int
    personal_chunks: int
    personal_articles: int
    articles_by_language: dict[str, int] = {}
    titles: list[str] = []
    suggested_topics: list[dict[str, str]] = []
    active_job: dict | None = None
    recent_jobs: list[dict] = []


class StatsResponse(BaseModel):
    total_chunks: int
    languages: dict[str, int]
    collection: str
    embedding_model: str
    chat_model: str
    cache: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    openai_configured: bool
