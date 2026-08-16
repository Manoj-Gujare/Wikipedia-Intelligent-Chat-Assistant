"""Stub services and state builders shared by the graph tests.

Every dependency is stubbed, so these tests run with no network and no
API key."""

from __future__ import annotations

import asyncio
import pytest
import time

from app.core.agent_tools import ToolCall, parse_tool_calls
from app.core.retriever import RetrievalResult
from app.core.vector_store import SearchHit


class Turn:
    def __init__(self, role: str, content: str, meta: dict | None = None) -> None:
        self.role, self.content, self.meta = role, content, meta


def chitchat_turn(content: str) -> Turn:
    """A user turn the agent classified as needing no evidence.

    Small talk is marked when it is written now, rather than re-derived on
    every history read, so a test that wants "the user said hello" has to say
    which verdict was recorded — exactly as production does.
    """
    return Turn("user", content, meta={"chitchat": True})


class StubSettings:
    agent_hop_deadline_ms = 1200
    agent_max_hops = 2
    agent_speculative_retrieval = True
    agent_speculative_generation = True
    agent_speculation_reuse_threshold = 0.6


class StubServices:
    """Records what each node asked for, so tests can assert on work avoided."""

    def __init__(
        self,
        history=None,
        result=None,
        answer="Grounded answer [1].",
        live_disambiguation=None,
        tool_call=None,
        live_result=None,
    ):
        self._history = history or []
        self._result = result or RetrievalResult()
        self._answer = answer
        # Empty by default: the score-gap path should fall back to related
        # topics unless a test explicitly supplies live options.
        self._live_disambiguation = live_disambiguation or []
        self._live_result = live_result or RetrievalResult(used_live_search=True)
        self.settings = StubSettings()
        self.retrieve_calls: list[str] = []
        self.fallback_calls: list[str] = []
        self.refusal_calls: list[tuple] = []
        self.live_search_calls: list[str] = []
        self.live_disambiguation_calls: list[str] = []
        self.generate_calls = 0
        self.decide_calls: list[list[dict]] = []
        self.tokens: list[str] = []
        # Every prompt `generate_stream` was handed, so a test can assert the
        # speculative and ordinary prompts are byte-identical.
        self.generated_prompts: list[list[dict]] = []
        # A list cycles one decision per hop; a bare ToolCall repeats forever.
        self.tool_call = tool_call or ToolCall("search_knowledge_base", "rewritten query")

    def history(self, session_id):
        return self._history

    def emit_token(self, text):
        self.tokens.append(text)

    async def retrieve(self, query, lang="en"):
        self.retrieve_calls.append(query)
        return self._result

    async def search_wikipedia_live(self, query, lang="en"):
        self.live_search_calls.append(query)
        return self._live_result

    async def live_disambiguation(self, term, lang="en"):
        self.live_disambiguation_calls.append(term)
        return list(self._live_disambiguation)

    async def decide_tools(self, message, history, observations=None):
        self.decide_calls.append(list(observations or []))
        if isinstance(self.tool_call, list):
            index = min(len(self.decide_calls) - 1, len(self.tool_call) - 1)
            return [self.tool_call[index]]
        return [self.tool_call]

    async def generate_stream(self, messages, emit: bool = True):
        self.generate_calls += 1
        # Mirrors production: a buffered generation emits nothing. Tests assert
        # on `tokens` to prove a speculative answer stayed off the wire.
        self.generated_prompts.append(messages)
        if emit:
            self.emit_token(self._answer)
        return self._answer

    async def fallback_articles(self, query, lang):
        self.fallback_calls.append(query)
        return [{"title": "Live result", "url": "https://en.wikipedia.org/wiki/X",
                 "lang": lang, "summary": ""}]

    async def compose_refusal(self, subject, titles, lang="en"):
        # Counted apart from `generate_calls`: wording a refusal is not the same
        # event as generating an answer, and tests assert on both.
        # `titles=None` means the article search is still in flight and this
        # wording is racing it, so it must not name any article.
        self.refusal_calls.append((subject, None if titles is None else list(titles), lang))
        answer = f"[no {subject}] {', '.join(titles or [])}"
        self.emit_token(answer)
        return answer

    async def refuse_and_route(self, subject, lang="en"):
        """Mirrors production: both calls start together, not one after the other."""
        articles, answer = await asyncio.gather(
            self.fallback_articles(subject, lang),
            self.compose_refusal(subject, None, lang),
        )
        if not articles:
            answer = await self.compose_refusal(subject, [], lang)
        return articles, answer

    @staticmethod
    def article_links(hits, sources):
        return [{"title": h.title, "url": "https://en.wikipedia.org/wiki/X",
                 "lang": "en", "summary": ""} for h in hits[:2]]


def _hit(title="Physics", score=0.8, section="Introduction") -> SearchHit:
    return SearchHit(
        id=f"{title}-{section}",
        text=f"{title} > {section}\n\nBody text.",
        score=score,
        metadata={
            "title": title,
            "section_title": section,
            "section_path": f"{title} > {section}",
            "section_url": f"https://en.wikipedia.org/wiki/{title}#{section}",
            "url": f"https://en.wikipedia.org/wiki/{title}",
            "lang": "en",
            "body": "Body text.",
        },
    )


def _state(**overrides):
    base = {
        "question": "who is albert einstein",
        "session_id": "s1",
        "lang": "en",
        "intent": "new_question",
        "messages": [],
        "node_timings": [],
        "hops": 0,
        "started_at": time.perf_counter(),
        "hop_deadline_ms": 1200,
        "max_hops": 2,
    }
    base.update(overrides)
    return base


class _FakeCompletions:
    def __init__(self, sink):
        self.sink = sink

    async def create(self, **payload):
        self.sink.append(payload)
        return _Response()


class _Response:
    def __init__(self):
        self.choices = [type("C", (), {"message": _Msg([])})()]


class _FakeClient:
    def __init__(self, sink):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(sink)})()


def _services_with(sink):
    from app.graph.services import GraphServices

    services = GraphServices.__new__(GraphServices)
    services.settings = StubSettings()
    services.settings.agent_model = "gpt-4.1-nano"
    services.client = _FakeClient(sink)
    return services


class _Fn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _Call:
    def __init__(self, name, arguments):
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, calls):
        self.tool_calls = calls
