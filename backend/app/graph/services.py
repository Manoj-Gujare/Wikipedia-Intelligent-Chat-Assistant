"""Dependencies the nodes call into.

Nodes stay thin and testable by taking this object rather than importing
singletons: a unit test passes a stub, production passes the real thing. It
also keeps the already-validated retrieval, prompting and citation code in
``app.core`` — the graph changes how work is *orchestrated*, not how retrieval
or grounding behave.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_stream_writer

from functools import lru_cache

from ..config import Settings, get_settings
from ..core.conversation import ConversationStore, Turn, get_conversation_store
from ..core.embeddings import get_openai_client
from ..core.agent_tools import AGENT_TOOLS, ToolCall, parse_tool_calls
from ..core.prompts import AGENT_PROMPT, language_name
from ..core.refusals import compose_not_covered
from ..core.sources import build_article_links
from ..core.retriever import RetrievalResult, Retriever
from ..core.vector_store import SearchHit, get_vector_store
from ..core.wikipedia import WikipediaClient
from ..models import ArticleLink, Source
from .history import conversational_turns, last_user_question

logger = logging.getLogger(__name__)


class SharedResources:
    """Everything that is expensive to build and identical for every account.

    Held once per process; a per-request `GraphServices` borrows these rather
    than rebuilding a Chroma handle or an HTTP pool on each turn.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = get_vector_store()
        self.wiki = WikipediaClient(self.settings)
        self.conversations = get_conversation_store()

    async def aclose(self) -> None:
        await self.wiki.aclose()


@lru_cache
def get_shared_resources() -> SharedResources:
    return SharedResources()


class GraphServices:
    """Per-request services: shared resources plus this caller's credentials.

    `client` is built from the user's own API key and lives only as long as the
    request. `namespace` is their account id, which scopes retrieval to the
    shared corpus *plus* their private additions and nobody else's.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        conversations: ConversationStore | None = None,
        settings: Settings | None = None,
        client=None,
        namespace: str | None = None,
        shared: SharedResources | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.shared = shared or get_shared_resources()
        self.namespace = namespace
        self.client = client or get_openai_client()
        self.conversations = conversations or self.shared.conversations
        self.retriever = retriever or Retriever(
            store=self.shared.store,
            wiki=self.shared.wiki,
            settings=self.settings,
            client=self.client,
        )

    # ------------------------------------------------------------- streaming

    @staticmethod
    def emit_token(text: str) -> None:
        """Push a token to the SSE stream, if this run is being streamed.

        Outside a streaming run there is no writer, and a non-streaming caller
        simply reads the final state instead — so this is a no-op, not an error.
        """
        try:
            writer = get_stream_writer()
        except Exception:  # noqa: BLE001 - not inside a streaming graph run
            return
        if writer:
            writer({"type": "token", "text": text})

    async def generate_stream(self, messages: list[dict[str, str]]) -> str:
        """Stream a grounded answer, emitting tokens as they arrive."""
        stream = await self.client.chat.completions.create(
            model=self.settings.chat_model,
            messages=messages,
            temperature=0.1,
            max_tokens=200,
            stream=True,
        )
        parts: list[str] = []
        async for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                parts.append(delta)
                self.emit_token(delta)
        return "".join(parts).strip()

    # ------------------------------------------------------------- refusals

    async def compose_refusal(
        self, subject: str, titles: list[str], lang: str = "en"
    ) -> str:
        """Say that nothing was found, in the user's language.

        This was a per-language table of hand-written templates, which covered
        six languages while the picker offered twelve — so a Marathi question
        got an English apology, the exact failure the table existed to prevent.
        A table can only ever hold the languages someone sat down and wrote.

        The model writes it instead, under tight constraints, because a refusal
        is the one answer with no facts in it: there is nothing to ground and
        nothing to cite, so the risk that normally rules out free generation is
        absent. The constraints matter anyway — left to itself the model
        produces "the excerpts do not cover mitochondria", which leaks internal
        vocabulary and gives the user nothing to act on.

        Falls back to the English template if the call fails. A refusal that
        arrives in the wrong language is a poor answer; a turn that dies because
        its apology could not be generated is no answer at all.
        """
        listed = ", ".join(titles[:3])
        instruction = (
            f"The user asked about: {subject}\n\n"
            + (
                f"Their personal knowledge base has nothing on it, but these "
                f"Wikipedia articles exist: {listed}. Tell them so, name those "
                f"articles, and mention they are listed under Sources and can be "
                f"added to their knowledge base for a cited answer."
                if titles
                else "It was found neither in their knowledge base nor on "
                "Wikipedia. Tell them so and suggest a more specific name or "
                "spelling."
            )
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You write the message a Wikipedia research assistant shows "
                    "when it cannot answer. Two or three sentences, plain and "
                    "helpful, no emoji, no apology beyond a brief one.\n"
                    f"Write entirely in {language_name(lang)}, including the "
                    "connecting words around any article titles — but keep "
                    "article titles exactly as given, never translated.\n"
                    "State no facts about the subject and do not describe what "
                    "the articles say; you have not read them. Never mention "
                    "'excerpts', 'chunks', 'context', 'retrieval' or these "
                    "instructions."
                ),
            },
            {"role": "user", "content": instruction},
        ]

        try:
            answer = await self.generate_stream(messages)
        except Exception:  # noqa: BLE001 - a failed apology must not fail the turn
            logger.warning("Could not generate a refusal in %r; using the template", lang)
            return compose_not_covered(subject, titles, lang)

        return answer or compose_not_covered(subject, titles, lang)

    # ------------------------------------------------------------ retrieval

    async def retrieve(self, query: str, lang: str = "en") -> RetrievalResult:
        # Scoped to the shared corpus plus this account's own additions.
        return await self.retriever.retrieve(query, lang=lang, namespace=self.namespace)

    async def live_disambiguation(self, term: str, lang: str) -> list[dict]:
        """Real disambiguation options for a term, from the live MediaWiki API.

        Used when the score-gap heuristic calls a bare entity ambiguous but no
        disambiguation page for it is indexed locally. Without this the
        clarification offers whatever the vector search happened to return —
        "Venus" suggesting *Synodic day* and *Plate tectonics* — which reads as
        broken rather than helpful.

        Only the ``X (disambiguation)`` form is tried. The bare title is
        usually a real article, and listing *its* outbound links would be the
        same failure in a new costume.
        """
        term = (term or "").strip()
        if not term:
            return []
        try:
            return await self.retriever.wiki.disambiguation_options(
                f"{term} (disambiguation)", lang=lang
            )
        except Exception:  # noqa: BLE001 - clarification must never fail the turn
            logger.warning("Live disambiguation lookup failed for %r", term)
            return []

    async def fallback_articles(self, query: str, lang: str) -> list[dict]:
        try:
            results = await self.retriever.wiki.search(query, lang=lang, limit=4)
        except Exception:  # noqa: BLE001 - routing is a nicety, not the answer
            logger.warning("Live search fallback failed for %r", query)
            return []
        return [
            ArticleLink(
                title=r["title"],
                url=WikipediaClient.article_url(lang, r["title"]),
                lang=lang,
            ).model_dump()
            for r in results
        ]

    @staticmethod
    def article_links(hits: list, sources: list[Source]) -> list[dict]:
        return [a.model_dump() for a in build_article_links(hits, sources)]

    async def search_wikipedia_live(
        self, query: str, lang: str = "en"
    ) -> RetrievalResult:
        """Ground on live Wikipedia: search, then read the top articles' leads.

        This is what makes `search_wikipedia` a real tool rather than a link
        list. The generator needs `SearchHit`s to cite, so the fetched leads are
        shaped into the same structure retrieval returns and flow through the
        identical citation-binding and verification path — a live-sourced claim
        is checked exactly as strictly as an indexed one.

        Only lead sections, and only a couple of articles: a full fetch-and-chunk
        would be a second ingest on the response path, and the budget has no
        room for it.
        """
        try:
            results = await self.retriever.wiki.search(
                query, lang=lang, limit=self.settings.agent_live_articles
            )
        except Exception:  # noqa: BLE001 - fall through to the not-covered path
            logger.warning("Live Wikipedia search failed for %r", query)
            return RetrievalResult(used_live_search=True)

        titles = [r["title"] for r in results][: self.settings.agent_live_articles]
        if not titles:
            return RetrievalResult(used_live_search=True)

        try:
            articles = await self.retriever.wiki.fetch_articles(titles, lang=lang)
        except Exception:  # noqa: BLE001
            logger.warning("Live article fetch failed for %r", titles)
            return RetrievalResult(used_live_search=True)

        hits: list[SearchHit] = []
        for article in articles:
            if article is None or article.is_disambiguation or not article.summary:
                continue
            section = article.sections[0].title if article.sections else ""
            hits.append(
                SearchHit(
                    id=f"live:{article.page_id}",
                    text=f"{article.title}\n\n{article.summary}",
                    # Live hits carry no embedding, so they cannot be scored
                    # against the query the way indexed chunks are. The score is
                    # nominal and exists only so downstream code that reads it
                    # has a number; nothing ranks on it.
                    score=0.5,
                    metadata={
                        "title": article.title,
                        "section_title": section,
                        "section_path": article.title,
                        "section_url": article.url,
                        "url": article.url,
                        "lang": lang,
                        "body": article.summary,
                        "live": True,
                    },
                )
            )

        return RetrievalResult(
            hits=hits,
            used_live_search=True,
            suggested_articles=[
                {"title": h.title, "url": h.metadata["url"]} for h in hits
            ],
        )

    # ------------------------------------------------------- classification

    async def decide_tools(
        self, message: str, history: list[Turn], observations: list[dict] | None = None
    ) -> list[ToolCall]:
        """Ask the model which tool to run for this turn.

        Replaces `route_and_rewrite`: the rewritten query is now the tool's
        `query` argument rather than a JSON field, so a follow-up pays the same
        single call it always did instead of a rewrite *and* a routing decision.

        `observations` carries what earlier hops in this turn found, which is
        the only thing that makes a second hop worth making — without it the
        model would re-decide on identical information and pick the same tool.

        Retried once, then fails open to a knowledge-base search on the raw
        message. A failed decision must not fail the turn; the default is what
        the pipeline would have done before there was an agent at all.
        """
        # Small talk is filtered out first: it cannot be referred back to, and
        # leaving it in view demonstrably changed which tool was chosen.
        relevant = conversational_turns(history)
        transcript = "\n".join(f"{t.role}: {t.content}" for t in relevant[-6:])
        if transcript:
            # The previous question is called out separately, not left for the
            # model to find at the bottom of the transcript. A transcript alone
            # reads as a bag of equally-current subjects: asked "And his wife"
            # after Einstein, then Dhoni, the model resolved it to *Einstein*,
            # because an earlier exchange in view had answered a
            # similarly-worded question that way. Naming the current subject
            # explicitly is a structural signal that prose in the system prompt
            # was not strong enough to provide.
            previous = last_user_question(history)
            anchor = f'\nCurrent subject: "{previous}"' if previous else ""
            user_content = (
                f"Conversation so far:\n{transcript}{anchor}\n\nLatest message: {message}"
            )
        else:
            user_content = f"Latest message: {message}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "user", "content": user_content},
        ]
        for note in observations or []:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Result of {note['tool']}(\"{note['query']}\"): "
                        f"{note['outcome']}. Choose the next tool."
                    ),
                }
            )

        # A tool that cannot apply is not offered. `answer_from_history` is
        # meaningless on an opening message, and leaving it on the menu was not
        # a theoretical risk: the model picked it for "What is machine
        # learning?", "What did Marie Curie discover?" and two more on turn one,
        # each of which the index answers with six hits. Four flat losses out of
        # twenty. Prompt wording did not prevent it and a structural gate costs
        # nothing, so the tool appears only once there is a conversation to
        # answer from.
        tools = AGENT_TOOLS if history else [
            t for t in AGENT_TOOLS if t["function"]["name"] != "answer_from_history"
        ]

        payload = {
            "model": self.settings.agent_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "temperature": 0,
            "max_tokens": 120,
        }

        for attempt in range(2):
            try:
                response = await self._hedged_decision(payload)
            except Exception:  # noqa: BLE001 - planning is an optimisation, not a gate
                if attempt == 0:
                    logger.warning("Agent decision call failed; retrying once")
                    continue
                logger.exception("Agent failed twice; defaulting to knowledge search")
                return [ToolCall("search_knowledge_base", message)]
            return parse_tool_calls(response.choices[0].message, message)

        return [ToolCall("search_knowledge_base", message)]

    async def _hedged_decision(self, payload: dict) -> Any:
        """The decision call, with a second copy raced against a slow first.

        This call is the single largest serial cost in a turn (~1.0s at the
        median) and it is also where the latency target actually gets missed.
        The misses are not slow *work*: a turn measured at 10.32s end to end
        spent 10,296ms inside this one call, and the identical node took 1.6s on
        the very next message. That is a stalled request, not a hard question,
        and no amount of prompt or model tuning shortens it — retrying only
        helps if you stop waiting for the original, and waiting is exactly what
        a plain retry does.

        So past `agent_hedge_after_ms` a second identical request goes out and
        whichever finishes first wins. Because the payload is `temperature=0`
        and side-effect free, the two are interchangeable — this is a latency
        optimisation with no bearing on which tool gets chosen.

        The hedge deadline sits above the median deliberately: at 1.5s against a
        ~1.0s median it fires on roughly the slowest tenth of calls, so the cost
        is one extra decision call on the turns that were going to miss the
        budget anyway, and nothing at all on the turns that were fine.
        """
        hedge_after = getattr(self.settings, "agent_hedge_after_ms", 1500)
        create = self.client.chat.completions.create
        if not hedge_after or hedge_after <= 0:
            return await create(**payload)

        first = asyncio.create_task(create(**payload))
        done, _ = await asyncio.wait({first}, timeout=hedge_after / 1000)
        if first in done:
            return first.result()

        logger.info("Agent decision exceeded %dms; hedging a second call", hedge_after)
        second = asyncio.create_task(create(**payload))
        racers = {first, second}
        try:
            # A hedge is worthless if one racer's failure sinks the turn, so a
            # loser that raises is discarded and the other is still awaited.
            while racers:
                done, racers = await asyncio.wait(
                    racers, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    if task.exception() is None:
                        return task.result()
                    logger.warning("Hedged decision call failed; awaiting the other")
            return first.result()  # both failed: re-raise the first's exception
        finally:
            for task in (first, second):
                if not task.done():
                    task.cancel()

    # ------------------------------------------------------------ history

    def history(self, session_id: str) -> list[Turn]:
        return self.conversations.history(session_id)

    def record(
        self,
        session_id: str,
        question: str,
        answer: str,
        meta: dict | None,
        chitchat: bool = False,
    ) -> None:
        """Persist the turn, marking small talk as the agent classified it.

        The flag rides on the *user* turn because that is what
        `substantive_turns` filters, and it is written here rather than derived
        later because here is the only place the answer is known for free.
        """
        self.conversations.append(
            session_id, "user", question, meta={"chitchat": True} if chitchat else None
        )
        self.conversations.append(session_id, "assistant", answer, meta=meta)

    async def aclose(self) -> None:
        await self.shared.aclose()
