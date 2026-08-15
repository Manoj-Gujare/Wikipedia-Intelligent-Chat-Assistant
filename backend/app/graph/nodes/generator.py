"""Node 7: grounded generation with citations, streamed token by token."""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from ...core.citations import verify_citations
from ...core.prompts import SYSTEM_PROMPT, build_context_block, language_name
from ...core.sources import build_sources
from ..references import _is_referential
from ..state import ChatState
from .timing import timed

logger = logging.getLogger(__name__)

def _question_for_generator(state: ChatState) -> str:
    """What to put after "Question:" — the user's words wherever possible.

    The agent's query is a *retrieval* artifact: keywords tuned to embed well,
    not a question anyone asked. Handing it to the generator changes the shape
    of the answer, and one of those shapes is actively harmful. Asked "Mona
    Lisa painter", the model replies "Leonardo da Vinci **is the** painter of
    the Mona Lisa" — a copular claim that citation verification drops as an
    unsupported category assertion, which empties the answer and sends a
    perfectly grounded turn down the not-covered path. Asked "Who painted the
    Mona Lisa?", it replies "Leonardo da Vinci painted the Mona Lisa", which
    passes. Same chunks, same model; only the question shape differed.

    The exception is a message that cannot stand alone. "What about his wife?"
    needs the resolved form, and that is exactly the case the agent's rewrite
    exists for.
    """
    raw = state["question"]
    resolved = state.get("standalone_query")
    if resolved and _is_referential(raw):
        return resolved
    return raw


def _search_query(state: ChatState) -> str:
    """What to *search* for, as opposed to what to phrase the answer around.

    The agent's rewritten query, which is the string retrieval actually ran. The
    generator's other question — the user's own wording — is the right input to
    a prompt and the wrong input to a search engine: measured against the live
    API, "can you tell me about photosynthesis please" returns *Robert Anton
    Wilson* and *Isaac Asimov*, while "photosynthesis" returns *Photosynthesis*.

    English hides this, because its Wikipedia search always returns something
    and the something looks plausible. Smaller editions do not: the same mistake
    in Marathi returned nothing at all for प्रकाशसंश्लेषण, and the user was told
    their subject was on neither Wikipedia nor their own knowledge base.
    """
    return state.get("tool_query") or state.get("standalone_query") or state["question"]


@timed("generator")
async def generator(state: ChatState, services) -> dict:
    """Grounded generation with citations, streamed token by token."""
    query = _question_for_generator(state)
    lang = state.get("lang", "en")
    hits = state.get("hits") or []
    history = services.history(state["session_id"])

    if not hits:
        # No evidence at all: point at whatever live search can find. Searched
        # and named by the agent's query, not the user's sentence — see
        # `_search_query`. The model is still not asked to *answer*, only to
        # word the refusal in the user's language.
        search = _search_query(state)
        articles = state.get("articles") or await services.fallback_articles(search, lang)
        answer = await services.compose_refusal(
            search, [a["title"] for a in articles], lang
        )
        return {
            "answer": answer,
            "sources": [],
            "articles": articles,
            "used_live_search": True,
            "messages": [AIMessage(content=answer)],
        }

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(language_name=language_name(lang))}
    ]
    for turn in history[-6:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Wikipedia excerpts:\n\n{build_context_block(hits)}\n\n"
                f"---\n\nQuestion: {query}"
            ),
        }
    )

    answer = await services.generate_stream(messages)

    # Before binding citations: a sentence whose cited chunk does not support it
    # is worse than no answer, because the citation makes it look checked. Live
    # hits go through this identically — a claim sourced from the live API gets
    # no more benefit of the doubt than an indexed one.
    answer, rejected = verify_citations(answer, hits)
    for sentence, reason in rejected:
        logger.warning("Dropped unsupported sentence (%s): %r", reason, sentence)

    sources = build_sources(hits, answer)

    if sources:
        articles = services.article_links(hits, sources)
        used_live = bool(state.get("used_live_search"))
    else:
        # No citations means the model refused: the retrieved chunks were noise,
        # so route from live search rather than linking them — and replace the
        # model's own wording, which tends to be an unhelpful dead end.
        search = _search_query(state)
        articles = await services.fallback_articles(search, lang)
        answer = await services.compose_refusal(
            search, [a["title"] for a in articles], lang
        )
        used_live = True

    return {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "articles": articles,
        "used_live_search": used_live,
        "messages": [AIMessage(content=answer)],
    }
