"""Node 7: grounded generation with citations, streamed token by token."""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from ...core.citations import verify_citations
from ...core.prompts import SYSTEM_PROMPT, build_context_block, language_name
from ...core.sources import build_sources
from ..references import _is_referential
from ..state import ChatState
from .speculation import _discard, collect
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


def build_generation_messages(question: str, lang: str, hits: list, history: list) -> list[dict]:
    """The grounded-answer prompt.

    Extracted so the speculative generation and the real one are built by the
    same code rather than by two implementations that have to be kept in step.
    A speculative answer is only ever flushed when it was produced from this
    function with the same arguments the ordinary path would have passed, which
    is what makes "identical to the non-speculative path" checkable instead of
    merely intended.
    """
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
                f"---\n\nQuestion: {question}"
            ),
        }
    )
    return messages


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
        #
        # Any speculative answer is void here by definition: it was written
        # from chunks this turn has just established it does not have.
        _discard(state.get("speculative_answer"))
        search = _search_query(state)
        parked = state.get("articles")
        if parked:
            # Already routed by an earlier node; only the wording is missing.
            articles = parked
            answer = await services.compose_refusal(
                search, [a["title"] for a in articles], lang
            )
        else:
            # Word the refusal while the article search runs, not after it.
            articles, answer = await services.refuse_and_route(search, lang)
        return {
            "answer": answer,
            "sources": [],
            "articles": articles,
            "used_live_search": True,
            "speculative_answer": None,
            "speculative_answer_used": False,
            "messages": [AIMessage(content=answer)],
        }

    answer, flushed = await _answer_text(state, services, query, lang, hits, history)

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
        # model's own wording, which tends to be an unhelpful dead end. Same
        # concurrency as the no-hits path above; this branch has already paid
        # for a full generation, so it needs the saving more, not less.
        search = _search_query(state)
        articles, answer = await services.refuse_and_route(search, lang)
        used_live = True

    return {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "articles": articles,
        "used_live_search": used_live,
        "speculative_answer": None,
        "speculative_answer_used": flushed,
        "messages": [AIMessage(content=answer)],
    }


async def _answer_text(
    state: ChatState, services, query: str, lang: str, hits: list, history: list
) -> tuple[str, bool]:
    """The answer, flushed from the speculative generation when it is valid.

    Three conditions have to hold before a buffered answer reaches anyone, and
    each rules out a different way of being wrong:

    * `speculation_hit` — the decision confirmed the knowledge base *and* the
      executor served the parked search. Without it the agent went somewhere
      else, or rewrote the query into a different search.
    * `parked.hits is hits` — the answer was written from the very chunk list
      the turn ended up with. Identity, because a reused result is the same
      object; equality would pass for two different searches that happened to
      tie.
    * a non-empty `text` — a partial or empty buffer is never flushed, only
      completed ones.

    The buffer is awaited only once the first condition already holds, so the
    discard path never pays for a generation it has decided not to use: it
    cancels and runs the ordinary call instead.

    Returns the answer and whether it came from the buffer.
    """
    parked_task = state.get("speculative_answer")

    if parked_task is not None:
        if state.get("speculation_hit"):
            parked = await collect(parked_task)
            if parked is not None and parked.hits is hits and parked.text:
                logger.info(
                    "speculative answer flushed (%d chars); the decision call "
                    "and the generation overlapped",
                    len(parked.text),
                )
                # Emitted whole. Nothing went to the client while it was
                # speculative, so this is the first the stream hears of it.
                services.emit_token(parked.text)
                return parked.text, True
            logger.info("speculative answer completed but did not match; discarding")
        else:
            logger.info("speculative answer discarded: the turn did not confirm it")
            _discard(parked_task)

    return await services.generate_stream(
        build_generation_messages(query, lang, hits, history)
    ), False
