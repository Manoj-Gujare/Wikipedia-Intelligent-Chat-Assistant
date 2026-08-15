"""The graph's nodes.

Seven nodes. The model chooses *whether evidence is wanted at all* and then
where it comes from — indexed corpus, live Wikipedia, or the conversation
itself. Two things stay in Python, because handing them to the model costs
latency it cannot earn back:

* retrieval on a plausible query starts *before* the agent has decided, racing
  the decision call instead of queueing behind it,
* a second hop is authorised by the wall clock, not by the agent's enthusiasm.

Classifying the message is not on that list, and used to be. A phrase table
answered greetings for free and could only ever recognise the phrasings
someone had written down; "Do you know me?" was on no list, so it was searched
for in an encyclopedia. Deciding what a message *is* turns out to be the one
job worth a round trip, and speculation is what pays for it.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import re
import time
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage

from ..core.rag import (
    SYSTEM_PROMPT,
    build_context_block,
    build_sources,
    language_name,
    verify_citations,
)
from ..core.wikipedia import WikipediaClient
from .services import substantive_turns
from .state import ChatState

logger = logging.getLogger(__name__)

# Ambiguity by score proximity is only considered for bare entity mentions
# ("Mercury", "Java"). A real question legitimately spans several articles, so
# applying this to long queries would fire constantly on healthy retrievals.
AMBIGUITY_MAX_WORDS = 3
AMBIGUITY_SCORE_GAP = 0.02
# The top hit must actually be a good match before near-ties mean anything.
# On a miss every weak hit scores about the same, which is indistinguishable
# from ambiguity by gap alone — that produced "MS Dhoni could refer to India,
# Mahatma Gandhi, Hindi cinema". Nothing matching is a miss, not a choice.
AMBIGUITY_MIN_SCORE = 0.45

KB_SEARCH = "search_knowledge_base"
LIVE_SEARCH = "search_wikipedia"
HISTORY_ANSWER = "answer_from_history"
DIRECT_REPLY = "respond_directly"


def timed(name: str):
    """Record per-node wall time into state and the log.

    Node-level timing is what proves the fast path is real: a chitchat turn
    should show one node and no retrieval, and any regression shows up as a
    node that suddenly appears where it should not.
    """

    def decorator(fn: Callable[..., Awaitable[dict]]):
        @functools.wraps(fn)
        async def wrapper(state: ChatState, *args: Any) -> dict:
            started = time.perf_counter()
            result = await fn(state, *args) or {}
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            logger.info("node=%-20s %7.1fms", name, elapsed)
            return {**result, "node_timings": [{"node": name, "ms": elapsed}]}

        return wrapper

    return decorator


# --------------------------------------------------------------------- node 1


@timed("gate")
async def gate(state: ChatState, services) -> dict:
    """Seed the turn's budget. Every message goes to the agent from here.

    This node used to divert small talk with a table of phrases and regexes,
    and to send opening questions straight to the index without a decision
    call. Both were latency wins bought with a classifier that could only
    recognise the messages someone had thought to list. "Do you know me?"
    matched nothing, so it was searched for in a Wikipedia index, and the user
    was told their own question was not covered by the corpus.

    The agent decides now, because the set of messages that need no evidence
    is open-ended and a phrase list is the wrong shape for it. Retrieval is
    what makes that affordable: it runs speculatively *underneath* the
    decision call (see `agent`), so a real question pays the decision and gets
    its search for free, and a greeting pays the decision and throws an
    embedding away.
    """
    # The hop budget is copied into state here because a conditional edge sees
    # only the state — it has no services and no config to read settings from.
    settings = services.settings
    return {
        "intent": "new_question",
        "started_at": time.perf_counter(),
        "hops": 0,
        "hop_deadline_ms": getattr(settings, "agent_hop_deadline_ms", 1200),
        "max_hops": getattr(settings, "agent_max_hops", 2),
    }


# --------------------------------------------------------------------- node 2


@timed("direct_responder")
async def direct_responder(state: ChatState, services) -> dict:
    """Emit the reply the agent already wrote. No retrieval, no second call.

    The wording arrives on the tool call rather than being fetched here, which
    is what keeps a greeting to a single round trip: the model decides that no
    evidence is needed and says what to say in the same breath. A separate
    generation call would double the latency of the cheapest turn there is.

    Any speculative search launched under the decision is thrown away — the
    whole point of this branch is that its results were never wanted.
    """
    _discard(state.get("speculative"))
    answer = state.get("direct_reply") or ""

    services.emit_token(answer)
    return {
        "intent": "chitchat",
        "answer": answer,
        "sources": [],
        "articles": [],
        "retrieved_chunks": [],
        "speculative": None,
        "messages": [AIMessage(content=answer)],
    }


# --------------------------------------------------------------------- node 3


@timed("agent")
async def agent(state: ChatState, services) -> dict:
    """Choose the tool for this hop, with retrieval already running underneath.

    The decision call takes about a second; a vector search takes about a third
    of one. Run them in sequence and the search is a second of added latency.
    Run the search speculatively on the raw message and it is usually finished
    and parked before the agent has finished choosing — so when the agent picks
    the knowledge base with substantially the query we guessed, the retrieval is
    free.

    Speculation is only worth it when the raw message is a plausible query.
    "What about his wife?" embeds to noise, so those turns skip it and pay the
    ordinary sequential retrieval — which is still no worse than the rewrite
    call they used to pay.
    """
    message = state["question"]
    history = services.history(state["session_id"])
    hops = state.get("hops", 0)

    speculative = state.get("speculative")
    updates: dict = {}

    # Only ever speculate before the first search. `tool_calls` is checked as
    # well as `hops` because a turn that has already run a search must not
    # launch another for a query we know comes back empty.
    if speculative is None and hops == 0 and not state.get("tool_calls"):
        guess = _speculation_query(message, history, services)
        if guess:
            speculative = asyncio.create_task(
                services.retrieve(guess, lang=state.get("lang", "en"))
            )
            updates["speculative"] = speculative
            updates["speculative_query"] = guess

    calls = await services.decide_tools(message, history, state.get("observations"))
    call = calls[0]

    query = call.query
    if call.name not in (HISTORY_ANSWER, DIRECT_REPLY):
        query = _resolve_against_current_subject(message, query, history)

    logger.info("agent hop=%d tool=%s query=%r", hops, call.name, query)

    if call.name == DIRECT_REPLY:
        updates["direct_reply"] = call.reply

    return {
        **updates,
        "pending_tool": call.name,
        "tool_query": query,
        "standalone_query": query or state.get("standalone_query"),
        "hops": hops + 1,
        "tool_calls": [{"tool": call.name, "query": query, "hop": hops}],
    }


def _resolve_against_current_subject(message: str, query: str, history: list) -> str:
    """Keep a referential follow-up pointed at the subject actually under discussion.

    The agent resolves pronouns well enough most of the time and wrongly often
    enough to matter. Asked "and his wife" after a turn about Einstein *and then*
    a turn about Dhoni, `gpt-4.1-nano` produced all three of these across runs:
    "MS Dhoni wife" (right), "his wife" (nothing resolved), and "Albert Einstein
    wife" (confidently the wrong person). The last is the dangerous one — it
    retrieves real chunks and cites them, so the answer looks grounded while
    being about someone the user stopped asking about two turns ago.

    So the agent's resolution is checked rather than trusted, on the one class of
    message where it can go wrong: a short referential one. The check is whether
    the subject it supplied appears in what the user recently asked about. If the
    agent invents a subject from nowhere in recent history, or resolves nothing
    at all, the previous question is stitched on instead — deterministic, and it
    reproduces the exact string speculation already searched, so being correct
    costs nothing extra.

    A resolution the previous question corroborates is left alone, which is most
    of them: "Who was Marie Curie?" -> "What did she discover?" -> "Marie Curie
    discoveries" shares *Marie Curie* with the question before it and passes
    untouched.

    Corroboration looks at the *immediately* previous question and no further
    back. Widening it to the last two was tried first and defeats the purpose:
    in the very conversation this exists for, Einstein *is* two questions back,
    so a two-question window corroborates the stale subject and waves it
    through. Chains still survive the narrow window, because a resolution built
    on a referential question tends to echo a word from it — "his wife" ->
    "Albert Einstein wife death" shares *wife* — so the corroboration lands
    without ever needing to see the name.
    """
    if not _is_referential(message) or len(re.findall(r"[a-z']+", message.lower())) > 8:
        return query

    previous = _last_user_message(history)
    if not previous:
        return query

    supplied = _terms(query) - _terms(message)
    if supplied and supplied & _terms(previous):
        return query

    reason = "resolved nothing" if not supplied else "named a stale subject"
    logger.info("Agent %s for %r; stitching to %r", reason, message, previous)
    return f"{previous} {message}"


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[\w']+", (text or "").lower()))


def _speculation_query(message: str, history: list, services) -> str | None:
    """What to search while the agent decides, or None to not bother.

    Three cases, and the third is what makes follow-ups fast:

    * A message that already stands alone is its own best guess.
    * A referential one ("what happens at its event horizon?") embeds to noise
      on its own — but stitched to the question before it, it does not. The
      concatenation is a poor *question* and a perfectly good *search*: it
      carries the subject the pronoun refers to, which is the only thing
      retrieval was missing. The agent's own query ("black hole event horizon")
      is then contained in it, so the containment check accepts the parked
      result rather than paying for a second search.
    * Anything else — an elliptical message with no previous question to stitch
      to — is left alone, because there is nothing to build a query from.

    A wasted speculation costs an embedding call the user pays for, so this
    stays conservative; but it is now conservative about *what to search*
    rather than *whether to search at all*, which is where the follow-up
    latency was going.
    """
    if not getattr(services.settings, "agent_speculative_retrieval", True):
        return None
    if not history or looks_self_contained(message):
        return message

    previous = _last_user_message(history)
    if previous:
        return f"{previous} {message}"
    return None


def _last_user_message(history: list) -> str | None:
    """The last question worth stitching to — never a greeting.

    Stitching "Hey" onto "what about his wife?" produces a search for a
    greeting and a pronoun, which is strictly worse than not speculating.
    """
    turns = substantive_turns(history)
    return turns[-1].content if turns else None


def route_after_agent(state: ChatState) -> str:
    tool = state.get("pending_tool")
    if tool == DIRECT_REPLY:
        return "direct_responder"
    if tool == HISTORY_ANSWER:
        return "history_answerer"
    return "tool_executor"


# Words that make a short message meaningless without the turn before it.
_REFERENTIAL = {
    "he", "she", "it", "they", "him", "her", "them", "his", "hers", "its",
    "their", "theirs", "that", "this", "those", "these", "there",
}


# Openers that mean the message continues the previous turn rather than
# starting a new one, even when it contains no pronoun at all.
_ELLIPTICAL_OPENERS = (
    "and ", "but ", "so ", "also ", "then ", "what about", "how about",
    "what else", "anything else", "tell me more", "more about", "why", "ok ",
    "okay ",
)

# A capitalised word is treated as a named subject. Three letters minimum keeps
# acronym-shaped noise out; the search deliberately skips the first word, which
# is capitalised in every sentence and says nothing about the subject.
_PROPER_NOUN = re.compile(r"\b[A-Z][a-z]{2,}")

# Below this, a question is almost certainly leaning on the previous turn
# ("Tell me more", "And the population?").
_MIN_SELF_CONTAINED_WORDS = 4


def looks_self_contained(message: str) -> bool:
    """Whether a message stands on its own without the conversation history.

    Under the agentic graph this no longer decides whether to call the model —
    the agent is called either way — it decides whether *speculating* on the
    raw message is worth an embedding call. The bias stays the same and the
    cost of being wrong drops: a false positive now wastes a cheap parallel
    search instead of retrieving noise, because the agent's own query is what
    actually runs when the two disagree.
    """
    text = message.strip()
    lowered = text.lower()

    if lowered.startswith(_ELLIPTICAL_OPENERS):
        return False
    if any(word in _REFERENTIAL for word in re.findall(r"[a-z']+", lowered)):
        return False

    words = text.split()
    if len(words) < _MIN_SELF_CONTAINED_WORDS:
        return False
    return bool(_PROPER_NOUN.search(" ".join(words[1:])))


# --------------------------------------------------------------------- node 4


@timed("tool_executor")
async def tool_executor(state: ChatState, services) -> dict:
    """Run the tool the agent chose, reusing the speculative search if it fits."""
    tool = state.get("pending_tool") or KB_SEARCH
    query = state.get("tool_query") or state["question"]
    lang = state.get("lang", "en")

    if tool == LIVE_SEARCH:
        _discard(state.get("speculative"))
        result = await services.search_wikipedia_live(query, lang=lang)
        shaped = _shape(result, lang, live=True)
        return {**shaped, "speculative": None, "observations": _observe(tool, query, shaped)}

    result, reused = await _kb_search(state, services, query, lang)
    shaped = _shape(result, lang, live=False)

    # Ambiguity is a property of *ranking*: several indexed articles matching a
    # bare term equally well. It is checked only for knowledge-base results —
    # live hits carry a nominal score with nothing behind it, so every pair of
    # them reads as tied and the gap test would call each live search ambiguous
    # by construction.
    #
    # Tested against the *user's* words, never the agent's query. The word-count
    # gate below encodes "a bare entity mention", which is a property of what
    # someone typed; the agent routinely compresses a full question into two
    # keywords, and scoring those made "What does photosynthesis produce?" ->
    # "photosynthesis products" trip a threshold written for people typing
    # "Mercury". The result was a disambiguation prompt for a question the
    # index answers outright.
    if shaped.get("hits") and _is_ambiguous_by_score(state["question"], result.hits):
        # The index has several equally-good matches for a bare entity but no
        # disambiguation page for it — otherwise `_shape` would have caught it.
        # Ask Wikipedia for the real options before falling back to the nearest
        # indexed topics, which are neighbours in embedding space and not
        # choices a reader would recognise.
        candidates = await services.live_disambiguation(query, lang)
        kind = "page"
        if not candidates:
            candidates = _distinct_articles(result.hits, lang)
            kind = "related"
        logger.info(
            "ambiguous entity %r -> %d candidates (%s)", query, len(candidates), kind
        )
        return {
            "intent": "ambiguous",
            "retrieved_chunks": shaped["retrieved_chunks"],
            "disambiguation_kind": kind,
            "disambiguation_candidates": candidates,
            "speculative": None,
        }

    return {
        **shaped,
        "speculation_hit": reused,
        "speculative": None,
        "observations": _observe(tool, query, shaped),
    }


def _observe(tool: str, query: str, shaped: dict) -> list[dict]:
    """What this hop found, in one line, for the agent's next decision.

    Only recorded on an empty result. A hop that found evidence goes straight
    to the generator and never re-enters the agent, so describing a successful
    search to a model that will not see it again is pure token spend.
    """
    if shaped.get("hits"):
        return []
    return [{"tool": tool, "query": query, "outcome": "no relevant results"}]


def _is_ambiguous_by_score(query: str, hits: list) -> bool:
    """True when a bare entity mention matches unrelated articles equally well.

    Only bare mentions qualify. "What is the periodic table organised by?"
    spanning several articles is healthy retrieval, not ambiguity.
    """
    if len(query.split()) > AMBIGUITY_MAX_WORDS or len(hits) < 2:
        return False

    # Raw similarity, not the ranked score: the lead-section boost is a display
    # preference, and letting it move these numbers made ambiguity detection a
    # side effect of retrieval tuning.
    top, second = hits[0].raw_score, hits[1].raw_score

    # A weak top hit means we found nothing, not that we found several things.
    if top < AMBIGUITY_MIN_SCORE:
        return False

    titles = {h.title for h in hits[:3]}
    if len(titles) < 2:
        return False
    return abs(top - second) < AMBIGUITY_SCORE_GAP


def _distinct_articles(hits: list, lang: str) -> list[dict]:
    seen: set[str] = set()
    candidates: list[dict] = []
    for hit in hits:
        if hit.title in seen:
            continue
        seen.add(hit.title)
        candidates.append(
            {
                "title": hit.title,
                "url": hit.metadata.get("url")
                or WikipediaClient.article_url(lang, hit.title),
            }
        )
    return candidates[:6]


async def _kb_search(state: ChatState, services, query: str, lang: str):
    """Await the parked speculative search, or run a fresh one.

    The parked result is reused only when the agent's query is close enough to
    the message it was launched on. A rewrite that changed the subject —
    "what about his wife?" becoming "Albert Einstein wife" — must not be served
    chunks retrieved for the pronoun; that would be a *worse* failure than the
    latency we are avoiding, because it looks like a successful retrieval.
    """
    task = state.get("speculative")
    if task is None:
        return await services.retrieve(query, lang=lang), False

    parked_query = state.get("speculative_query") or ""
    threshold = getattr(services.settings, "agent_speculation_reuse_threshold", 0.7)
    if _similar(parked_query, query, threshold):
        try:
            result = await task
        except Exception:  # noqa: BLE001 - speculation must never fail the turn
            logger.warning("Speculative retrieval failed; searching again")
            return await services.retrieve(query, lang=lang), False
        logger.info("speculation hit: reused search for %r", parked_query)
        return result, True

    logger.info("speculation miss: %r -> %r", parked_query, query)
    _discard(task)
    return await services.retrieve(query, lang=lang), False


def _discard(task) -> None:
    """Cancel an unused speculative task and swallow whatever it was doing.

    Without this, a cancelled or failed retrieval surfaces as an unretrieved
    task exception on the event loop — a stack trace printed under a turn that
    actually succeeded, which is worse than useless when reading logs.

    The suppression has to reach `BaseException`: `.exception()` on a cancelled
    task *raises* `CancelledError`, and that inherits from `BaseException`, not
    `Exception`. Catching only `Exception` here logged a traceback on every
    speculation miss — the exact noise this function exists to prevent.
    """
    if task is None or not hasattr(task, "cancel"):
        return
    task.cancel()

    def _swallow(finished) -> None:
        with contextlib.suppress(BaseException):
            finished.exception()

    task.add_done_callback(_swallow)


def _similar(speculated: str, chosen: str, threshold: float) -> bool:
    """How much of the agent's query the speculated message already contained.

    Containment, not Jaccard, and the asymmetry is the point. What the agent
    does to a standalone question is *strip* it: "What is the event horizon of
    a black hole?" comes back as "event horizon of a black hole". Those two
    retrieve the same chunks, but symmetric overlap scores them 0.67 — the
    filler words the agent removed count against the match — so a Jaccard gate
    at any sane threshold rejected every real question this was built to
    accelerate. Measured on the smoke suite: 0/3 hits with Jaccard, 2/3 with
    containment.

    Asking instead "is the agent's query already covered by what we searched?"
    scores that pair 1.0, while a genuine rewrite still fails: "what about his
    wife?" -> "Albert Einstein wife" shares only *wife*, or 0.33. Word sets, no
    stopword list, so it behaves the same in every language the index holds.
    """
    parked = set(re.findall(r"[\w']+", speculated.lower()))
    wanted = set(re.findall(r"[\w']+", chosen.lower()))
    if not parked or not wanted:
        return False
    return len(wanted & parked) / len(wanted) >= threshold


def _shape(result, lang: str, live: bool) -> dict:
    """Turn a RetrievalResult into state updates, flagging ambiguity."""
    if result.disambiguation:
        return {
            "intent": "ambiguous",
            "retrieved_chunks": [],
            "disambiguation_kind": "page",
            "disambiguation_candidates": [
                {"title": o.title, "url": o.url} for o in result.disambiguation
            ],
        }

    chunks = [
        {
            "text": hit.metadata.get("body") or hit.text,
            "article_title": hit.title,
            "url": hit.section_url,
            "article_url": hit.metadata.get("url", ""),
            "section": hit.metadata.get("section_title", ""),
            "score": round(hit.score, 4),
        }
        for hit in result.hits
    ]

    if not chunks:
        return {
            "retrieved_chunks": [],
            "hits": [],
            "used_live_search": result.used_live_search or live,
            "articles": [
                {"title": a["title"], "url": a["url"], "lang": lang, "summary": ""}
                for a in result.suggested_articles
            ],
        }

    return {"retrieved_chunks": chunks, "hits": result.hits, "used_live_search": live}


def route_after_tools(state: ChatState) -> str:
    """Clarify, hop again, or answer with what we have.

    The hop gate is wall-clock: a second search plus generation needs roughly
    1.9s, so authorising one at 1.2s lands near 3.1s and authorising one at
    1.6s does not. A hop *count* cannot tell those apart — it would let a slow
    first call spend the rest of the budget it already overran.
    """
    if state.get("intent") == "ambiguous":
        return "clarification"

    if state.get("hits"):
        return "generator"

    deadline = state.get("hop_deadline_ms", 1200)
    max_hops = state.get("max_hops", 2)

    elapsed_ms = (time.perf_counter() - state.get("started_at", 0)) * 1000
    if state.get("hops", 0) < max_hops and elapsed_ms < deadline:
        logger.info(
            "empty retrieval at %.0fms; hopping again (budget %dms)", elapsed_ms, deadline
        )
        return "agent"

    logger.info(
        "empty retrieval at %.0fms; answering without another hop", elapsed_ms
    )
    return "generator"


# --------------------------------------------------------------------- node 5


@timed("clarification")
async def clarification(state: ChatState, services) -> dict:
    """Ask which entity was meant rather than answering the wrong one.

    The wording tracks where the options came from. Real disambiguation options
    can be presented as *the* meanings of the word; nearest indexed topics
    cannot, and claiming otherwise is how "Venus" came to offer *Plate
    tectonics* as one of its senses.
    """
    candidates = state.get("disambiguation_candidates") or []
    titles = ", ".join(c["title"] for c in candidates[:3])
    term = state["question"]

    if not titles:
        answer = (
            f'"{term}" could refer to a few different things on Wikipedia. '
            f"Which one did you mean?"
        )
    elif state.get("disambiguation_kind") == "related":
        answer = (
            f'I don\'t have a disambiguation page for "{term}", and it matches '
            f"several unrelated things in your knowledge base. The closest are "
            f"{titles}. Did you mean one of those, or can you give me a more "
            f"specific name?"
        )
    else:
        answer = (
            f'"{term}" could refer to a few different things on Wikipedia — '
            f"{titles}, and others. Which one did you mean?"
        )

    services.emit_token(answer)
    return {
        "answer": answer,
        "sources": [],
        "articles": [],
        "messages": [AIMessage(content=answer)],
    }


# --------------------------------------------------------------------- node 6


@timed("history_answerer")
async def history_answerer(state: ChatState, services) -> dict:
    """Answer a question about the conversation itself, with no retrieval.

    "What did you just tell me?" has its evidence in the transcript, and
    searching the index for it returns whatever the words happen to sit near.
    This path cites nothing, by construction: the sources belong to the turns
    being summarised, and re-attaching them here would claim the summary was
    independently grounded when it was not.
    """
    _discard(state.get("speculative"))
    history = services.history(state["session_id"])
    lang = state.get("lang", "en")

    if not history:
        # The agent picked this on an opening message, where there is no
        # history to answer from. Fall back rather than emit an empty turn.
        logger.info("answer_from_history with no history; refusing instead")
        answer = await services.compose_refusal(state["question"], [], lang)
        return {
            "answer": answer,
            "sources": [],
            "articles": [],
            "speculative": None,
            "messages": [AIMessage(content=answer)],
        }

    messages = [
        {
            "role": "system",
            "content": (
                "Answer using only this conversation. Do not add facts that are "
                "not already in it, and do not use citation markers. Two "
                f"sentences at most. Write in {language_name(lang)}."
            ),
        }
    ]
    for turn in history[-6:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": state["question"]})

    answer = await services.generate_stream(messages)
    return {
        "answer": answer,
        "sources": [],
        "articles": [],
        "speculative": None,
        "messages": [AIMessage(content=answer)],
    }


# --------------------------------------------------------------------- node 7


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


def _is_referential(message: str) -> bool:
    """Whether the message leans on the turn before it to mean anything."""
    lowered = message.strip().lower()
    if lowered.startswith(_ELLIPTICAL_OPENERS):
        return True
    return any(word in _REFERENTIAL for word in re.findall(r"[a-z']+", lowered))


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
