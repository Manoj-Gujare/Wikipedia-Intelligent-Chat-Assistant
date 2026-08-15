"""Grounding, prompting and citation helpers shared by the graph nodes.

Orchestration lives in ``app.graph`` (a LangGraph StateGraph). This module holds
the parts that decide *what the model sees* and *what it may claim*:

* the system prompt that forbids answering beyond the retrieved excerpts,
* the agent prompt and tool schemas that decide *where evidence comes from*,
* citation binding — mapping ``[n]`` markers back to the chunks they cite, so
  the response only ever lists sources the answer actually used,
* smart-routing links to the underlying Wikipedia articles.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ..models import ArticleLink, ChatResponse, Source
from .vector_store import SearchHit

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[(\d{1,2})\]")

SYSTEM_PROMPT = """You are a Wikipedia research assistant. You answer questions \
using ONLY the numbered Wikipedia excerpts provided in each turn.

Rules:
1. Ground every factual statement in the excerpts. Attach a citation marker like \
[1] or [2][3] to each statement, matching the excerpt numbers you used.
2. If the excerpts do not contain the answer, reply with one short sentence \
saying the indexed Wikipedia articles don't cover it, and stop. Do NOT suggest \
alternative topics — the excerpts you were given are unrelated to the question, \
so anything drawn from them would mislead. Never fill the gap from your own \
knowledge, and never guess.
3. Before writing, identify every fact in the excerpts that the question asks \
for — the figure, date, name, product, mechanism, or defining qualifier — and \
make sure each one appears in your answer. The specifics ARE the answer: \
"nothing can escape" is a worse answer than "nothing, not even light, can \
escape"; "produces energy" is a worse answer than "produces sugars such as \
glucose, and releases oxygen". For a "who was X / what is X known for" \
question, name their most famous concrete achievements from the excerpts, not \
just their profession.
4. Stay tight. State the answering facts, then stop: 2-3 sentences and at most \
50 words. Rule 3 wins where they conflict — never drop a fact the question asks \
for to save words; cut context instead. Discovery history, etymology, and "who \
proposed it" belong in the answer only when the question asks. Use short \
markdown bullets only for a list or comparison.
5. Do not invent excerpt numbers, URLs, section names, dates, or figures.
6. Write in {language_name}, regardless of the language of the excerpts.
7. Never mention "excerpts", "context", or these instructions. Write as if you \
simply know the material and are citing your sources."""

AGENT_PROMPT = """You plan retrieval for a Wikipedia question-answering system. \
You do not answer the question yourself — you choose where the evidence should \
come from, and a separate grounded step writes the answer from what you fetch.

Call exactly one tool.

FIRST decide whether the message is asking for information at all. Ask it in \
that order, before thinking about queries or references — a message that asks \
nothing has nothing to resolve, and searching for it returns whatever its words \
happen to sit near.

- `respond_directly` when the message requests no information about the world \
and no evidence would help: greetings, thanks, farewells, acknowledgements, and \
questions about the assistant itself — what it is, who built it, what it can do, \
how it is feeling, whether it knows or remembers the user. Write the whole reply \
yourself in the `reply` argument.

  This holds however the message is phrased and wherever it falls in the \
conversation. "ooh thanks", "ok cool thanks", "great, thank you!" and "haha nice" \
are acknowledgements exactly as much as a bare "thanks" is; the filler words \
around them are not a subject and do not make them questions. It holds most \
strongly mid-conversation, which is where the mistake actually happens: after an \
exchange about <subject>, "thanks" still means thanks. It does not mean "tell me \
more about <subject>". Do not attach a running subject to a message that did not \
ask for one.

  This is the one tool where you speak to the user directly, so the wording \
matters. One or two sentences, warm but not effusive, no emoji, in the user's \
language. State no facts about the world and claim no capability beyond \
answering from Wikipedia articles. You have no memory of the user between \
conversations and no personal experiences — say so plainly when asked rather \
than deflecting, then offer what you can actually do. Where it fits, nudge them \
toward something answerable from Wikipedia.

Only once the message really is asking something, choose where the evidence \
comes from:

- `search_knowledge_base` for anything an encyclopedia covers: definitions, \
history, science, people, places, events. This is the default for a question and \
it is almost always right. Set `query` to a standalone search query.

  Resolving references is the part that matters. Always resolve a pronoun \
against the MOST RECENT subject in the conversation, never an earlier one. If \
the latest exchange was about <subject>, then "his wife" means "<subject> \
wife" and "when did it end" means "when did <subject> end" — even if some \
earlier exchange was about someone else, and even if that earlier exchange \
answered a similarly worded question. The conversation moves on; the pronoun \
follows it. Keep proper nouns exactly as written, and pass the message \
unchanged when it already stands alone.
- `search_wikipedia` ONLY when the indexed corpus plainly cannot hold the \
answer: recent events, or a subject a prior `search_knowledge_base` call in \
this same turn already failed to cover. Never reach for it first.
- `answer_from_history` when the question is about this conversation itself \
("what did you just tell me?", "summarise that") and needs no new evidence. \
This is rare — a question that merely follows on from the last turn still \
needs a search.

Prefer one call. A second search costs the user a second of latency, so make \
it only when the first genuinely came back with nothing usable."""

# Given to the model as OpenAI tool schemas. Descriptions are deliberately
# short: the routing policy lives in AGENT_PROMPT above, and duplicating it
# here just gives the model two places to disagree with itself.
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Semantic search over the indexed Wikipedia corpus. The default "
                "source for any factual question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Standalone search query, with every pronoun "
                            "resolved against the most recent subject in the "
                            "conversation rather than an earlier one."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": (
                "Live MediaWiki search against wikipedia.org. Only for subjects "
                "the indexed corpus cannot cover, such as recent events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms for the live Wikipedia API.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_from_history",
            "description": (
                "Answer from the conversation so far without retrieving anything."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "respond_directly",
            "description": (
                "Reply conversationally, with no retrieval. For messages that "
                "ask nothing about the world: greetings, thanks, farewells, "
                "acknowledgements, and questions about the assistant itself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reply": {
                        "type": "string",
                        "description": (
                            "The complete reply to show the user, in their "
                            "language. One or two sentences."
                        ),
                    }
                },
                "required": ["reply"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in AGENT_TOOLS}


@dataclass(frozen=True)
class ToolCall:
    """One tool the agent chose, with its arguments already parsed."""

    name: str
    query: str = ""
    # Only `respond_directly` sets this: the reply the model wrote for a
    # message that needs no evidence. Carried on the tool call so a greeting
    # costs one round trip — deciding and answering are the same call, rather
    # than a routing call followed by a generation call.
    reply: str = ""


def parse_tool_calls(message, fallback_query: str) -> list[ToolCall]:
    """Read tool calls off a chat completion message.

    Fails open to a knowledge-base search on anything unexpected — no tool
    call, an unknown name, malformed argument JSON. The failure modes are
    asymmetric: searching the index when we did not need to costs a few hundred
    milliseconds, while dropping a real question because the model emitted bad
    JSON loses the answer entirely.
    """
    default = [ToolCall("search_knowledge_base", fallback_query)]

    raw_calls = getattr(message, "tool_calls", None) or []
    calls: list[ToolCall] = []
    for raw in raw_calls:
        name = getattr(getattr(raw, "function", None), "name", "") or ""
        if name not in TOOL_NAMES:
            logger.warning("Agent asked for unknown tool %r; ignoring", name)
            continue

        args: dict = {}
        if name != "answer_from_history":
            try:
                args = json.loads(getattr(raw.function, "arguments", "") or "{}")
            except json.JSONDecodeError:
                logger.warning("Agent sent malformed arguments for %s", name)
                args = {}

        if name == "respond_directly":
            # The reply *is* the tool's output, so an empty one leaves nothing
            # to say. Dropping the call falls through to the default search,
            # which for a greeting costs a stiff "not in the index" reply —
            # markedly better than emitting an empty turn.
            reply = str((args or {}).get("reply") or "").strip()
            if not reply:
                logger.warning("respond_directly carried no reply; falling back to search")
                continue
            calls.append(ToolCall(name, "", reply))
            continue

        query = ""
        if name != "answer_from_history":
            query = str((args or {}).get("query") or "").strip() or fallback_query
        calls.append(ToolCall(name, query))

    if not calls:
        return default
    # One tool per hop: parallel calls would mean two retrievals racing into
    # one context block with no budget left to reconcile them.
    return calls[:1]


# Only used for the "answer in this language" instruction; unknown codes fall
# back to the code itself, which GPT models handle fine.
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "pt": "Portuguese",
    "it": "Italian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
    "mr": "Marathi",
}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code.lower(), code)


def build_context_block(hits: list[SearchHit]) -> str:
    """Render retrieved chunks as a numbered, attributable block."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        body = hit.metadata.get("body") or hit.text
        blocks.append(
            f"[{i}] {hit.metadata.get('section_path', hit.title)}\n"
            f"URL: {hit.section_url}\n"
            f"{body}"
        )
    return "\n\n---\n\n".join(blocks)


def _snippet(hit: SearchHit, limit: int = 260) -> str:
    body = (hit.metadata.get("body") or hit.text).strip().replace("\n", " ")
    return body if len(body) <= limit else body[: limit - 1].rsplit(" ", 1)[0] + "…"


def cited_indices(answer: str, source_count: int) -> list[int]:
    """Citation numbers the model actually used, in order of first appearance."""
    seen: list[int] = []
    for match in CITATION_PATTERN.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= source_count and n not in seen:
            seen.append(n)
    return seen


# Words that carry no subject information, so their presence tells us nothing
# about whether a claim matches the source it cites.
_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does for from had
    has have he her him his how however in into is it its more most not of on
    only or she some such than that the their them then there these they this
    those to was were what when where which who whom whose will with would you
    your also been over under between during about
    """.split()
)

_WORD = re.compile(r"[A-Za-zÀ-ÿ0-9']+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LIST_PREFIX = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s*)")

# Fraction of a sentence's content words that must also appear in the chunk it
# cites. Deliberately permissive: this catches a citation pointing at unrelated
# material, and is not meant to police paraphrase or inference.
_MIN_CITATION_OVERLAP = 0.25


def _ordered_terms(text: str) -> list[str]:
    return [
        word.lower()
        for word in _WORD.findall(text)
        if len(word) > 2 and word.lower() not in _STOPWORDS
    ]


def _content_terms(text: str) -> set[str]:
    return set(_ordered_terms(text))


def _term_present(term: str, chunk_terms: set[str]) -> bool:
    """Membership that tolerates the singular/plural split.

    Without it "language" misses a chunk that only ever says "languages", which
    would fail honest sentences far more often than dishonest ones.
    """
    if term in chunk_terms:
        return True
    stem = term.rstrip("s")
    return any(other.rstrip("s") == stem for other in chunk_terms)


# "X is a Y" asserts a category, and the category is precisely what a conflated
# answer gets wrong: *"Java is a programming language …"* cited to the island
# article. Bag-of-words overlap cannot see it — every other word in that
# sentence does come from the chunk.
#
# The pattern is deliberately narrow, because an earlier, looser version broke
# well-grounded answers at a 25% rate. It requires a determiner ("is **a**
# programming language"), which is what separates a category claim from a
# composition statement ("DNA is made of…") or a hedge ("was not a single
# route…") — both of which paraphrase far too freely to check this way. Only
# the first two predicate words are examined: that is the category itself,
# before the qualifiers start.
_DEFINITION = re.compile(
    r"^\s*[\"“']?[A-Z][\w'’\-]*(?:\s+[\w'’\-()]+){0,3}\s+(?:is|was|are|were)\s+"
    r"(?:a|an|the)\s+(?P<predicate>[^,.;:]{3,80})"
)
_NEGATION = re.compile(r"\b(?:not|no|never|rather than|instead of)\b", re.I)
_DEFINITION_TERMS = 2


def _definition_grounded(sentence: str, chunk_terms: set[str]) -> bool:
    match = _DEFINITION.match(sentence)
    if not match or _NEGATION.search(sentence):
        return True
    predicate = _ordered_terms(match.group("predicate"))[:_DEFINITION_TERMS]
    if len(predicate) < _DEFINITION_TERMS:
        return True
    return all(_term_present(term, chunk_terms) for term in predicate)


def _sentence_supported(sentence: str, hits: list[SearchHit]) -> tuple[bool, str]:
    """Whether a sentence's vocabulary matches the chunks it cites."""
    indices = cited_indices(sentence, len(hits))
    if not indices:
        # Uncited text is connective tissue, not a sourced claim. Leaving it
        # alone is the point: only citations get held to a citation's standard.
        return True, ""

    body = CITATION_PATTERN.sub("", sentence)
    terms = _content_terms(body)
    if not terms:
        return True, ""

    best = 0.0
    definition_ok = False
    for n in indices:
        hit = hits[n - 1]
        chunk_terms = _content_terms(
            f"{hit.title} {hit.metadata.get('section_path', '')} "
            f"{hit.metadata.get('body') or hit.text}"
        )
        best = max(best, len(terms & chunk_terms) / len(terms))
        # One cited chunk carrying the definition is enough to justify it.
        definition_ok = definition_ok or _definition_grounded(body, chunk_terms)

    if best < _MIN_CITATION_OVERLAP:
        return False, f"{best:.0%} term overlap with cited chunk"
    if not definition_ok:
        return False, "defining claim absent from cited chunk"
    return True, ""


def verify_citations(
    answer: str, hits: list[SearchHit]
) -> tuple[str, list[tuple[str, str]]]:
    """Drop sentences whose citation points at unrelated material.

    On an ambiguous query the generator occasionally welds two subjects
    together — *"Java is a programming language with four main spoken languages
    on the island of Java [2]"*, where chunk [2] is the island article and says
    nothing about a programming language. The URL is real and the marker is in
    range, so neither :func:`cited_indices` nor :func:`build_sources` can see
    the problem: the mismatch is between the **claim** and its **source**.

    Runs before :func:`build_sources`, so a dropped sentence also drops its
    citation from the source list rather than leaving evidence for a claim that
    is no longer there. An answer emptied by this check leaves the caller with
    no cited sources, which routes it to the ordinary not-covered path.

    Returns the cleaned answer and the (sentence, reason) pairs removed.
    """
    if not hits or not answer.strip():
        return answer, []

    rejected: list[tuple[str, str]] = []
    kept_lines: list[str] = []

    for line in answer.splitlines():
        if not line.strip():
            kept_lines.append(line)
            continue

        # Keep any bullet/number marker attached to the line rather than the
        # sentence, so removing a sentence never leaves a dangling "- ".
        match = _LIST_PREFIX.match(line)
        prefix = match.group(1) if match else ""
        body = line[len(prefix):]

        kept: list[str] = []
        for sentence in _SENTENCE_SPLIT.split(body):
            if not sentence.strip():
                continue
            supported, reason = _sentence_supported(sentence, hits)
            if supported:
                kept.append(sentence)
            else:
                rejected.append((sentence.strip(), reason))

        if kept:
            kept_lines.append(prefix + " ".join(kept))

    return "\n".join(kept_lines).strip(), rejected


def build_sources(hits: list[SearchHit], answer: str) -> list[Source]:
    """Keep only cited chunks; markers stay valid because we do not renumber."""
    used = cited_indices(answer, len(hits))
    # If the model cited nothing (e.g. a refusal), show nothing rather than
    # implying evidence that was not used.
    return [
        Source(
            index=n,
            title=hits[n - 1].title,
            section=str(hits[n - 1].metadata.get("section_title", "")),
            url=hits[n - 1].section_url,
            article_url=str(hits[n - 1].metadata.get("url", "")),
            lang=str(hits[n - 1].metadata.get("lang", "en")),
            snippet=_snippet(hits[n - 1]),
            score=round(hits[n - 1].score, 4),
        )
        for n in used
    ]


# Conversational scaffolding that reads badly when quoted back at the user:
# "I don't have anything about and about his wife" -> "…about his wife".
_LEADING_FILLER = ("and ", "but ", "so ", "also ", "then ", "ok ", "okay ")

# Lead-ins we can strip and still be left with a noun phrase that reads
# naturally after "anything about …":
#   "What is the current stock price of Tesla?" -> "the current stock price of Tesla"
#   "Tell me about the Nazca lines"             -> "the Nazca lines"
# Deliberately *not* listed: how/when/where/why and auxiliary openers ("did the
# Berlin Wall fall"). Stripping those leaves a bare verb phrase, which is
# ungrammatical after "about", so they fall through to the generic phrasing.
_SUBJECT_LEAD_INS = (
    re.compile(r"^(?:can you |could you |please )?tell me (?:about|more about)\s+", re.I),
    re.compile(r"^(?:do you know|do you have)\s+(?:anything\s+)?(?:about|on)\s+", re.I),
    re.compile(r"^i (?:want|need|would like) to know (?:about|more about)\s+", re.I),
    re.compile(r"^(?:what|which|who)\s+(?:is|are|was|were)\s+", re.I),
    re.compile(r"^what about\s+", re.I),
    re.compile(r"^about\s+", re.I),
    # Same shape in the other languages we render refusals in.
    re.compile(r"^¿?(?:qué|que|cuál|cual|quién|quien)\s+(?:es|son|era|eran|fue|fueron)\s+", re.I),
    re.compile(r"^¿?(?:háblame|hablame|cuéntame|cuentame) de\s+", re.I),
    re.compile(r"^(?:qu'est-ce que|quel est|quelle est|qui est|qui était)\s+", re.I),
    re.compile(r"^parle-moi de\s+", re.I),
)

# Openers that mean the text is still a clause, not a topic.
_INTERROGATIVE_OPENERS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "will",
    "would", "should", "has", "have", "had", "tell", "give", "list", "explain",
    "qué", "que", "cuál", "cual", "quién", "quien", "cuándo", "cuando", "dónde",
    "donde", "cómo", "como", "por", "quel", "quelle", "qui", "quand", "où",
    "comment", "pourquoi",
}

# A stripped subject longer than this is almost certainly still a whole clause.
_MAX_SUBJECT_WORDS = 9


def _looks_like_question(text: str) -> bool:
    words = text.split()
    if len(words) <= 1:
        return False
    return words[0].strip("¿¡").lower() in _INTERROGATIVE_OPENERS


def extract_subject(topic: str) -> str | None:
    """What a question is *about*, or ``None`` when we cannot tell.

    Returning ``None`` matters as much as returning a subject. The refusal reads
    "I don't have anything about <subject>", so a bad extraction produces
    "…anything about What is the current stock price of Tesla?" — the exact
    ungrammatical echo this exists to prevent. Where confidence is low the
    caller substitutes a generic stand-in instead of guessing.
    """
    subject = " ".join((topic or "").split()).strip()
    # Trailing punctuation belongs to the question, never to the subject.
    subject = subject.rstrip("?!. ").strip()
    if not subject:
        return None

    lowered = subject.lower()
    for filler in _LEADING_FILLER:
        if lowered.startswith(filler):
            subject = subject[len(filler):].strip()
            lowered = subject.lower()

    stripped_lead_in = False
    for pattern in _SUBJECT_LEAD_INS:
        candidate, count = pattern.subn("", subject, count=1)
        if count:
            subject = candidate.strip()
            stripped_lead_in = True
            break

    if not subject:
        return None
    # A bare topic ("Mercury", "the Silk Road") is already its own subject; one
    # that still opens with an interrogative is not something we can quote back.
    if not stripped_lead_in and _looks_like_question(subject):
        return None
    if len(subject.split()) > _MAX_SUBJECT_WORDS:
        return None
    return subject[:80]


# Refusal wording per language. Only languages whose phrasing we can actually
# check are listed — anything else falls back to English rather than shipping an
# unverified translation. `subject` is either the extracted topic or the entry's
# own generic stand-in, so each template needs only one form.
_REFUSALS: dict[str, dict[str, str]] = {
    "en": {
        "generic_subject": "that",
        "conjunction": "and",
        "suggest": (
            "I don't have anything about {subject} in your knowledge base yet — "
            "but Wikipedia does. Have a look at {titles}, listed under Sources. "
            "You can open them directly, or add one to your knowledge base and "
            "ask me again for a cited answer."
        ),
        "empty": (
            "I couldn't find anything about {subject}, either in your knowledge "
            "base or on Wikipedia. Try a more specific name or spelling."
        ),
    },
    "es": {
        "generic_subject": "eso",
        "conjunction": "y",
        "suggest": (
            "Todavía no tengo nada sobre {subject} en tu base de conocimiento, "
            "pero Wikipedia sí. Echa un vistazo a {titles}, que aparecen en "
            "Fuentes. Puedes abrirlos directamente, o añadir uno a tu base de "
            "conocimiento y preguntarme de nuevo para obtener una respuesta con citas."
        ),
        "empty": (
            "No he encontrado nada sobre {subject}, ni en tu base de conocimiento "
            "ni en Wikipedia. Prueba con un nombre o una grafía más específicos."
        ),
    },
    "fr": {
        "generic_subject": "cela",
        "conjunction": "et",
        "suggest": (
            "Je n'ai encore rien sur {subject} dans votre base de connaissances, "
            "mais Wikipédia en a. Jetez un œil à {titles}, indiqués sous Sources. "
            "Vous pouvez les ouvrir directement, ou en ajouter un à votre base de "
            "connaissances et me reposer la question pour une réponse sourcée."
        ),
        "empty": (
            "Je n'ai rien trouvé sur {subject}, ni dans votre base de connaissances "
            "ni sur Wikipédia. Essayez un nom ou une orthographe plus précis."
        ),
    },
    "de": {
        "generic_subject": "das",
        "conjunction": "und",
        "suggest": (
            "Zu {subject} habe ich noch nichts in deiner Wissensdatenbank, aber "
            "Wikipedia schon. Schau dir {titles} an, unter Quellen aufgeführt. Du "
            "kannst sie direkt öffnen oder einen Artikel zu deiner Wissensdatenbank "
            "hinzufügen und mich erneut fragen, um eine Antwort mit Quellen zu erhalten."
        ),
        "empty": (
            "Ich konnte nichts zu {subject} finden, weder in deiner Wissensdatenbank "
            "noch auf Wikipedia. Versuche einen genaueren Namen oder eine andere "
            "Schreibweise."
        ),
    },
    "pt": {
        "generic_subject": "isso",
        "conjunction": "e",
        "suggest": (
            "Ainda não tenho nada sobre {subject} na sua base de conhecimento, mas "
            "a Wikipédia tem. Dê uma olhada em {titles}, listados em Fontes. Pode "
            "abri-los diretamente, ou adicionar um à sua base de conhecimento e "
            "perguntar de novo para obter uma resposta com citações."
        ),
        "empty": (
            "Não encontrei nada sobre {subject}, nem na sua base de conhecimento nem "
            "na Wikipédia. Tente um nome ou grafia mais específicos."
        ),
    },
    "it": {
        "generic_subject": "questo",
        "conjunction": "e",
        "suggest": (
            "Non ho ancora nulla su {subject} nella tua base di conoscenza, ma "
            "Wikipedia sì. Dai un'occhiata a {titles}, elencati sotto Fonti. Puoi "
            "aprirli direttamente, oppure aggiungerne uno alla tua base di conoscenza "
            "e chiedermelo di nuovo per una risposta con citazioni."
        ),
        "empty": (
            "Non ho trovato nulla su {subject}, né nella tua base di conoscenza né su "
            "Wikipedia. Prova con un nome o un'ortografia più specifici."
        ),
    },
}


def _join_titles(titles: list[str], conjunction: str = "and") -> str:
    if len(titles) == 1:
        return f"“{titles[0]}”"
    quoted = [f"“{t}”" for t in titles]
    return ", ".join(quoted[:-1]) + f" {conjunction} {quoted[-1]}"


def compose_not_covered(
    topic: str, article_titles: list[str], lang: str = "en"
) -> str:
    """The answer given when nothing indexed supports a question.

    Written here rather than by the model, for two reasons. A refusal contains
    no facts, so there is nothing for the model to add — and left to itself it
    produces dead ends like "The excerpts do not cover mitochondria", which
    both leaks internal vocabulary and tells the user nothing they can act on.

    When live search found real articles, name them: the useful part of a
    refusal is what to read instead. Written in the caller's language, because a
    Spanish question answered with an English apology is its own kind of miss.
    """
    strings = _REFUSALS.get(lang.lower(), _REFUSALS["en"])
    subject = extract_subject(topic) or strings["generic_subject"]

    if article_titles:
        return strings["suggest"].format(
            subject=subject,
            titles=_join_titles(article_titles[:3], strings["conjunction"]),
        )
    return strings["empty"].format(subject=subject)


def turn_meta(response: ChatResponse) -> dict | None:
    """Citation payload stored with an assistant turn, so a restored
    conversation keeps its sources and article links, not just its text."""
    if not response.sources and not response.articles:
        return None
    return {
        "sources": [s.model_dump() for s in response.sources],
        "articles": [a.model_dump() for a in response.articles],
    }


def build_article_links(hits: list[SearchHit], sources: list[Source]) -> list[ArticleLink]:
    """Smart routing: one deduplicated link per distinct article, cited first."""
    cited_titles = {s.title for s in sources}
    ordered: list[ArticleLink] = []
    seen: set[str] = set()

    for hit in sorted(hits, key=lambda h: (h.title not in cited_titles, -h.score)):
        if hit.title in seen:
            continue
        seen.add(hit.title)
        ordered.append(
            ArticleLink(
                title=hit.title,
                url=str(hit.metadata.get("url", hit.section_url)),
                lang=str(hit.metadata.get("lang", "en")),
                summary=_snippet(hit, limit=160),
            )
        )
    return ordered[:5]
