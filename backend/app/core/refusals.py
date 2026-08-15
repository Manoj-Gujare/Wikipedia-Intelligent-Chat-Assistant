"""What to say when nothing indexed supports the question.

Written in Python rather than by the model. A refusal contains no facts, so
there is nothing for a model to add -- and left to itself it produces dead ends
like "The excerpts do not cover mitochondria", which leaks internal vocabulary
and tells the user nothing they can act on.

The hard part is `extract_subject`: quoting the question back only reads well
when what you quote is a noun phrase.
"""

from __future__ import annotations

import re


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
