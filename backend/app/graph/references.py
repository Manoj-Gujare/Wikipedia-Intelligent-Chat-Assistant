"""Whether a message leans on the turn before it.

Two nodes need this and neither owns it: the agent uses it to decide what is
worth speculating on, and the generator uses it to decide whether to phrase the
answer around the user's words or the resolved query. Both questions are really
one question about the message.
"""

from __future__ import annotations

import re

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


# A capitalised word used to be required here, as a proxy for "the message
# names its own subject". It was measured out: capitalisation is a property of
# how someone types, not of whether their question stands alone. In the running
# app "i want to know about the albert einstein" and "what is machine learning"
# both failed it and paid 0.36-1.29s of serial retrieval, while the same
# questions in the eval suite passed purely because the suite is written in
# textbook capitalisation — so the benchmark could not see the loss.
#
# What remains is the part that was doing real work: not elliptical, not
# referential, and long enough to embed. Those describe a message that does not
# lean on the previous turn, which is exactly what speculation needs. Dropping
# the proxy admits some genuinely elliptical messages ("What is the capital?"),
# and that is affordable *here specifically* — this function only decides
# whether to launch a parallel embedding. Whether the parked result is ever
# used is `speculation._similar`'s decision, and it is unchanged.


# Below this, a question is almost certainly leaning on the previous turn
# ("Tell me more", "And the population?").
_MIN_SELF_CONTAINED_WORDS = 4


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[\w']+", (text or "").lower()))


def _is_referential(message: str) -> bool:
    """Whether the message leans on the turn before it to mean anything."""
    lowered = message.strip().lower()
    if lowered.startswith(_ELLIPTICAL_OPENERS):
        return True
    return any(word in _REFERENTIAL for word in re.findall(r"[a-z']+", lowered))


def contains_pronoun(message: str) -> bool:
    """Whether the message has a word that *only* the history can resolve.

    The distinction this draws is between the two ways a message can look
    referential. "what about his wife?" cannot be searched as written — `his`
    points outside the message. "what about black hole" opens the same way but
    names its own subject, and needs nothing from the turn before it.

    `_is_referential` deliberately treats both as referential, because for
    speculation the opener alone is reason enough to be careful. The
    subject-corroboration backstop needs the sharper question, because its
    remedy is to graft the previous question on — help for the first message
    and corruption for the second.
    """
    lowered = (message or "").strip().lower()
    return any(word in _REFERENTIAL for word in re.findall(r"[a-z']+", lowered))


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

    return len(text.split()) >= _MIN_SELF_CONTAINED_WORDS
