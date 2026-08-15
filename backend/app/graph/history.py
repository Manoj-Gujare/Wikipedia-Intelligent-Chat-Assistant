"""Which parts of a transcript a turn should actually see.

Two different questions, and conflating them changed answers (see
``conversational_turns``): what the *subject* of the conversation is, and what
the agent should read as context. Small talk is excluded from both, but in
different shapes — one turn at a time for the subject, whole exchanges for the
transcript.
"""

from __future__ import annotations

from ..core.conversation import Turn


def substantive_turns(history: list[Turn]) -> list[Turn]:
    """User turns that carry an information need — small talk excluded.

    "Hey" is a turn in the transcript but not a *subject*: nothing in it can be
    referred back to, and a conversation consisting only of greetings is still
    a conversation nobody has asked anything in yet. Stitching a follow-up onto
    one produces a search for a greeting and a pronoun, which is strictly worse
    than not speculating at all.

    The verdict is *read* here, not computed. It was computed, by the same
    phrase-matching classifier the gate used, on every turn of every history
    lookup — which meant the classifier had to be synchronous, and that is
    precisely why it could never be the model. Recording the agent's decision
    when the turn is written inverts that: each turn is classified once, by the
    model that was already deciding, and this stays a list comprehension.
    """
    return [t for t in history if t.role == "user" and not (t.meta or {}).get("chitchat")]


def conversational_turns(history: list[Turn]) -> list[Turn]:
    """History with small-talk exchanges removed, both halves of each.

    The transcript the agent decides against used to be built from raw history
    while the *subject* it resolves against excluded small talk — half an idea,
    and the missing half changed answers. Measured: "Who are you?" routes to
    `respond_directly` on an empty transcript and after a real exchange, but
    after `Hey` / `Hello! How can I assist you today?` it was routed to
    `search_knowledge_base` with the query "What is ChatGPT". An assistant
    offering to help, followed by a question about the assistant, reads as a
    request to look something up unless the pleasantries are out of view.

    The assistant's reply goes with the user turn that prompted it. Dropping
    only the user's "thanks" would leave "You're welcome." standing on its own,
    which is a stranger transcript than either keeping or removing the pair.
    """
    kept: list[Turn] = []
    drop_reply = False
    for turn in history:
        if turn.role == "user":
            drop_reply = bool((turn.meta or {}).get("chitchat"))
            if drop_reply:
                continue
        elif drop_reply:
            drop_reply = False
            continue
        kept.append(turn)
    return kept


def last_user_question(history: list[Turn]) -> str | None:
    """The most recent thing the user actually asked about."""
    turns = substantive_turns(history)
    return turns[-1].content if turns else None
