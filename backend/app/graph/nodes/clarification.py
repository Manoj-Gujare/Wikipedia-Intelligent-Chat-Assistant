"""Node 5: ask which entity was meant rather than answering the wrong one."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ..state import ChatState
from .timing import timed

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
