"""Node 2: emit the reply the agent already wrote. No retrieval, no second call."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ..state import ChatState
from .speculation import _discard
from .timing import timed

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
