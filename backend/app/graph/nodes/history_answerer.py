"""Node 6: answer a question about the conversation itself, with no retrieval.

Cites nothing, by construction: the sources belong to the turns being
summarised, and re-attaching them here would claim the summary was
independently grounded when it was not.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from ...core.prompts import language_name
from ..state import ChatState
from .speculation import _discard
from .timing import timed

logger = logging.getLogger(__name__)

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
    _discard(state.get("speculative_answer"))
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
        "speculative_answer": None,
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
        "speculative_answer": None,
        "messages": [AIMessage(content=answer)],
    }
