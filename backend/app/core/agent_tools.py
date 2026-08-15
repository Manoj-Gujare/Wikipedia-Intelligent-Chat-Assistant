"""The tool menu the agent chooses from, and how its choice is read back.

One tool per hop. The names are constants rather than literals because the
graph nodes branch on them, and a typo in a branch is a silently unreachable
node.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# The agent's four options. `respond_directly` is the one that answers without
# retrieving anything; the other three all end in some form of evidence.
KB_SEARCH = "search_knowledge_base"
LIVE_SEARCH = "search_wikipedia"
HISTORY_ANSWER = "answer_from_history"
DIRECT_REPLY = "respond_directly"

# Given to the model as OpenAI tool schemas. Descriptions are deliberately
# short: the routing policy lives in AGENT_PROMPT above, and duplicating it
# here just gives the model two places to disagree with itself.
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": KB_SEARCH,
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
            "name": LIVE_SEARCH,
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
            "name": HISTORY_ANSWER,
            "description": (
                "Answer from the conversation so far without retrieving anything."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": DIRECT_REPLY,
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
    default = [ToolCall(KB_SEARCH, fallback_query)]

    raw_calls = getattr(message, "tool_calls", None) or []
    calls: list[ToolCall] = []
    for raw in raw_calls:
        name = getattr(getattr(raw, "function", None), "name", "") or ""
        if name not in TOOL_NAMES:
            logger.warning("Agent asked for unknown tool %r; ignoring", name)
            continue

        args: dict = {}
        if name != HISTORY_ANSWER:
            try:
                args = json.loads(getattr(raw.function, "arguments", "") or "{}")
            except json.JSONDecodeError:
                logger.warning("Agent sent malformed arguments for %s", name)
                args = {}

        if name == DIRECT_REPLY:
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
        if name != HISTORY_ANSWER:
            query = str((args or {}).get("query") or "").strip() or fallback_query
        calls.append(ToolCall(name, query))

    if not calls:
        return default
    # One tool per hop: parallel calls would mean two retrievals racing into
    # one context block with no budget left to reconcile them.
    return calls[:1]
