"""The graph's nodes, one module each.

Seven nodes. The model chooses *whether evidence is wanted at all* and then
where it comes from -- indexed corpus, live Wikipedia, or the conversation
itself. Two things stay in Python, because handing them to the model costs
latency it cannot earn back:

* retrieval on a plausible query starts *before* the agent has decided, racing
  the decision call instead of queueing behind it (see `speculation`),
* a second hop is authorised by the wall clock, not by the agent's enthusiasm
  (see `tool_executor.route_after_tools`).

Classifying the message is not on that list, and used to be. A phrase table
answered greetings for free and could only ever recognise the phrasings
someone had written down; "Do you know me?" was on no list, so it was searched
for in an encyclopedia. Deciding what a message *is* turns out to be the one
job worth a round trip, and speculation is what pays for it.

Re-exported here so `build.py` and the tests import the node set from one
place rather than tracking which module each node lives in.
"""

from ..references import looks_self_contained
from .agent import agent, route_after_agent
from .clarification import clarification
from .direct_responder import direct_responder
from .gate import gate
from .generator import generator
from .history_answerer import history_answerer
from .timing import timed
from .tool_executor import route_after_tools, tool_executor

__all__ = [
    "agent",
    "clarification",
    "direct_responder",
    "gate",
    "generator",
    "history_answerer",
    "looks_self_contained",
    "route_after_agent",
    "route_after_tools",
    "timed",
    "tool_executor",
]
