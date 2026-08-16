"""Measure the two headline metrics: answer accuracy and response time.

    python -m scripts.evaluate            # full suite
    python -m scripts.evaluate --limit 5  # smoke test

Accuracy is graded by an LLM judge that sees the question, the expected key
facts, and the produced answer. It is strict: an answer that is vague, hedged
into uselessness, or missing a required fact is marked incorrect. Latency is
reported as p50/p95 over the same run, with the cache disabled so numbers
reflect cold-path work.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass

from app.config import get_settings
from app.core.cache import get_cache
from app.core.embeddings import get_openai_client
from app.graph.build import get_graph, get_services
from app.graph.runner import stream_turn
from app.models import ChatResponse

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# The summary table draws box characters, and a Windows console defaults to
# cp1252 — which cannot encode them, so the run would die on the last print
# after every case had already passed. Force UTF-8 on our own streams.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # pragma: no cover - non-reconfigurable stream
        pass

TARGET_ACCURACY = 0.90
# The specification's "<3s response time": complete response, not first token.
TARGET_SECONDS = 3.0


@dataclass(frozen=True)
class EvalCase:
    question: str
    must_include: str
    lang: str = "en"
    expect_refusal: bool = False


SUITE: list[EvalCase] = [
    EvalCase("What is the speed of light in a vacuum?", "roughly 299,792,458 metres per second"),
    EvalCase("Who developed the theory of general relativity?", "Albert Einstein"),
    EvalCase("What does photosynthesis produce?", "glucose/sugars and oxygen from CO2, water and light"),
    EvalCase("What is a black hole's event horizon?", "the boundary beyond which nothing, not even light, can escape"),
    EvalCase("When did World War II end?", "1945"),
    EvalCase("What is DNA made of?", "nucleotides with the bases adenine, thymine, guanine, cytosine; double helix"),
    EvalCase("Who was Alan Turing and what is he known for?", "British mathematician, computer science pioneer, Turing machine, Enigma codebreaking"),
    EvalCase("What is machine learning?", "field of AI where systems learn patterns from data rather than explicit rules"),
    EvalCase("Which is the tallest mountain above sea level?", "Mount Everest"),
    EvalCase("What was the Silk Road?", "network of trade routes connecting East Asia with the Mediterranean"),
    EvalCase("What is the periodic table organised by?", "atomic number, with recurring chemical properties in groups/periods"),
    EvalCase("What did Marie Curie discover?", "radioactivity research; polonium and radium"),
    EvalCase("What causes plate tectonics?", "convection in the mantle moving lithospheric plates"),
    EvalCase("What is the Big Bang theory?", "the universe expanded from an extremely hot dense initial state ~13.8 billion years ago"),
    EvalCase("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    EvalCase("What is an operating system?", "software managing hardware resources and providing services to programs"),
    EvalCase("What is natural selection?", "differential survival and reproduction of individuals with advantageous heritable traits"),
    EvalCase("What is the Taj Mahal?", "marble mausoleum in Agra, India, built by Shah Jahan for Mumtaz Mahal"),
    EvalCase("What is public-key cryptography?", "asymmetric key pair: public key encrypts/verifies, private key decrypts/signs"),
    EvalCase(
        "What is the current stock price of Tesla?",
        "an admission that the indexed Wikipedia content does not cover this",
        expect_refusal=True,
    ),
]

@dataclass(frozen=True)
class FollowUpCase:
    """A two-turn exchange where the second turn depends on the first.

    Only the follow-up is measured. Under the agentic graph both paths pay the
    same one decision call, so the follow-up is no longer slower by
    construction — what separates them now is *speculation*: an opening
    question's retrieval usually runs underneath the decision call, while a
    referential follow-up ("what about his wife?") cannot be speculated on and
    pays its retrieval in series. The two are still reported apart, because
    that difference is real and averaging them would hide it.
    """

    setup: str
    question: str
    must_include: str
    lang: str = "en"


FOLLOW_UPS: list[FollowUpCase] = [
    FollowUpCase(
        "Tell me about the Taj Mahal",
        "Who built it and why?",
        "Shah Jahan, as a mausoleum for his wife Mumtaz Mahal",
    ),
    FollowUpCase(
        "Who was Marie Curie?",
        "What did she discover?",
        "radioactivity research; the elements polonium and radium",
    ),
    FollowUpCase(
        "What is a black hole?",
        "What happens at its event horizon?",
        "it is the boundary beyond which nothing, not even light, can escape",
    ),
    FollowUpCase(
        "Tell me about Alan Turing",
        "What is he best known for?",
        "the Turing machine, founding computer science, wartime codebreaking",
    ),
    # The four above all resolve a pronoun, so none of them can be speculated
    # on — their retrieval waits for the agent's rewritten query. These two
    # switch topic mid-conversation instead: the question names its own
    # subject, so speculation fires and the search should be parked and waiting
    # by the time the agent asks for it. Without them the suite cannot see that
    # fast path at all, and an optimisation nothing measures is
    # indistinguishable from one that does not work.
    FollowUpCase(
        "Tell me about the Taj Mahal",
        "When did World War II end?",
        "1945",
    ),
    FollowUpCase(
        "Who was Alan Turing?",
        "What did Marie Curie discover?",
        "radioactivity research; the elements polonium and radium",
    ),
]


JUDGE_PROMPT = """You grade a Wikipedia assistant's answers. Reply with JSON only: \
{"correct": true|false, "reason": "<10 words"}.

Mark correct=true if the answer conveys the substance of the expected facts and \
contains no statement that contradicts them. Judge meaning, not wording: \
"breaking German wartime ciphers" conveys codebreaking even without the word \
"Enigma", and extra accurate detail is fine. Mark correct=false only if a \
required fact is substantively absent, wrong, or so hedged that the question is \
left unanswered."""

REFUSAL_JUDGE_PROMPT = """You grade a Wikipedia assistant's answers. Reply with JSON \
only: {"correct": true|false, "reason": "<10 words"}.

This question CANNOT be answered from static Wikipedia articles. Mark correct=true \
only if the assistant declines / says it lacks the information. Mark correct=false \
if it fabricates a specific answer."""


# The judge is the meterstick, so it should be the strongest model that is
# affordable at 20 calls per run — a sloppy judge fails correct answers on
# literal phrase-matching, which corrupts the accuracy number in both directions.
JUDGE_MODEL = "gpt-4.1"


def _answered_directly(response) -> bool:
    """Whether the agent answered conversationally instead of retrieving."""
    tools = response.agent.tools
    return bool(tools) and tools[0].tool == "respond_directly"


def _path_label(response) -> str:
    """Whether retrieval ran underneath the decision call or after it.

    A direct reply is called out separately: it speculates and discards, so
    "spec miss" would read as a failure when nothing was ever wanted.
    """
    if _answered_directly(response):
        return "direct   "
    return "spec HIT " if response.agent.speculation_hit else "spec miss"


async def judge(case: EvalCase, answer: str) -> tuple[bool, str]:
    client = get_openai_client()
    response = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": REFUSAL_JUDGE_PROMPT if case.expect_refusal else JUDGE_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Question: {case.question}\n"
                    f"Expected: {case.must_include}\n"
                    f"Answer: {answer}"
                ),
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        verdict = json.loads(response.choices[0].message.content or "{}")
        return bool(verdict.get("correct")), str(verdict.get("reason", ""))
    except json.JSONDecodeError:
        return False, "judge returned invalid JSON"


async def measure_turn(
    question: str, session_id: str | None, lang: str
) -> tuple[ChatResponse | None, float, float, str | None]:
    """Drive one streaming turn, timing it exactly as the UI experiences it.

    Returns (response, total seconds, seconds to first token, conversation id).
    """
    started = time.perf_counter()
    first_token: float | None = None
    response: ChatResponse | None = None
    conversation_id = session_id

    async for event, payload in stream_turn(question, session_id=session_id, lang=lang):
        if event == "meta":
            conversation_id = payload.get("conversation_id", conversation_id)
        elif event == "token" and first_token is None:
            first_token = time.perf_counter() - started
        elif event == "done":
            response = ChatResponse.model_validate(payload)

    elapsed = time.perf_counter() - started
    return response, elapsed, (first_token if first_token is not None else elapsed), conversation_id


async def run_follow_ups(cases: list[FollowUpCase]) -> tuple[list[float], int]:
    """Measure the follow-up turn of each exchange. Returns (times, correct)."""
    totals: list[float] = []
    correct = 0

    print("\nFollow-up turns (pronoun resolution, cold path)…\n")
    for case in cases:
        get_cache()._entries.clear()  # noqa: SLF001 - deliberate for benchmarking

        # The setup turn establishes context and is deliberately not timed.
        _, _, _, session_id = await measure_turn(case.setup, None, case.lang)
        response, elapsed, _, _ = await measure_turn(case.question, session_id, case.lang)
        if response is None:
            print(f"FAIL  {elapsed:5.2f}s  stream ended without a result: {case.question}")
            continue

        ok, reason = await judge(
            EvalCase(case.question, case.must_include, case.lang), response.answer
        )
        totals.append(elapsed)
        correct += int(ok)
        # Whether speculation fired is the point of half these cases, so show
        # it per-case rather than leaving it inside the average.
        agent = f"agent {response.timings.rewrite_ms:>4}ms"
        print(
            f"{'PASS' if ok else 'FAIL'}  {elapsed:5.2f}s total  {agent}  "
            f"{_path_label(response)}  "
            f"{len(response.sources)} src  {case.setup!r} -> {case.question!r}"
        )
        if not ok:
            print(f"      ↳ {reason}")

    return totals, correct


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate accuracy and latency")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N cases")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.openai_configured:
        print("OPENAI_API_KEY is missing or still the placeholder. Set it in .env.")
        return 1

    cases = SUITE[: args.limit] if args.limit else SUITE
    services = get_services()
    get_graph()  # compile before timing, as the server does at startup

    # Connection setup and Chroma's lazy index load are one-off process costs,
    # not per-request latency. The server does the same on startup, so paying it
    # here would measure something no user experiences.
    await _warmup(services)

    totals: list[float] = []
    first_tokens: list[float] = []
    retrievals: list[float] = []
    words: list[int] = []
    hops: list[int] = []
    correct = 0
    cited = 0
    spec_hits = 0
    direct = 0

    print(f"Running {len(cases)} cases against {settings.chat_model}…\n")

    for case in cases:
        # Fresh conversation per case, and never a cache hit: measure cold path.
        get_cache()._entries.clear()  # noqa: SLF001 - deliberate for benchmarking

        # Drives the same LangGraph streaming path the UI uses, so these
        # numbers describe what actually ships rather than a parallel codepath.
        response, elapsed, first_token, _ = await measure_turn(
            case.question, None, case.lang
        )
        if response is None:
            print(f"FAIL  {elapsed:5.2f}s  stream ended without a result: {case.question}")
            continue

        ok, reason = await judge(case, response.answer)
        totals.append(elapsed)
        first_tokens.append(first_token)
        retrievals.append(response.timings.retrieval_ms / 1000)
        words.append(len(response.answer.split()))
        correct += int(ok)
        spec_hits += int(response.agent.speculation_hit)
        direct += int(_answered_directly(response))
        hops.append(response.agent.hops)
        # A refusal legitimately has no citations; only grounded answers must cite.
        if response.sources or case.expect_refusal:
            cited += 1

        print(
            f"{'PASS' if ok else 'FAIL'}  {elapsed:5.2f}s total  "
            f"{first_tokens[-1]:4.2f}s first token  {_path_label(response)}  "
            f"{response.agent.hops} agent  {len(response.sources)} src  "
            f"{case.question}"
        )
        if not ok:
            print(f"      ↳ {reason}")

    follow_up_times, follow_up_correct = await run_follow_ups(
        FOLLOW_UPS[: args.limit] if args.limit else FOLLOW_UPS
    )

    accuracy = correct / len(cases)
    citation_rate = cited / len(cases)
    full_p50 = statistics.median(totals)
    full_p95 = percentile(totals, 0.95)
    tt_p95 = percentile(first_tokens, 0.95)

    # Pass criteria come straight from the specification: "<3s response time"
    # means the complete response, so that is what is gated. Time to first
    # token is reported because it explains *where* the time goes, but it is
    # not the requirement and does not decide pass/fail.
    accuracy_ok = accuracy >= TARGET_ACCURACY
    latency_ok = full_p50 < TARGET_SECONDS
    over = sum(1 for t in totals if t >= TARGET_SECONDS)

    # The follow-up median is gated too, not merely printed. It used to be
    # reported with an "OVER TARGET" label that nothing acted on, so a run
    # could announce a 4.85s follow-up median and still exit 0 — which is
    # exactly the shape of a benchmark people stop trusting. A multi-turn
    # answer is a response like any other, and the specification's budget
    # does not exempt it.
    follow_up_p50 = statistics.median(follow_up_times) if follow_up_times else 0.0
    follow_up_ok = not follow_up_times or follow_up_p50 < TARGET_SECONDS

    print("\n" + "=" * 68)
    print(f"Accuracy          {accuracy:6.1%}  (target >{TARGET_ACCURACY:.0%})   "
          f"{'OK' if accuracy_ok else 'BELOW TARGET'}")
    print(f"Citation rate     {citation_rate:6.1%}")
    print(f"Full answer  p50  {full_p50:6.2f}s  (target <{TARGET_SECONDS:.0f}s)   "
          f"{'OK' if latency_ok else 'OVER TARGET'}")
    print(f"Full answer  p95  {full_p95:6.2f}s")
    print(f"Over {TARGET_SECONDS:.0f}s        {over:4d}/{len(totals)} responses")
    print(f"  ├ retrieval p50 {statistics.median(retrievals):6.2f}s")
    print(f"  ├ first token   {statistics.median(first_tokens):6.2f}s")
    print(f"  └ answer words  {statistics.median(words):6.0f}")
    # The agent's own numbers. These are where the latency goes when this
    # pipeline is slow, and none of them is visible in the timings above.
    # Every turn pays a decision call now, so the question is no longer whether
    # one was avoided but whether it was overlapped: a speculation hit means
    # the retrieval was already finished when the agent chose. A direct reply
    # never wanted retrieval, so it is counted apart rather than scored as a
    # speculation miss.
    print(f"Answered directly {direct / len(totals):6.1%}  ({direct}/{len(totals)}) "
          f"— no retrieval wanted")
    print(f"Speculation hit   {spec_hits / len(totals):6.1%}  ({spec_hits}/{len(totals)}) "
          f"— retrieval overlapped the decision")
    print(f"Agent consulted   {sum(1 for h in hops if h):4d}/{len(hops)} turns")

    # Reported separately, never averaged into the headline. A referential
    # follow-up leans on machinery an opening question never touches, so
    # blending the two would hide a regression in it behind the easier path.
    if follow_up_times:
        fu_over = sum(1 for t in follow_up_times if t >= TARGET_SECONDS)
        print("-" * 68)
        print(
            f"Follow-up acc.    {follow_up_correct / len(follow_up_times):6.1%}  "
            f"({follow_up_correct}/{len(follow_up_times)} multi-turn cases)"
        )
        print(f"Follow-up    p50  {follow_up_p50:6.2f}s  (target <{TARGET_SECONDS:.0f}s)   "
              f"{'OK' if follow_up_ok else 'OVER TARGET'}")
        print(f"Follow-up    p95  {percentile(follow_up_times, 0.95):6.2f}s")
        print(f"Over {TARGET_SECONDS:.0f}s        {fu_over:4d}/{len(follow_up_times)} responses")
    print("=" * 68)

    if not (accuracy_ok and latency_ok and follow_up_ok):
        failed = ", ".join(
            name
            for name, ok in (
                ("accuracy", accuracy_ok),
                ("single-turn latency", latency_ok),
                ("follow-up latency", follow_up_ok),
            )
            if not ok
        )
        print(f"FAILED: {failed}")
        return 2
    return 0


async def _warmup(services) -> None:
    """Exactly what `app.main` warms at startup, so the suite measures what ships.

    The MediaWiki leg is the one that was missing, and it mattered: the
    not-covered case is the only case here that reaches the live API, so it was
    paying a TLS handshake that the served application pays once at startup and
    no user ever sees. That made the suite's slowest case slower than the real
    thing rather than faster — measured at 4.82s cold against 2.38-2.56s warm.
    """
    try:
        vector = await services.retriever.embedder.embed_query("warmup")
        for lang in services.retriever.store.indexed_languages() or {"en": 0}:
            await services.retriever.store.query(vector, top_k=1, lang=lang)
            await services.retriever.store.query(
                vector, top_k=1, lang=lang, section_title="Introduction"
            )
    except Exception:  # noqa: BLE001 - warmup is best effort
        pass

    try:
        await services.retriever.wiki.search("Wikipedia", lang="en", limit=1)
    except Exception:  # noqa: BLE001 - warmup is best effort
        pass


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; `statistics.quantiles` needs n>1 and interpolates."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
