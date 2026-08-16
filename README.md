# Wikipedia Intelligent Chat Assistant

An AI chat application that answers natural-language questions from indexed Wikipedia
content and routes users to the relevant articles — down to the exact section.

Answers are generated **only** from indexed Wikipedia text. Every factual claim carries
a citation marker linking to the section it came from, and the assistant says it does
not know rather than answering from the model's own memory.

**Stack:** Python · FastAPI · LangGraph · ChromaDB · Next.js · OpenAI · MediaWiki API

> **LLM provider.** The spec allows Claude or OpenAI. This build uses OpenAI —
> `gpt-4.1-nano` for both generation and the agent's tool-decision call. The provider is
> isolated to `graph/services.py` and `core/embeddings.py`, so switching means changing
> two modules, not the graph. Model names live in `.env`.

---

## Headline metrics

Measured by `scripts/evaluate.py` on 2026-08-16 against the shipping build and the
1,311-article / 53,818-chunk English index. Twenty single-turn cases (one of which
*must* be refused) plus six multi-turn exchanges, cache cleared per case, driving the
same LangGraph streaming path the UI uses, graded by a `gpt-4.1` judge. Ranges are the
spread across **two consecutive runs**, not a confidence interval.

| Metric | Target | Result |
| --- | --- | --- |
| Answer accuracy | >90% | **100%** |
| Citation rate | — | **100%** |
| Single-turn median | **<3s** | **1.81–1.87s** |
| Single-turn p95 | — | 2.46–2.72s |
| **Single-turn over 3s** | — | **0–1 / 20** |
| Follow-up accuracy | >90% | **100%** (6/6) |
| Follow-up median | **<3s** | **2.22–2.38s** |
| **Follow-up over 3s** | — | **0–1 / 6** |
| Retrieval, median | — | **0.00s** |
| Speculation reuse | — | 18–19 / 20 |
| Median answer length | — | 36–38 words |

The two responses that crossed 3s across those runs are both known paths rather than
random slowness. The **not-covered** case ("current stock price of Tesla") reaches the
live MediaWiki API and came in at 5.63s on one run against 2.13s on the other — it is
the only case that leaves the index, so it is the only one exposed to a second remote
service. The other was a follow-up whose decision call alone took 2,368ms against a
~1.0s median: a stalled request, the failure mode described under *Hedging* below.

```bash
cd backend && python -m scripts.evaluate    # exits non-zero if either gate fails
```

"<3s" is graded on the **complete response**, not time to first token — the harness
reports first-token separately because it explains where time goes, but grading on it
would be scoring an easier exam.

**Read p95 from a 20-case live-API suite with suspicion.** Repeated runs of an identical
build have produced p95 anywhere from 2.3s to 30.7s. Those are stalls inside the OpenAI
call, not work this pipeline does — an embedding of four words was measured at 0.44s and
6.35s twelve seconds apart on the same connection. The p50 and the over-3s count are the
stable signals.

---

## Quick start

Prerequisites: Python 3.11+, Node 20+, an OpenAI API key.

```bash
# 1. Backend
cd backend && python -m venv .venv && .venv\Scripts\activate   # Unix: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # add OPENAI_API_KEY

# 2. Index (~30 articles is enough to demo; --limit 300 for the full corpus)
python -m scripts.ingest --limit 30 --no-categories

# 3. API
uvicorn app.main:app --port 8000          # docs at /docs

# 4. UI, second terminal
cd frontend && npm install && cp .env.local.example .env.local && npm run dev
```

Open <http://localhost:3000>, create an account (any email, 8+ char password — accounts
are local to your SQLite file), and try:

| Ask | To see |
| --- | --- |
| *What is a black hole's event horizon?* | A cited answer, each marker linking to the exact **section** |
| *What happens if you fall into one?* | Multi-turn context — the pronoun resolves against the previous turn |
| *What is the current stock price of Tesla?* | An honest refusal plus routing to real articles |
| *hi* | Small talk answered conversationally, no retrieval in the path |

The crawler is resumable; `--reset` rebuilds, `--force` re-indexes. Add a language with
`python -m scripts.ingest --mirror-langs es,fr,hi --limit 60`.

---

## Requirements coverage

| # | Requirement | Where | Status |
| - | --- | --- | --- |
| 1 | **Web crawling** — official API, various topics | `core/wikipedia/`, `scripts/ingest.py` | Rate-limited, `maxlag`-aware, resumable; 1,311 articles across 6 domains |
| 2 | **Knowledge base** — vector DB, chunking for long articles | `core/vector_store.py`, `core/chunker.py` | Section-aware chunking, 350-token windows, 60-token overlap, per-language collections |
| 3 | **Chat interface** — natural language, multi-turn | `frontend/`, `core/conversation.py` | Next.js UI, SSE streaming, SQLite threads surviving restarts |
| 4 | **RAG pipeline** — accurate, cited responses | `graph/`, `core/prompts.py`, `core/citations.py` | **Agentic**: the model picks its evidence source per turn and may retry against another; MMR retrieval, grounded prompt, citations bound to actually-used sources |
| 5 | **Smart routing** — URLs, languages, disambiguation | `core/retriever.py`, `graph/nodes/` | Section-deep URLs; four editions indexed; disambiguation as a first-class branch with live MediaWiki lookup |
| 6 | **Key metrics** — <3s, >90% accuracy, section citations | `scripts/evaluate.py` | **100% accuracy, 100% citation rate, 1.81–1.87s median, 0–2 of 26 responses over 3s** across two runs |

Legal safety: official API only, descriptive User-Agent, self-throttling, CC BY-SA
attribution via links back to every source article and section.

---

## How it works

A LangGraph `StateGraph` running an agentic loop. The model chooses where each turn's
evidence comes from; the graph caps what that choice may cost.

```
START
  → gate                        (seeds the turn's budget; classifies nothing)
      → agent  ◀────────────────────────────────┐
        ├─ respond_directly                     │ nothing found,
        │      → direct_responder ──────────────│──────────────→ END
        ├─ answer_from_history                  │ budget left
        │      → history_answerer ──────────────│──────────────→ END
        └─ search_kb | search_wiki              │
               → tool_executor ─────────────────┘
                    ├─ ambiguous → clarification ───────────────→ END
                    └─ → generator ─────────────────────────────→ END
```

Every message reaches the agent, and **what a message is** is the agent's first
decision — including whether it wants evidence at all. `respond_directly` is a tool like
any other and carries its own reply, so a greeting is one round trip rather than a
classification call followed by a generation call.

Ingestion is a separate offline pipeline: crawl → section-split → chunk → embed → upsert
into per-language ChromaDB collections, each chunk carrying a deep link
(`…/wiki/Title#Section`).

### The latency design

A turn's cost is two OpenAI round trips — a decision call and a generation call — each
with a ~0.72s floor. Everything else is noise: the entire local RAG path (Chroma
retrieval, MMR, prompt building, citation verification, SQLite) profiles at **~40ms**,
about 2% of a turn. So the work went into removing serial round trips, not into tuning
retrieval.

**Speculative retrieval.** The vector search runs *underneath* the decision call rather
than after it. A referential message embeds to noise alone, so it is stitched to the
previous question — a poor *question* and a good *search* — which carries the subject the
pronoun refers to. Reuse is decided by **containment**: how much of the agent's query the
message we searched already contained. The asymmetry matters — the agent's usual edit is
to *strip* filler ("What is the event horizon of a black hole?" → "event horizon of a
black hole"), and a symmetric measure counts the removed words against the match, scoring
that pair 0.67 and rejecting it. Containment scores it 1.0 while a genuine rewrite still
fails ("what about his wife?" → "Albert Einstein wife" shares only *wife*, 0.33).

**Speculative generation.** The answer is written under the decision call too, buffered
and unflushed, then flushed only when three conditions hold: the decision confirmed
`search_knowledge_base`, the parked search was the one reused, and the buffer is
complete. Measured A/B, interleaved in one process, 27 turns per arm:

| | off | on |
| --- | --- | --- |
| knowledge-base p50 | 2.39s | **1.79s** |
| knowledge-base p95 | 2.61s | 2.47s |
| citation rate | 21/21 | **21/21** |

Outcome split: **67% flushed, 11% discarded, 22% never written.** The discarded slice is
what this costs — a generation paid for and thrown away, because until the decision
resolves we do not know which kind of turn this is. Referential messages never speculate
on the answer at all: the generator phrases those around the agent's *resolved* query,
which does not exist yet, so writing early would answer a question nobody asked.

Disable either with `AGENT_SPECULATIVE_RETRIEVAL` / `AGENT_SPECULATIVE_GENERATION`.

**Hedging the decision call.** A turn was observed spending 10,296ms of its 10,320ms
inside one decision call, with the same node taking 1.6s on the next message. That is a
stalled request, and a plain retry cannot help because it keeps waiting on the stall. Past
`AGENT_HEDGE_AFTER_MS` (1500ms) a second identical request goes out and the first to
finish wins; the payload is `temperature=0` and side-effect free, so the hedge cannot
change which tool is chosen. Its individual contribution to the headline number is
**unmeasured** — it is kept because it is cheap, one config value away from off, and
targets a stall directly observed in the running app.

**The hop budget is a clock, not a counter.** A second search plus generation needs ~1.9s.
Authorising one at 1.2s elapsed lands near 3.1s; at 1.6s it cannot. A hop count cannot
tell those apart.

---

## Design decisions

**Section-aware chunking.** `exsectionformat=wiki` keeps `== Heading ==` markers, so
chunks split along real topic boundaries and each knows its heading path — which is what
makes section-level citations possible. References/External links/See also are dropped.

**Breadcrumb-prefixed embeddings.** Each chunk embeds as
`Article > Section > Subsection\n\n<body>`. Wikipedia prose is pronoun-heavy, so the
breadcrumb is often the only signal tying a chunk to its subject.

**MMR over plain top-k** (λ=0.6). Naive top-k returns five near-identical chunks from one
section; MMR trades a little relevance for coverage across articles.

**A pronoun resolves to the most recent subject, with a deterministic backstop.** Asked
*"who is albert einstein"* → *"his wife"* → *"what about MS Dhoni"* → *"and his wife"*,
the pipeline answered about **Einstein**. Across runs of one identical conversation
`gpt-4.1-nano` produced "MS Dhoni wife" (right), "his wife" (nothing resolved) and
"Albert Einstein wife" (confidently wrong). The last is the dangerous one — it retrieves
real chunks and cites them, so it looks grounded. So on short referential messages the
subject the agent supplied must appear in the question before it; otherwise the previous
question is stitched on deterministically. Prompt wording alone did not fix this.

**"Resolves nothing" and "needed nothing resolved" are different.** Both look identical
to that check, but only one wants stitching. *"what about his wife?"* → `his wife` failed;
*"what about black hole"* → `black hole` added nothing because there was nothing to add.
Stitching the second corrupts it — observed live, a Tesla refusal followed by "what about
black hole" searched for both halves and answered both. A referential pronoun is what
separates them.

**Speculation is not made more permissive than retrieval quality allows.** Plural-tolerant
matching in the reuse check was tried: it turned the one remaining miss on the smoke path
from 0.50 to 0.75 and saved ~0.5s per turn — and turned a cited answer into a refusal on
every run, because the stitched query retrieves general chunks while the discovery history
only surfaces for the agent's own wording. The words were covered; the *chunks* were not.
Backed out, with a regression test.

**The tool menu is filtered by what can apply.** `answer_from_history` is offered only
once there is a conversation. With it always on the menu, `gpt-4.1-nano` picked it for
four plain opening questions the index answers outright. A structural gate cannot be
talked out of; prompt wording could.

**Live Wikipedia is a real tool, not a link list.** Fetched article leads are shaped into
the same `SearchHit` structure and flow through the identical citation-binding and
verification path — a live-sourced claim is checked as strictly as an indexed one.

**Citations reflect actual use**, and **each is checked against the claim it carries.**
Marker bookkeeping only proves a citation points at a *retrieved* chunk, not that the
chunk *supports the sentence*. On an ambiguous query the generator occasionally welds two
subjects together — *"Java is a programming language with four main spoken languages on
the island of Java [2]"* — and every other mechanism passes it. So each cited sentence is
compared with its chunk before sources are bound. The check is deliberately conservative:
a first version also verified verbs and adjectives and rejected 3 of 12 well-grounded
answers.

**Refusals are written by the model, in the caller's language.** This was a table of
templates covering six languages while the picker offered twelve, so a Marathi question
got an English apology — the exact failure the table existed to prevent. A refusal is the
one answer with no facts in it, which is what makes generating it safe.

**Swappable vector store.** ChromaDB sits behind a narrow `add`/`query`/`count`
interface. **SQLite for conversations**, not the vector store — history is ordered,
exact-lookup data, the wrong shape for a similarity index.

---

## Languages

The pipeline is language-agnostic, but an index only answers in the languages it was
built with.

| Edition | Chunks | Status |
| --- | --- | --- |
| `en` | 53,818 | 1,311 articles across six domains |
| `fr` | 5,080 | ~60 articles, mirrored via `langlinks` |
| `es` | 3,578 | 50 articles, mirrored via `langlinks` |
| `hi` | 2,932 | ~60 articles, mirrored via `langlinks` |
| `mr`, `ja` | 0 | collections exist; run `--mirror-langs` to populate |

A question in an indexed language gets a cited answer from that edition's own articles,
in that language (`hi.wikipedia.org/…`). A question in an empty-index language falls
through to live MediaWiki search and is routed to real articles rather than answered.

**The mirrored editions are narrow and it shows at the edges.** Asked
"ताजमहल किसने बनवाया?", the assistant answers *शाहजहां* — correct — but cites the
*उस्मानी साम्राज्य* article, because the Hindi *ताजमहल* article was never mirrored. The
citation verifier passes it, which is the honest limit of a word-overlap check: it can
tell you a citation is unrelated, not that a better source existed and was not indexed.

---

## API

Every route is mounted twice — bare (`/chat`) and under `/api`. `/health`, `/stats` and
`/auth/register|login` are open; everything else needs `Authorization: Bearer <jwt>`, and
routes that spend money additionally need `X-OpenAI-Key`.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/auth/register` · `/auth/login` · `/auth/password` | Accounts; return a JWT |
| `POST` | `/auth/verify-key` | Check an OpenAI key before storing it client-side |
| `POST` | `/chat` | Run one graph turn, full response in one payload |
| `POST` | `/chat/stream` | Same, streamed over SSE (what the UI uses) |
| `GET`/`DELETE` | `/conversations` · `/conversations/{id}` | List, restore, drop threads |
| `GET`/`POST`/`DELETE` | `/kb` · `/kb/ingest` · `/kb/articles` · `/kb/jobs/{id}` | Per-account knowledge base and build jobs |
| `GET` | `/stats` · `/health` | Index size, models, cache; liveness |

`POST /chat` accepts `session_id` (per spec) or `conversation_id` (used by the frontend).

```bash
TOKEN=$(curl -s localhost:8000/api/auth/register -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"choose-a-password"}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

curl -s localhost:8000/api/chat -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" -H "X-OpenAI-Key: $OPENAI_API_KEY" \
  -d '{"message":"What is the event horizon of a black hole?","lang":"en"}'
```

```jsonc
{
  "answer": "A black hole's event horizon is the boundary beyond which nothing, not even light, can escape [1]. …",
  "sources": [{
    "index": 1, "title": "Black hole", "section": "Event horizon",
    "url": "https://en.wikipedia.org/wiki/Black_hole#Event_horizon",
    "snippet": "The defining feature of a black hole is…", "score": 0.61
  }],
  "articles": [{ "title": "Black hole", "url": "https://en.wikipedia.org/wiki/Black_hole" }],
  "timings": { "rewrite_ms": 890, "retrieval_ms": 0, "generation_ms": 1310, "total_ms": 2210 },
  "agent": { "hops": 1, "speculation_hit": true, "speculative_answer_used": true }
}
```

`speculation_hit: true` with `retrieval_ms: 0` is the normal shape of a fast turn — the
search finished while the agent was still choosing. `speculative_answer_used` says whether
the answer was written under the decision call and flushed, or thrown away.

Without `Authorization` the call returns **401**; with a valid token but no key, **428** —
a distinct status so the UI prompts for a key instead of bouncing to the login screen.

**SSE events:** `meta` → `intent` → `rewrite` (resolved query) → `retrieval` (chunk
count) → `token` (repeated) → `done` (the complete `ChatResponse`). Failures arrive as an
`error` event inside the stream.

---

## Access control

The rule is ownership: **you can read the public Wikipedia corpus plus documents you
added, and nothing else.** Enforced in two independent places — each account's documents
live in their own Chroma collections *and* every chunk is stamped with `owner_id`.
Retrieval picks the collection from the token, then re-checks the stamp, so a leak needs
both the wrong collection and a mislabelled chunk. The check runs **before ranking**, so a
denied chunk never reaches the prompt, the citations, or the article links — titles are
data, and `CEO_Compensation_2026` leaks a secret's shape without one word of content.

**Identity is never client-asserted.** The account id comes from a verified signature;
`algorithms` is pinned when decoding, which blocks `alg: none` and confusion downgrades.

**The API key is never stored.** It arrives on `X-OpenAI-Key`, builds a request-scoped
client, and is dropped when the request ends — a database breach exposes no keys because
there are none in it. A shared connection pool (carrying no credentials) keeps per-request
clients affordable; a per-request TLS handshake would not be.

**Password and key are separate concerns.** bcrypt at work factor 12; the account id is a
random uuid, so rotating your key keeps your conversations and knowledge base. An earlier
design derived identity from the key and silently handed you a fresh workspace when it
changed.

Failed logins return one message whether the email is unknown or the password wrong, with
a dummy hash verified in the unknown case so timing does not give it away either.
**Repeated failures are throttled** — five failures in five minutes lock that caller out
for five more, answered as `429`. The counter is keyed on (client, email), not email
alone: keying on the address would let anyone lock a known user out of their own account.

**A prompt instruction is not access control.** "Only use the provided context" is in the
system prompt for grounding quality; if a restricted chunk reaches the model, the boundary
has already failed.

**Per-account knowledge bases.** `wikipedia_en` is the shared corpus; `u<account_id>_en`
holds what that account added. Retrieval queries both and merges on relevance, so a new
user gets useful answers immediately and nobody re-pays to embed what someone else
indexed. Builds run as background jobs, billed to the requesting user's key.

---

## Tests

```bash
cd backend && python -m pytest      # 286 tests, no network or API key required
```

Verified by running with `.env` removed, so a missing key or network cannot make the
suite pass by accident.

`tests/graph/` mirrors `app/graph/nodes/`, one module per node, each tested against a stub
services object that records what it was asked to do. That is how routing is pinned:
`test_a_greeting_reaches_end_without_retrieval_or_generation` asserts the visited-node
list is exactly `["gate", "agent", "direct_responder"]`. Both that and its sibling would
pass just as happily with a stray retrieval in the path if they only checked the answer —
which is why they check the path.

What the model actually *decides* is deliberately not unit-tested: a stub can assert the
graph honours a decision, never that the decision was right. That belongs to
`scripts/evaluate.py`, which runs real turns against the real model.

Failure modes that are easy to regress each get a test pinned to the case that motivated
it: speculation reuse in both directions; the stitched speculation; subject corroboration
in all six directions; no double search after a miss; speculation discarded on every
non-retrieving branch; agent-compressed queries not mistaken for bare entities; question
shape reaching the generator; tool availability; hop-loop termination; cancelled
speculations staying quiet; claim/citation mismatch; disambiguation source; refusal
phrasing; login throttling.

The speculative-generation guards are **mutation-tested** — each was removed in turn to
confirm a test fails. That process found a real bug in the buffer collector, which caught
`CancelledError` and so would have swallowed client-disconnect cancellation.

---

## Wikipedia compliance

- **User-Agent** — descriptive UA with working contact details on every request, per the
  [Wikimedia policy](https://meta.wikimedia.org/wiki/User-Agent_policy). The default
  carries a real repository URL and address, so an unedited checkout is already compliant.
- **`maxlag=5`** — the API sheds our load when replicas fall behind; the client backs off.
- **Self-throttling** — token bucket (8 req/s) plus a concurrency cap; `Retry-After`
  honoured on 429/503.
- **Official API only** — read-only `action=query` calls. No HTML scraping.
- **Attribution** — answers link back to source articles and sections, satisfying CC BY-SA
  in the place users actually read.

---

## Known limitations

**A coverage miss looks exactly like a retrieval bug.** In an earlier ~300-article index,
"Saturn" answered with the equatorial ridges of its moons because no *Saturn* article was
seeded. It read as a ranking problem and was not one — a larger lead-section boost was
tried, measured, and backed out. The fix for a partial miss is indexing the article, not
tuning retrieval. Still true for anything outside the 1,311.

**An uncovered question is misrouted as small talk mid-conversation.** Asked cold, "What
is the current stock price of Tesla?" searches, finds nothing, and routes to real
articles. Asked as the third turn of a conversation, the agent picks `respond_directly`
instead (3/3 reproductions). The refusal is still honest, but no article links are
returned and the turn is recorded as chitchat. The suite cannot see this — its one refusal
case is always turn 1.

**A weak retrieval is never reconsidered.** A second opinion is only sought when the first
search comes back genuinely *empty*, so a question the corpus answers badly is answered
badly from the corpus. Gating the hop on a relevance threshold would fix it and cost
latency; that trade is unmeasured here.

**The agent almost never takes a second hop** — 0/20 on the suite, so the loop is pinned
by unit tests rather than demonstrated by the metrics.

**Latency depends on a connection this project does not control.** Everything above was
measured on a healthy link. Identical four-word embedding calls have been observed at
0.44s and 6.35s twelve seconds apart, and the hedge cannot help when both racers are slow.
p50 and p95 sit comfortably under 3s with margin; a bad network stretch will still breach
it, and no code change prevents that.

**Answers run long on open-ended questions.** "Who is Albert Einstein" produced 114 words
against a prompt asking for 50, and latency is roughly linear in length. Two prompt fixes
were tried and both were backed out: forcing prose made the model open with copular
definitions that citation verification then deleted (accuracy 100% → 95%), and a hard
50-word ceiling dropped a required fact from another case (also 95%). Trading a validated
accuracy number for a latency win that does not show up in the measurement is a bad deal.

**The answer occasionally leaks the word "excerpts".** Rule 7 forbids it, but the prompt
says "excerpts" a dozen times while forbidding it once. Renaming the concept throughout
was tried and backed out — it cost accuracy (100% → 90%) via the same copular-definition
failure. The prompt is in a measured local optimum whose vocabulary is load-bearing in
ways that are not obvious from reading it.

**Citation verification is word overlap, not entailment.** It catches a citation pointing
at unrelated material and a category claim the chunk never makes. It will not catch a
subtly wrong date. Doing that properly needs an NLI model on the response path.

**Disambiguation depends on the term being ambiguous *to the index*.** "Python" answers
about the language rather than offering choices, because the language article outranks the
indexed disambiguation page.

**Sessions and throttling are per-process.** `JWT_SECRET` defaults to a per-process random
value; the login throttle and answer cache are in-memory. Multi-worker deployments need
`JWT_SECRET` set and would want Redis behind both.

---

## Project layout

```
backend/
  app/
    main.py              FastAPI app, CORS, lifespan (compiles graph, warms index)
    config.py            settings (.env)          models.py   request/response schemas
    api/
      dependencies.py    RequestIdentity, require_* dependencies
      routers/           auth · chat · conversations · knowledge_base · system
    graph/
      state.py           ChatState threaded through every node
      build.py           StateGraph wiring, conditional edges, the agent loop
      runner.py          drives one turn; blocking and SSE entry points
      services.py        dependencies nodes call into (stubbable in tests)
      history.py         which turns a decision sees   references.py  referential messages
      nodes/
        gate · agent · speculation · tool_executor · clarification
        history_answerer · generator · direct_responder · timing
    core/
      wikipedia/         MediaWiki client, models, parsing, rate limiting
      chunker.py         section-aware chunking      embeddings.py  batched, backoff
      vector_store.py    ChromaDB adapter            retriever.py   search, MMR, fallback
      prompts.py         system + agent prompts      agent_tools.py tool schemas
      citations.py       sentence-to-chunk checking  sources.py     sources, article links
      refusals.py        not-covered wording         conversation.py SQLite memory
      accounts.py · tokens.py · ratelimit.py · db.py · cache.py
  scripts/               ingest.py · seeds.py · evaluate.py
  tests/                 286 tests; graph/ mirrors app/graph/nodes/

frontend/
  app/                   page.tsx (login or chat), styles/ one sheet per area
  components/            chat/ · auth/ · sources/ · knowledge-base/
  hooks/                 useConversation, useConversationList, useArticleIngest
  lib/                   api/ (one module per resource), types.ts, languages.ts
```
