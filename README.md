# Wikipedia Intelligent Chat Assistant

Chat application that answers natural-language questions from indexed Wikipedia content
and routes users to the relevant articles — down to the exact section.

Answers come **only** from indexed Wikipedia text. Every factual claim carries a citation
marker linking to the section it came from, and the assistant says it does not know
rather than answering from the model's own memory.

**Stack:** Python · FastAPI · LangGraph · ChromaDB · Next.js · OpenAI · MediaWiki API

> The spec allows Claude or OpenAI. This build uses OpenAI (`gpt-4.1-nano` for both
> generation and the agent's tool-decision call), isolated to `graph/services.py` and
> `core/embeddings.py` — switching provider means changing two modules, not the graph.

---

## Metrics

`scripts/evaluate.py`, 2026-08-16, against the 1,311-article / 53,818-chunk English
index. 20 single-turn cases (one *must* be refused) + 6 multi-turn exchanges, cache
cleared per case, graded by a `gpt-4.1` judge. Ranges are the spread across **two
consecutive runs**.

| Metric | Target | Result |
| --- | --- | --- |
| Answer accuracy | >90% | **100%** |
| Citation rate | — | **100%** |
| Single-turn median | **<3s** | **1.81–1.87s** |
| Single-turn p95 | — | 2.46–2.72s |
| Single-turn over 3s | — | 0–1 / 20 |
| Follow-up accuracy | >90% | **100%** (6/6) |
| Follow-up median | **<3s** | **2.22–2.38s** |
| Follow-up over 3s | — | 0–1 / 6 |
| Retrieval, median | — | **0.00s** |
| Speculation reuse | — | 18–19 / 20 |

```bash
cd backend && python -m scripts.evaluate    # exits non-zero if a gate fails
```

"<3s" is graded on the **complete response**, not first token. The two responses that
crossed 3s are known paths, not random: the not-covered case (the only one that leaves
the index for the live API, 5.63s vs 2.13s across runs) and a decision call that stalled
at 2,368ms against a ~1.0s median.

**Treat p95 from a 20-case live-API suite with suspicion.** Identical four-word embedding
calls have been measured at 0.44s and 6.35s twelve seconds apart. The median and the
over-3s count are the stable signals; latency on a degraded connection is not something
this code controls.

---

## Quick start

Python 3.11+, Node 20+, an OpenAI API key.

```bash
# Backend
cd backend && python -m venv .venv && .venv\Scripts\activate   # Unix: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # add OPENAI_API_KEY
python -m scripts.ingest --limit 30 --no-categories   # index; --limit 300 for the full corpus
uvicorn app.main:app --port 8000          # docs at /docs

# Frontend, second terminal
cd frontend && npm install && cp .env.local.example .env.local && npm run dev
```

Open <http://localhost:3000>, create an account (any email, 8+ char password), and try:

| Ask | To see |
| --- | --- |
| *What is a black hole's event horizon?* | A cited answer, each marker linking to the exact **section** |
| *What happens if you fall into one?* | Multi-turn context — the pronoun resolves against the previous turn |
| *What is the current stock price of Tesla?* | An honest refusal plus routing to real articles |
| *hi* | Small talk answered conversationally, no retrieval in the path |

The crawler is resumable; `--reset` rebuilds, `--force` re-indexes. Add a language with
`--mirror-langs es,fr,hi --limit 60`.

---

## Requirements coverage

| # | Requirement | Where | Status |
| - | --- | --- | --- |
| 1 | Web crawling, official API | `core/wikipedia/`, `scripts/ingest.py` | Rate-limited, `maxlag`-aware, resumable; 1,311 articles / 6 domains |
| 2 | Knowledge base, chunking | `core/vector_store.py`, `core/chunker.py` | Section-aware, 350-token windows, 60-token overlap, per-language collections |
| 3 | Chat interface, multi-turn | `frontend/`, `core/conversation.py` | Next.js UI, SSE streaming, SQLite threads surviving restarts |
| 4 | RAG pipeline, cited responses | `graph/`, `core/citations.py` | **Agentic**: the model picks its evidence source per turn and may retry another; MMR retrieval, citations bound to actually-used sources |
| 5 | Smart routing, languages, disambiguation | `core/retriever.py`, `graph/nodes/` | Section-deep URLs; 4 editions indexed; disambiguation as a first-class branch with live MediaWiki lookup |
| 6 | Key metrics | `scripts/evaluate.py` | See above — both gates met on both paths |

Indexed editions, each answering from its own articles in its own language:
`en` 53,818 chunks · `fr` 5,080 · `es` 3,578 · `hi` 2,932 · `mr`/`ja` plumbed but empty
(`--mirror-langs` populates them). The mirrored editions carry ~60 articles against
English's 1,311, so a question just outside the mirrored set is grounded on whatever is
nearest rather than on the right article.

---

## How it works

A LangGraph `StateGraph` running an agentic loop. The model chooses where each turn's
evidence comes from; the graph caps what that choice may cost.

```
START → gate → agent ◀──────────────────────┐
                ├─ respond_directly    ──────│──→ END
                ├─ answer_from_history ──────│──→ END   nothing found,
                └─ search_kb | search_wiki   │          budget left
                       → tool_executor ──────┘
                            ├─ ambiguous → clarification → END
                            └─ → generator ──────────────→ END
```

Every message reaches the agent, and **what a message is** is the agent's first decision.
`respond_directly` is a tool like any other and carries its own reply, so a greeting is
one round trip rather than a classification call plus a generation call.

### The latency design

A turn is two OpenAI round trips — decision and generation — each with a ~0.72s floor.
Everything else is noise: the entire local RAG path (Chroma, MMR, prompt building,
citation verification, SQLite) profiles at **~40ms**, ~2% of a turn. So the work went
into removing serial round trips, not tuning retrieval.

**Speculative retrieval** runs the vector search *underneath* the decision call. A
referential message embeds to noise alone, so it is stitched to the previous question — a
poor *question*, a good *search*. Reuse is decided by **containment**, not symmetric
overlap: the agent's usual edit is to strip filler ("What is the event horizon of a black
hole?" → "event horizon of a black hole"), which a symmetric measure scores 0.67 and
rejects; containment scores it 1.0 while a genuine rewrite still fails ("what about his
wife?" → "Albert Einstein wife" shares only *wife*, 0.33).

**Speculative generation** writes the answer under the decision call too, buffered, and
flushes it only when the decision confirmed `search_knowledge_base`, the parked search was
the one reused, and the buffer is complete. A/B, interleaved, 27 turns per arm:

| | off | on |
| --- | --- | --- |
| knowledge-base p50 | 2.39s | **1.79s** |
| p95 | 2.61s | 2.47s |
| citation rate | 21/21 | **21/21** |

Outcome: **67% flushed, 11% discarded, 22% never written.** The discarded slice is the
cost — a generation paid for and thrown away, because until the decision resolves we do
not know which kind of turn this is. Referential messages never speculate on the answer:
the generator phrases those around the agent's *resolved* query, which does not exist yet.

**Hedging.** A turn was observed spending 10,296ms of its 10,320ms inside one decision
call, the same node taking 1.6s next message. Past `AGENT_HEDGE_AFTER_MS` a second
identical request goes out and the first to finish wins; `temperature=0` and side-effect
free, so it cannot change which tool is chosen. Its individual contribution is
**unmeasured** — kept because it is cheap and targets a directly observed stall.

**The hop budget is a clock, not a counter.** A second search plus generation needs ~1.9s;
authorising one at 1.2s elapsed lands near 3.1s, at 1.6s it cannot.

---

## Design decisions

- **Section-aware chunking** — `exsectionformat=wiki` keeps `== Heading ==` markers, so
  chunks split on real boundaries and each knows its heading path. That is what makes
  section-level citations possible.
- **Breadcrumb-prefixed embeddings** — `Article > Section\n\n<body>`. Wikipedia prose is
  pronoun-heavy, so the breadcrumb is often the only signal tying a chunk to its subject.
- **MMR over top-k** (λ=0.6) — plain top-k returns five near-identical chunks from one
  section.
- **Pronoun resolution is checked, not trusted.** Across runs of one conversation
  `gpt-4.1-nano` produced "MS Dhoni wife" (right), "his wife" (unresolved) and "Albert
  Einstein wife" (confidently wrong). The last is dangerous — it retrieves real chunks and
  cites them, so it looks grounded. On short referential messages the supplied subject
  must appear in the previous question, or it is stitched on deterministically.
- **"Resolves nothing" ≠ "needed nothing resolved."** Stitching the second corrupts it:
  observed live, a Tesla refusal followed by "what about black hole" searched for both
  halves and answered both.
- **Citations are checked against the claim they carry.** Marker bookkeeping only proves a
  citation points at a *retrieved* chunk. Each cited sentence is compared with its chunk
  before sources are bound — deliberately conservative, since a stricter first version
  rejected 3 of 12 well-grounded answers.
- **Refusals are model-written, in the caller's language.** A template table covered six
  languages while the picker offered twelve, so a Marathi question got an English apology.
- **Swappable vector store** behind a narrow `add`/`query`/`count` interface; **SQLite for
  conversations**, since history is ordered exact-lookup data, not a similarity problem.

Two experiments that were **measured and backed out**, recorded because the reasoning
generalises: making the speculation reuse check plural-tolerant saved ~0.5s and turned a
cited answer into a refusal every run (the words were covered; the chunks were not); and
forcing prose over bullets, or a hard 50-word cap, each cost accuracy (100% → 95%) by
producing copular definitions that citation verification then deleted.

---

## API

Routes are mounted bare and under `/api`. `/health`, `/stats` and `/auth/*` are open;
the rest need `Authorization: Bearer <jwt>`, and money-spending routes also need
`X-OpenAI-Key` (**never stored** — request-scoped client, dropped when the request ends).

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/auth/register` · `/login` · `/password` · `/verify-key` | Accounts, JWT |
| `POST` | `/chat` · `/chat/stream` | One graph turn; SSE variant is what the UI uses |
| `GET`/`DELETE` | `/conversations[/{id}]` | List, restore, drop threads |
| `GET`/`POST`/`DELETE` | `/kb` · `/kb/ingest` · `/kb/articles` · `/kb/jobs/{id}` | Per-account knowledge base |
| `GET` | `/stats` · `/health` | Index size, models; liveness |

```jsonc
{
  "answer": "A black hole's event horizon is the boundary beyond which nothing … escape [1].",
  "sources": [{ "index": 1, "title": "Black hole", "section": "Event horizon",
                "url": "https://en.wikipedia.org/wiki/Black_hole#Event_horizon", "score": 0.61 }],
  "timings": { "rewrite_ms": 890, "retrieval_ms": 0, "generation_ms": 1310, "total_ms": 2210 },
  "agent": { "hops": 1, "speculation_hit": true, "speculative_answer_used": true }
}
```

`speculation_hit: true` with `retrieval_ms: 0` is the shape of a fast turn — the search
finished while the agent was still choosing. No `Authorization` → **401**; valid token but
no key → **428**, distinct so the UI prompts for a key instead of bouncing to login.

**SSE:** `meta` → `intent` → `rewrite` → `retrieval` → `token`… → `done`. Failures arrive
as an `error` event inside the stream.

---

## Access control

**You can read the public corpus plus documents you added, and nothing else** — enforced
twice. Each account's documents live in their own Chroma collections *and* every chunk is
stamped with `owner_id`; retrieval picks the collection from the token, then re-checks the
stamp. The check runs **before ranking**, so a denied chunk never reaches the prompt, the
citations or the article links — titles are data, and `CEO_Compensation_2026` leaks a
secret's shape without one word of content.

Identity is never client-asserted (verified signature, `algorithms` pinned when decoding).
Password and key are separate concerns: bcrypt work factor 12, account id a random uuid,
so rotating your key keeps your conversations. Failed logins return one message either
way, with a dummy hash verified on unknown emails so timing does not leak; five failures
in five minutes lock that caller out, keyed on (client, email) so nobody can lock a known
user out of their own account.

---

## Tests

```bash
cd backend && python -m pytest      # 286 tests, no network or API key required
```

`tests/graph/` mirrors `app/graph/nodes/`, one module per node, each tested against a stub
that records what it was asked to do. That is how routing is pinned:
`test_a_greeting_reaches_end_without_retrieval_or_generation` asserts the visited-node
list is exactly `["gate", "agent", "direct_responder"]` — it would pass just as happily
with a stray retrieval if it only checked the answer, which is why it checks the path.

What the model *decides* is deliberately not unit-tested — a stub can assert the graph
honours a decision, never that the decision was right. That belongs to `evaluate.py`.

The speculative-generation guards are **mutation-tested**: each was removed in turn to
confirm a test fails. That found a real bug in the buffer collector, which caught
`CancelledError` and would have swallowed client-disconnect cancellation.

---

## Wikipedia compliance

Descriptive User-Agent with working contact details per the
[Wikimedia policy](https://meta.wikimedia.org/wiki/User-Agent_policy) · `maxlag=5` ·
self-throttling token bucket (8 req/s) with `Retry-After` honoured · read-only
`action=query` calls only, no scraping · CC BY-SA attribution via links back to every
source article and section.

---

## Layout

```
backend/
  app/
    main.py                  FastAPI app, CORS, lifespan (compiles graph, warms index)
    config.py                settings loaded from .env
    models.py                request/response schemas
    api/
      dependencies.py        RequestIdentity, require_identity / require_openai
      routers/
        auth.py              register, login, password change, key validation
        chat.py              session bootstrap, blocking chat, SSE stream
        conversations.py     list, restore, clear threads
        knowledge_base.py    inspect, ingest, add articles, job status
        system.py            stats, health
    graph/
      state.py               ChatState threaded through every node
      build.py               StateGraph wiring, conditional edges, the agent loop
      runner.py              drives one turn; blocking and SSE entry points
      services.py            dependencies nodes call into (stubbable in tests)
      history.py             which turns a decision sees; small talk filtered out
      references.py          whether a message leans on the turn before it
      nodes/
        gate.py              seeds the turn's budget
        agent.py             picks the tool for this hop, resolves the subject
        speculation.py       retrieval and generation raced against the decision
        tool_executor.py     runs the tool, shapes results, the hop gate
        clarification.py     asks which entity was meant
        history_answerer.py  answers from the conversation itself
        generator.py         grounded generation, citations, buffer flush
        direct_responder.py  emits the agent's own reply, no second call
        timing.py            per-node latency instrumentation
    core/
      wikipedia/
        client.py            MediaWiki API client
        models.py            article and section models
        parsing.py           section splitting, disambiguation detection
        rate_limit.py        token bucket, maxlag, Retry-After
      chunker.py             section-aware chunking with token overlap
      embeddings.py          batched OpenAI embeddings with backoff
      vector_store.py        ChromaDB adapter, per-language collections
      retriever.py           semantic search, MMR, disambiguation, live fallback
      prompts.py             system and agent prompts, context block, languages
      agent_tools.py         tool schemas, tool names, tool-call parsing
      citations.py           holds each sentence to the chunk it cites
      sources.py             cited chunks to sources and article links
      refusals.py            subject extraction and not-covered wording
      knowledge_base.py      per-account ingest jobs
      accounts.py            registration, sign-in, password hashing
      tokens.py              session token issue and verification
      ratelimit.py           failed-login throttling
      conversation.py        multi-turn memory, SQLite-persisted, TTL eviction
      db.py                  one shared SQLite connection behind one lock
      cache.py               TTL answer cache
  scripts/
    ingest.py                crawler + indexer CLI
    seeds.py                 seed topics and categories
    evaluate.py              accuracy + latency benchmark
  tests/                     286 tests, no network or API key needed
    graph/                   stubs.py, plus one module per node and per mechanism:
                             test_gate · test_agent · test_agent_tools
                             test_speculation · test_speculative_generation
                             test_hedged_decision · test_tool_executor
                             test_hop_gate · test_clarification · test_generator
                             test_history_answerer · test_direct_responder
                             test_refuse_and_route · test_references · test_routing
    test_accounts.py         registration, sign-in, password hashing, migration
    test_rbac.py             ownership rules and fail-closed access checks
    test_ratelimit.py        login throttling, expiry, per-account isolation
    test_wikipedia.py        MediaWiki client against a mocked API
    test_chunker.py          section splitting, overlap, token budgets
    test_citations.py        claim-versus-chunk verification
    test_sources.py          cited chunks to sources and article links
    test_refusals.py         subject extraction and not-covered wording
    test_retriever.py        MMR and relevance filtering
    test_conversations.py    persistence across restarts, TTL eviction
    test_cache.py            answer cache keys and expiry
    test_config.py           settings loading and validation
    factories.py             shared fixtures
  requirements.txt · pytest.ini · .env.example

frontend/
  app/
    page.tsx                 login screen or chat screen
    layout.tsx               root layout
    globals.css              imports styles/ in order
    styles/                  tokens · layout · messages · composer · sidebar
                             sources · source-panel · answer-footer · login
                             api-key-bar · articles · disambiguation · empty
                             modal · responsive
  components/
    chat/                    ChatScreen · ChatHeader · MessageList · MessageBubble
                             AnswerText · Composer · Sidebar · EmptyState
                             DisambiguationList
    auth/                    LoginScreen · ApiKeyBar
    sources/                 SourcePanel
    knowledge-base/          KnowledgeBasePanel
  hooks/                     useConversation · useConversationList · useArticleIngest
  lib/
    api/                     client · auth · chat · conversations · knowledgeBase
    types.ts · credentials.ts · languages.ts
  next.config.mjs · tsconfig.json · package.json · package-lock.json · .env.local.example
```
