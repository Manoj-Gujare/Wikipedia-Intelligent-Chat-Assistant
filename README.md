# wikipedia intelligent chat assistant

An AI-powered chat application that answers natural-language questions using Wikipedia
content and routes users to the relevant Wikipedia articles — down to the exact section.

Answers are generated only from indexed Wikipedia text. Every factual claim carries a
citation marker that links to the section it came from, and the assistant says it does
not know rather than answering from the model's own memory.

**Stack:** Python · FastAPI · LangGraph · ChromaDB · Next.js · OpenAI · MediaWiki API

> **LLM provider.** The specification allows either the Claude API or OpenAI. This
> build uses OpenAI — `gpt-4.1-nano` for both generation (`gpt-4.1-mini` is a
> one-env-var switch) and the agent's tool-decision call. The provider is isolated
> to `graph/services.py` and `core/embeddings.py`, so moving to Claude means
> changing those two modules, not the graph. Model names live in `.env`
> (`CHAT_MODEL`, `AGENT_MODEL`, `EMBEDDING_MODEL`).

---

## Requirements coverage

Each core requirement from the assignment, and where it is implemented.

| # | Requirement | Where | Status |
| - | ----------- | ----- | ------ |
| 1 | **Web crawling** — official MediaWiki API, various topics/categories | `core/wikipedia.py`, `scripts/ingest.py`, `scripts/seeds.py` | Rate-limited, `maxlag`-aware, resumable; ~300 articles across 6 domains |
| 2 | **Knowledge base** — vector DB, chunking strategy for long articles | `core/vector_store.py` (ChromaDB), `core/chunker.py` | Section-aware chunking, 350-token windows, 60-token overlap, per-language collections |
| 3 | **Chat interface** — natural language, multi-turn context and history | `frontend/`, `core/conversation.py` | Next.js chat UI, SSE streaming, SQLite-persisted threads that survive restarts |
| 4 | **RAG pipeline** — accurate, cited responses from Wikipedia | `graph/` (LangGraph), `core/rag.py`, `core/retriever.py` | **Agentic**: the model picks its evidence source per turn (indexed corpus / live MediaWiki / the conversation) and may retry against a different one; MMR retrieval, grounded prompt, citations bound to actually-used sources |
| 5 | **Smart routing** — article URLs, multiple languages, disambiguation | `core/retriever.py`, `graph/nodes.py` | Section-deep URLs; **English + Spanish indexed** (9,134 + 3,578 chunks), other editions plumbed but unpopulated — see [Languages](#languages); disambiguation as a first-class branch, with a live MediaWiki lookup when the index has no disambiguation page |
| 6 | **Key metrics** — <3s, >90% accuracy, section-linked citations | `scripts/evaluate.py` | **1.65–1.90s median single-turn, 2.49–2.92s median follow-up / 100% accuracy / 100% citation rate** — ranges are the spread across repeated runs against a live API; see [Metrics](#metrics) |

Legal safety is addressed in [Wikipedia compliance](#wikipedia-compliance): official
API only, descriptive User-Agent, self-throttling, and CC BY-SA attribution via links
back to every source article and section.

---

## Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [API](#api)
- [Design decisions](#design-decisions)
- [Languages](#languages)
- [Metrics](#metrics)
- [Tests](#tests)
- [Wikipedia compliance](#wikipedia-compliance)
- [Known limitations](#known-limitations)
- [Project layout](#project-layout)

---

## How it works

Orchestration is a **LangGraph `StateGraph`** running an agentic loop. The model
chooses where each turn's evidence comes from; the graph decides what that choice
is allowed to cost:

```
START
  → gate                        (seeds the turn's budget; classifies nothing)
      → agent  ◀────────────────────────────────┐
        │                                       │ nothing found,
        ├─ respond_directly                     │ budget left
        │      → direct_responder ──────────────│───────────────────────→ END
        ├─ answer_from_history                  │
        │      → history_answerer ──────────────│───────────────────────→ END
        └─ search_kb | search_wiki              │
               → tool_executor ─────────────────┘
                    ├─ ambiguous → clarification ─────────────────────→ END
                    └─ → generator ───────────────────────────────────→ END
```

Every message reaches the agent, and **what a message is** is the agent's first
decision — including whether it wants evidence at all. `respond_directly` is a
tool like any other, and it carries its own reply, so a greeting is one round
trip rather than a classification call followed by a generation call.

Two things stay in Python, because handing them to the model costs latency it
cannot earn back:

* **retrieval starts before the agent has decided**, racing the decision call
  rather than queueing behind it — this is what pays for the uniformity above,
* **a second hop is authorised by the wall clock**, not by the agent's appetite
  for one.

That is the design: the model judges what the turn *needs*, and the graph caps
what that judgement is allowed to cost.

Ingestion is a separate offline pipeline:

```
   MediaWiki API ───crawl───▶  scripts/ingest.py
   (rate-limited, UA-compliant) fetch → section-split → chunk → embed → upsert
                                            ▼
                                ChromaDB (per-language collections, cosine)
```

**Ingestion** walks curated seed topics and MediaWiki categories, pulls each article's
plain-text extract with its `== Section ==` markers intact, splits along real section
boundaries, packs paragraphs into overlapping token windows, and stores each chunk with
a deep link (`…/wiki/Title#Section`) in ChromaDB.

### Why greetings bypass retrieval

"hi" carries no information need, but a naive RAG pipeline treats it like any other
input: embed it (~300ms API call), search the vector index, stuff six irrelevant chunks
into a prompt, and generate. Measured on this app before routing existed, "Hey" took
**14.9s** and returned article links to *HTTP*, *Aporia (company)* and *missile lofting* —
whatever a greeting happens to sit near in embedding space. Confidently wrong, slowly.

The first fix was a phrase table: greeting/thanks/farewell/ack phrasings plus a few
"who are you"-style regexes, matched before the agent ever ran. It made "hi" cost
16ms, and it was wrong in the way every keyword classifier is wrong — it could only
recognise the phrasings someone had thought to write down. **"Do you know me?"** was
on no list, so it was embedded, searched for in an encyclopedia, and answered with
"I don't have anything about that in the indexed articles." The user's question
about the assistant came back as a failed lookup about the user.

So classification is the agent's job now, and `respond_directly` is a tool
alongside the two searches:

| Case | What the agent does | Cost |
| --- | --- | --- |
| Small talk, or a question about the assistant | `respond_directly`, writing the reply into the tool call | one call, ~1.1s |
| Question | `search_knowledge_base`, with retrieval already running underneath | one call; the search is free |
| Follow-up | same, with the reference resolved into the query | one call |

The reply rides on the tool call rather than coming from a second generation call.
That is what keeps the cheapest turn in the app to a single round trip, and it is
also why no reply wording lives in source code: the model writes it per turn, in
the user's language, with the conversation in view.

**The trade, stated plainly.** A greeting went from **~16ms to ~1.1s**. That is
the price of answering messages nobody enumerated in advance, and it stays well
inside the 3s budget. Questions are unaffected — they were already paying for a
decision call or about to.

### What a greeting is *not*

Small talk is still excluded from the history that follow-ups resolve against:
stitching "what about his wife?" onto "Hey" produces a search for a greeting and a
pronoun. But that verdict can no longer be recomputed on every history read, since
the classifier is now an API call.

It is recorded instead. When a turn is written, the agent's decision is persisted
on the user turn (`meta.chitchat`), and `substantive_turns` reads the flag. Each
turn is classified exactly once, by the model that was already deciding, and the
hot path stays a list comprehension.

### Speculative retrieval

Follow-ups still consult the agent — resolving "what happens at its event
horizon?" is exactly the judgement worth paying for — so the decision call is back
on the critical path there. A vector search takes about a third of the time that
call does, so it runs *underneath* it rather than after it.

```
t=0.00  ├─ agent decides                         ─────────────────▶ ~1.1s
        └─ retrieve("What is a black hole?
                     What happens at its event horizon?") ──▶ 0.34s  (parked)

t=1.10  agent asks for search_kb("black hole event horizon")
        query already covered by what we searched?  →  reuse, 0ms
        query materially different?                →  search again, +0.34s
```

The stitched query is the trick. A referential message embeds to noise on its own,
so an earlier version skipped speculation for exactly the turns that needed it
most. Joined to the question before it, it carries the subject the pronoun refers
to — a poor *question* and a perfectly good *search*. The agent's own resolved
query is then contained in it, so the parked result is accepted. **6 of 6
follow-ups now reuse their speculation**, where none could before.

Reuse is decided by *containment*: how much of the agent's query the message we
searched already contained. The asymmetry matters. What the agent does to a
standalone question is strip it — "What is the event horizon of a black hole?"
comes back as "event horizon of a black hole" — and a symmetric measure counts the
removed filler against the match, scoring that pair 0.67. The first implementation
used Jaccard at a 0.7 threshold and hit **0 of 3** real questions in the smoke
suite. Containment scores the same pair 1.0, while a genuine rewrite still fails:
"what about his wife?" → "Albert Einstein wife" shares only *wife*, or 0.33.

The threshold sits in a measured gap rather than on either population's edge —
rewrites that should reuse scored 0.67–1.00, rewrites that must not scored
0.00–0.50, with nothing in between.

### Why the hop budget is a clock, not a counter

A second search plus generation needs roughly 1.9s. Authorising one at 1.2s elapsed
lands near 3.1s; authorising one at 1.6s cannot. A hop *count* cannot tell those
apart — it would let a slow first call spend a budget the turn has already overrun.
`agent_max_hops` still exists as a backstop, but the clock is what actually decides.

### Node count

Seven nodes, each removing work from some path: the gate avoids the agent,
speculation avoids waiting for it, clarification avoids answering the wrong
"Mercury", `history_answerer` avoids searching for something already said. A
planner/critic/researcher swarm would add hops and latency without changing a
single answer.

---

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 20+
- An OpenAI API key (used for both chat completions and embeddings)

### 1. Backend

```bash
cd backend

python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env        # then put your OPENAI_API_KEY in .env
```

### 2. Build the knowledge base

```bash
# ~300 English articles across science, history, tech, geography, arts and people.
# Takes a few minutes and costs a few cents in embeddings.
python -m scripts.ingest --limit 300

# Faster smoke test:
python -m scripts.ingest --limit 30 --no-categories

# Add another language edition (titles mapped via Wikipedia's own langlinks).
# The validated index was built with es; fr/hi/mr work the same way.
python -m scripts.ingest --mirror-langs es --limit 50
```

The crawler is resumable — re-running skips articles already indexed. Use `--reset` to
rebuild from scratch and `--force` to re-index everything.

### 3. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive docs at <http://localhost:8000/docs>.

### 4. Run the UI

```bash
cd frontend
npm install
cp .env.local.example .env.local     # points at http://localhost:8000
npm run dev
```

Open <http://localhost:3000>.

---

## API

Every route is mounted twice — bare (`/chat`, `/health`) and under `/api`. Both are
the same handler. `/health`, `/stats` and the `/auth/register`|`/auth/login` pair are
open; everything else needs a `Authorization: Bearer <jwt>` header, and the routes that
spend money additionally need `X-OpenAI-Key` — see
[Accounts](#accounts-and-credentials).

| Method   | Path                  | Purpose                                        |
| -------- | --------------------- | ---------------------------------------------- |
| `POST`   | `/auth/register`      | Create an account (email + password), returns a JWT |
| `POST`   | `/auth/login`         | Sign in, returns a JWT                          |
| `POST`   | `/auth/password`      | Change password (requires the current one)      |
| `POST`   | `/auth/verify-key`    | Check an OpenAI key works before storing it client-side |
| `POST`   | `/session`            | Validate credentials, return the account's state |
| `POST`   | `/kb/articles`        | Index specific article titles on demand         |
| `GET`    | `/conversations`      | Sidebar list of this account's threads          |
| `GET`    | `/kb`                 | Knowledge-base summary, suggestions, active job |
| `POST`   | `/kb/ingest`          | Queue a background build (billed to the caller) |
| `GET`    | `/kb/jobs/{id}`       | Build progress                                  |
| `POST`   | `/kb/jobs/{id}/cancel`| Stop a running build                            |
| `DELETE` | `/kb`                 | Remove everything this account added            |

| Method   | Path                            | Purpose                                             |
| -------- | ------------------------------- | --------------------------------------------------- |
| `POST`   | `/chat`                         | Run one graph turn, full response in one payload     |
| `POST`   | `/chat/stream`                  | Same, streamed over SSE (what the UI uses)           |
| `GET`    | `/conversations/{id}`           | Stored history (with citations) to resume a thread   |
| `DELETE` | `/conversations/{id}`           | Drop a conversation's history                        |
| `GET`    | `/stats`                        | Index size, languages, models, cache stats           |
| `GET`    | `/health`                       | Liveness + whether the index and API key are ready   |

`POST /chat` accepts `session_id` (per spec) or `conversation_id` (used by the
frontend) — they are interchangeable.

### Example

`/chat` needs both credentials: the JWT says *who you are*, the key says *what
you spend*. Register once (or `POST /auth/login` if you already have an
account), then pass the token and your key on the chat call.

```bash
# 1. Get a session token.
TOKEN=$(curl -s localhost:8000/api/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email": "you@example.com", "password": "choose-a-password"}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

# 2. Ask a question.
curl -s localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-OpenAI-Key: $OPENAI_API_KEY" \
  -d '{"message": "What is the event horizon of a black hole?", "lang": "en"}'
```

Without the `Authorization` header the call returns **401**; with a valid token
but no `X-OpenAI-Key` it returns **428** (the session is fine, it just needs a
key). `/health` and `/stats` are open and need neither.

```jsonc
{
  "conversation_id": "5f3c…",
  "answer": "A black hole's event horizon is the boundary beyond which nothing, not even light, can escape [1]. …",
  "sources": [
    {
      "index": 1,
      "title": "Black hole",
      "section": "Event horizon",
      "url": "https://en.wikipedia.org/wiki/Black_hole#Event_horizon",
      "article_url": "https://en.wikipedia.org/wiki/Black_hole",
      "snippet": "The defining feature of a black hole is the appearance of an event horizon…",
      "score": 0.61
    }
  ],
  "articles": [{ "title": "Black hole", "url": "https://en.wikipedia.org/wiki/Black_hole", "lang": "en" }],
  "timings": { "rewrite_ms": 890, "retrieval_ms": 0, "generation_ms": 1310, "total_ms": 2210 },
  "agent": {
    "tools": [{ "tool": "search_knowledge_base", "query": "event horizon of a black hole", "hop": 0 }],
    "hops": 1,
    "speculation_hit": true
  }
}
```

`agent` reports what the turn actually planned. `speculation_hit: true` with
`retrieval_ms: 0` is the normal shape of a fast turn — the search finished while the
agent was still choosing. `hops > 1` means the first source came back empty and the
agent tried another one within the budget.

Pass the returned `conversation_id` on the next request to continue the thread.

### SSE events

`/chat/stream` emits: `meta` (session id) → `intent` (which graph branch was taken; the
UI uses this to suppress the thinking indicator for instant chitchat replies) →
`rewrite` (resolved query, if the question was rewritten) → `retrieval` (chunk count) →
`token` (repeated) → `done` (the complete `ChatResponse`). Failures arrive as an `error`
event inside the stream.

---

## Access control

The rule is ownership: **you can read the public Wikipedia corpus plus documents you
added, and nothing else.** It is enforced in two independent places.

```
POST /auth/register|login  {email, password}  →  signed JWT {sub, role, exp}
                                          │
every request:  Authorization: Bearer <jwt>   ← who you are (verified signature)
                X-OpenAI-Key:  sk-…           ← what you spend (never stored)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  │ 1. collection chosen from the token's account │
                  │ 2. owner_id on each chunk re-checked          │
                  └───────────────────────────────────────────────┘
```

**Password, then key — two different things.** Signing in needs an email and
password (bcrypt, work factor 12). The OpenAI key is a *runtime* credential entered
from the bar at the top of the app, so it can be added, replaced or removed at any
time without touching the account. Endpoints that spend money answer **428** when no
key is present — a distinct status from 401, so the UI prompts for a key instead of
bouncing the user back to the login screen. Listing conversations never demands one.

**The account id is a random uuid**, unrelated to email or key. Rotating your API key
keeps your conversations and knowledge base; an earlier design derived identity from
the key and silently gave you a fresh workspace when it changed.

Failed logins return one message whether the email is unknown or the password is
wrong, with a dummy hash verified in the unknown-email case so timing does not give
it away either — otherwise the login form is an account enumerator.

**Repeated failures are throttled.** bcrypt makes each guess expensive for the server,
which is the point, but nothing in the hash cost stops an attacker issuing guesses
concurrently — so five failures inside five minutes lock that caller out for five more,
answered as `429` with `Retry-After` rather than the `401` a single wrong password
gets. The counter is keyed on (client, email) rather than email alone: keying on the
address would let anyone lock a known user out of their own account by failing logins
on their behalf, turning the defence into a denial of service.

**Identity is never client-asserted.** The account id comes out of a verified
signature, so no header, body field or query string can claim to be another account
or a different role. `algorithms` is pinned to one value when decoding, which is what
blocks the `alg: none` and algorithm-confusion downgrades.

**The API key is not in the token.** A JWT is signed, not encrypted — its payload is
base64 and readable by anyone holding it. Putting a spendable credential there would
publish it, so the key travels on its own header and is never persisted.

**Two independent checks, not one.** Each account's documents live in their own Chroma
collections *and* every chunk is stamped with `owner_id`. Retrieval picks the
collection from the token, then re-checks the stamp. A leak needs both the wrong
collection and a mislabelled chunk. Measured cost drove this: metadata filters ran
250–950ms against ~15ms unfiltered on 9,134 chunks, so the fast structural boundary
carries the load and the metadata check is the backstop.

**The check runs before ranking, not at answer time.** A denied chunk never reaches
the prompt — and equally never reaches the citations, the article links, or the
disambiguation list. Titles are data: `CEO_Compensation_2026` leaks the secret's
existence and shape without one word of content.

**A prompt instruction is not access control.** "Only use the provided context" is in
the system prompt for grounding quality. It is not load-bearing for security: if a
restricted chunk reaches the model, the boundary has already failed, and no wording
reliably prevents its use.

Unowned chunks are public Wikipedia content and readable by all; anything with an
owner that does not match fails closed.

### Accounts and credentials

Users register with an email and password, then supply their own OpenAI API key as a
separate runtime credential. Two properties are worth stating plainly, because both
are deliberate:

**The server never stores the API key.** It arrives on the `X-OpenAI-Key` header,
builds a request-scoped OpenAI client, and is dropped when the request ends. Nothing
is written to disk, logged, or cached between requests — a breach of the database
exposes no keys, because there are none in it. A shared HTTP connection pool (carrying
no credentials of its own) keeps this affordable: per-request clients are cheap, a
per-request TLS handshake would not be.

**Password and key are separate concerns.** The password authenticates you (bcrypt,
work factor 12) and the account id is a random uuid, so rotating your API key keeps
your conversations and knowledge base intact. An earlier design derived the account id
from `sha256(email + api_key)`, which silently handed you a fresh empty workspace the
moment you rotated your key; `AccountStore._migrate` moves rows from that scheme aside
rather than dropping them.

Ownership is enforced server-side, not just hidden in the UI — a conversation id is
useless to anyone but its owner (`403`), and job ids are scoped the same way.

### Per-account knowledge bases

Chroma collections are namespaced. `wikipedia_en` is the shared corpus every account
can search; `u<account_id>_en` holds what that account added. Retrieval queries both
and merges on relevance, so:

* a new user gets useful answers immediately, off the shared corpus;
* nobody re-pays to embed articles someone else already indexed;
* private additions are visible only to the account that built them.

Builds run as background jobs with polled progress, because ingesting a few hundred
articles takes minutes. Embedding is billed to the requesting user's key, so the job
reports articles and chunks as it goes.

The answer cache is keyed by `(account, generation, language, question)`. The account
is in the key because two users asking the same question legitimately search different
corpora; the generation is bumped when a build finishes, which retires that account's
stale answers without walking the cache.

## Design decisions

**Section-aware chunking.** `exsectionformat=wiki` keeps `== Heading ==` markers in the
plain-text extract, so chunks split along real topic boundaries instead of arbitrary
character offsets. Each chunk also knows its heading path, which is what makes
section-level citations possible. Boilerplate sections (References, External links, See
also, Further reading) are dropped at parse time — they are pure noise for retrieval.

**Breadcrumb-prefixed embeddings.** Every chunk is embedded as
`Article > Section > Subsection\n\n<body>`. Wikipedia prose is pronoun-heavy ("He then
published…"), so the breadcrumb is often the only signal that ties a chunk to its
subject.

**MMR over plain top-k.** Naive top-k returns five near-identical chunks from the same
section. Maximal Marginal Relevance (λ = 0.6) trades a little relevance for coverage,
which matters for multi-part questions and produces citations spanning more than one
article.

**History-aware query rewriting, folded into the tool call.** Follow-ups are
rewritten into standalone queries before retrieval, because embedding *"what about
his wife?"* retrieves nothing useful. That rewrite is no longer a separate step —
it is the `query` argument of the agent's tool call, so a follow-up pays one call
where it used to pay a rewrite *and* get routed. Best-effort throughout: a failed
decision falls open to a knowledge-base search on the raw message, which is what
the pipeline would have done before there was an agent at all.

**A pronoun resolves to the most recent subject, and there is a deterministic
backstop when it does not.** Asked *"who is albert einstein"* → *"his wife"* →
*"And what about MS Dhoni"* → *"And his wife"*, the pipeline answered the last one
about **Einstein**. Three things were wrong at once, and the first is the one worth
remembering:

* The agent prompt's own example was `"what about his wife?" → "Albert Einstein
  wife"`. The user typed "And his wife" and the model reproduced the example
  verbatim. A few-shot example that resembles a real query is not a demonstration,
  it is a default. The example is now written against an abstract `<subject>`, and
  the prompt states explicitly that pronouns bind to the *most recent* subject
  even when an earlier exchange answered a similarly-worded question.
* The transcript alone reads as a bag of equally-current subjects, so the request
  now names the current one on its own line rather than leaving the model to infer
  recency from message order.
* **The agent's resolution is checked, not trusted.** Across runs of one identical
  conversation, `gpt-4.1-nano` produced all three of "MS Dhoni wife" (right),
  "his wife" (nothing resolved), and "Albert Einstein wife" (confidently the
  wrong person). The last is the dangerous one: it retrieves real chunks and
  cites them, so the answer looks grounded while being about someone the user
  stopped asking about two turns ago. A first version only caught the
  *unchanged* case and let that straight through.

So on short referential messages — the only class where this goes wrong — the
subject the agent supplied has to appear in the question before it. If it does,
the rewrite stands, which is the common case: "Who was Marie Curie?" → "What did
she discover?" → "Marie Curie discoveries" shares *Marie Curie* and passes
untouched. If the agent invents a subject from nowhere recent, or resolves
nothing, the previous question is stitched on deterministically — and that
produces exactly the string speculation already searched, so being correct costs
nothing extra.

Corroboration looks at the immediately previous question and no further back.
Widening it to two was tried and defeats the purpose: in the very conversation
this exists for, Einstein *is* two questions back, so the wider window
corroborates the stale subject and waves it through. Chains survive the narrow
window anyway, because a resolution built on a referential question echoes a word
from it — "his wife" → "Albert Einstein wife death" shares *wife* — so
corroboration lands without ever needing to see the name.

Prompt wording alone was not enough here, and the fix that holds is the
deterministic one. Measured over six runs of the failing conversation: 0/6 wrong
subject, against a failure that reproduced intermittently before. The prompt
reduces how often the backstop is needed; it does not replace it.

**The agent is consulted where its judgement changes something.** Not on
greetings, not on opening questions — on follow-ups, where a reference has to be
resolved, and after a miss, where a different source has to be chosen. The
alternative reading of "agentic" is that the model decides everything, and it was
measured: 8 of 20 responses over 3s, against 1 of 20, for identical answers. An
agent that spends 0.9s confirming the obvious is not more agentic, it is slower.

**The tool menu is filtered by what can actually apply.** `answer_from_history` is
offered only once there is a conversation to answer from. This is not a
theoretical nicety — with the tool always on the menu, `gpt-4.1-nano` picked it for
four plain opening questions the index answers outright, losing the answer
entirely. Prompt wording did not prevent it. A structural gate costs nothing and
cannot be talked out of.

**Live Wikipedia is a real tool, not just a link list.** When the agent picks
`search_wikipedia`, the fetched article leads are shaped into the same `SearchHit`
structure retrieval returns and flow through the identical citation-binding and
verification path — a live-sourced claim is checked exactly as strictly as an
indexed one. Only lead sections, and only a couple of articles: a full
fetch-and-chunk would be a second ingest on the response path, and the budget has
no room for it.

**Disambiguation as a first-class path.** Disambiguation pages are indexed, flagged via
`pageprops`, and excluded from answer material. When one ranks first, the assistant
returns the options as clickable choices instead of guessing which "Mercury" you meant.

**Disambiguation options come from Wikipedia, not from the neighbours.** A bare entity
can also look ambiguous by score — several unrelated articles matching a one-word query
equally well — and the index may hold no disambiguation page for it. Listing the
nearest retrieved titles in that case produced *"Venus could refer to Synodic day,
Solar System, Plate tectonics"*: those are neighbours in embedding space, not senses of
a word, and offering them as choices reads as broken. So that path asks the live
MediaWiki API for `<term> (disambiguation)` first, and only falls back to nearby topics
if Wikipedia has none — worded as "the closest are…", never as "could refer to".

**Citations reflect actual use.** The response only lists sources whose `[n]` marker
appears in the generated answer, so the citation list never implies evidence the model
did not use. A refusal correctly carries zero sources.

**A citation is checked against the claim it carries.** Marker bookkeeping proves a
citation points at a *retrieved* chunk, not that the chunk *supports the sentence*. On
an ambiguous query the generator occasionally welds two subjects together — *"Java is a
programming language with four main spoken languages on the island of Java [2]"*, where
chunk [2] is the island article — and every mechanism above passes it, because the URL
is real and the marker is in range. So each cited sentence is compared with the chunk
it cites before sources are bound: near-zero vocabulary overlap means the citation
points at unrelated material, and an "X is a Y" claim whose category is absent from the
chunk means the model supplied the category itself. Failing sentences are dropped and
logged; if that empties the answer, the turn takes the ordinary not-covered path.

The check is deliberately conservative, and it is worth saying why. A first version
also verified verbs and adjectives and rejected 3 of 12 well-grounded answers — a 25%
false-positive rate — because honest paraphrase varies more than the heuristic allowed.
Requiring the determiner ("is **a** programming language") narrows it to genuine
category claims and excludes composition ("DNA is made of…") and negation ("was not a
single route…"), which are the constructions that paraphrase most freely. Catching a
real defect is not worth breaking four honest answers to catch it; a claim-level
entailment check that could go further needs an NLI model, not a word-overlap rule.

**Live-search fallback.** When nothing in the index clears the relevance floor, the
assistant queries the MediaWiki search API and routes the user to real articles rather
than synthesising an answer from unindexed material.

**Swappable vector store.** ChromaDB sits behind a narrow `add`/`query`/`count`
interface, so moving to Pinecone or Weaviate is one adapter, not a pipeline rewrite.

**SQLite for conversations, not the vector store.** History is ordered,
exact-lookup data — the wrong shape for a similarity index. A zero-dependency
SQLite file (WAL mode) makes threads survive backend restarts and work across
workers; assistant turns store their citations as JSON, so a reloaded page
resumes the conversation with sources intact. Scaling to a fleet means
reimplementing five methods against Redis or Postgres.

---

## Languages

The pipeline is language-agnostic — collections, chunking, retrieval, citations and
refusals are all per-language — but an index only answers in the languages it was
actually built with. What ships:

| Edition | Chunks | Status |
| ------- | ------ | ------ |
| `en` | 9,134 | ~300 articles across six domains |
| `es` | 3,578 | 50 articles, mirrored via `langlinks` |
| `fr`, `hi`, `mr` | 0 | collections exist; run `--mirror-langs` to populate |

A question in an indexed language gets a cited answer from that edition's own articles
(`es.wikipedia.org/...`), in that language. A question in a language with an empty
index falls through to live MediaWiki search and is routed to real articles rather than
answered — the same honest miss any uncovered topic gets, and it is written in the
caller's language, not English.

**Refusals are written by the model, in the caller's language.** This was a table of
hand-written templates covering six languages while the picker offered twelve — so a
Marathi question got an English apology, the exact failure the table existed to prevent.
A table can only hold the languages someone sat down and wrote.

A refusal is the one answer with no facts in it, which is what makes generating it safe:
there is nothing to ground and nothing to cite. The prompt still constrains it hard —
state no facts about the subject, do not describe what the articles say, never say
"excerpts" — because left to itself the model produces dead ends like *"the excerpts do
not cover mitochondria"*. Article titles are passed through untranslated.

The English template survives as the fallback when that call fails: a refusal in the
wrong language is a poor answer, and a turn that dies because its apology could not be
generated is no answer at all.

---

## Metrics

Measured by `scripts/evaluate.py` — 20 single-turn cases (including one that *must* be
refused) plus 4 multi-turn exchanges, cold path (cache cleared per case), driving the
same LangGraph streaming path the UI uses, answers graded by a `gpt-4.1` judge against
expected key facts. Results from the validation run on 2026-08-11 against the
~300-article / 9,134-chunk English index:

| Metric                       | Target  | Agentic (current) | Naive agentic | Pre-agentic |
| ---------------------------- | ------- | ----------------- | ------------- | ----------- |
| Answer accuracy              | >90%    | **100%** (20/20)  | 100%          | 100%        |
| Citation rate                | —       | **100%**          | 100%          | 100%        |
| Single-turn, median          | **<3s** | **1.65–1.90s**    | 2.41s         | 1.87s       |
| Single-turn, p95             | —       | **2.31–3.50s**    | 5.52s         | 2.57s       |
| Single-turn over 3s          | —       | **1–2 / 20**      | 8 / 20        | 0 / 20      |
| **Follow-up accuracy**       | >90%    | **100%** (6/6)    | 100% (6/6)    | 100% (4/4)  |
| **Follow-up, median**        | **<3s** | **2.49–2.92s**    | 2.74s         | 2.95s       |
| **Follow-up over 3s**        | —       | **0–3 / 6**       | 1 / 6         | 2 / 4       |
| Fast path (no decision call) | —       | **20 / 20**       | 0 / 20        | n/a         |
| Follow-up speculation reuse  | —       | **6 / 6**         | 2 / 6         | n/a         |
| Extra hops taken             | —       | 0 / 20            | 0 / 20        | n/a         |
| Chitchat response (HTTP API) | <300ms  | **2–16ms**        | 2–16ms        | 2–16ms      |

> **These numbers predate agent-decided routing and have not been re-measured.**
> The "Fast path" row no longer exists as a concept — every turn pays a decision
> call now — and the chitchat row is stale in the direction that matters: small
> talk moved from a dictionary lookup to one model call, measured at **~1.1s** on
> a warm process rather than 2–16ms. Question latency is expected to be roughly
> unchanged, since speculation still hides retrieval under the decision call, but
> `python -m scripts.evaluate` needs a run to confirm it.

"Naive agentic" is the same graph with the fast path and stitched speculation
removed — every non-chitchat turn paying a decision call. It is kept in the table
because it is what agentic RAG costs if you do not fight for the latency back, and
the gap between those two columns is the entire engineering argument.

**Both targets were met with margin, on both paths.** The route there was not
monotonic: the first agentic build cleared the median but pushed 8 of 20 responses
over 3s, because a ~0.9s decision call landed on every question including the ones
that had nothing to decide. Removing it where it was uninformative — opening
questions — and overlapping it where it was not — follow-ups — recovered all of
that and then some.

That fast path is gone now, deliberately, and the trade is worth naming: it was
bought with a keyword classifier deciding what each message *was*, and that
classifier could not recognise "do you know me". Correct routing on messages
nobody enumerated in advance was judged worth ~1s on small talk.

**Read p95 from a 20-case live-API suite with suspicion.** Repeated runs of an
identical build produced p95 anywhere from 2.3s to 30.7s, with individual cases
moving from 48.6s to 2.6s between runs minutes apart. Those are stalls inside the
generation call — the harness reports first-token time precisely so they can be
told apart from work this pipeline does. Hence the ranges above: they are the
spread across runs, not a confidence interval. The p50 and the over-3s count are
the stabler signals; a single quoted p95 from this suite is not a measurement.

**Follow-ups are now the slower path, and are still reported apart.** They pay the
decision call an opening question skips, which is the one place the model's
judgement is worth its latency: resolving "what happens at its event horizon?"
against the turn before it. Median 2.49s against 1.65s. Averaging the two would
hide that difference rather than report it.

**Accuracy held at 100%, but not on the first attempt**, and the failures are worth
recording because none of them were latency problems:

* The agent picked `answer_from_history` for four plain opening questions
  ("What is machine learning?") that the index answers with six hits each. Prompt
  wording did not fix it; removing the tool from the menu when there is no history
  did. A tool that cannot apply is not offered.
* The ambiguity heuristic started firing on agent-compressed queries. "What does
  photosynthesis produce?" became "photosynthesis products" — two words, tied
  scores — and tripped a threshold written for people typing "Mercury". It now
  scores the user's words, never the agent's.
* The generator was being handed the search string instead of the question. Asked
  "Mona Lisa painter" it wrote *"Leonardo da Vinci **is the** painter of the Mona
  Lisa"*, a copular claim that citation verification correctly drops; asked "Who
  painted the Mona Lisa?" it wrote "…painted the Mona Lisa", which passes. Same
  chunks, same model, different question shape. The generator now gets the user's
  question unless the message is referential.

Together those were 6 of 20 cases — a 65% run. All three are pinned by tests.

"<3s response time" is graded on the **complete response**, which is what the
specification asks for. Time to first token (~1.12s) is reported by the harness because
it explains where the time goes, but it is not the target and does not decide
pass/fail — grading on it would be scoring an easier exam. The UI streams, so what a
user perceives on a follow-up is still ~1.1s to first token.

`gpt-4.1-nano` is the default for both the agent and the generator because it clears
both targets with the most margin; `gpt-4.1-mini` measured ~0.5s slower for no
accuracy gain at this suite size, and `gpt-4o-mini` measured 80% (it drops facts the
retrieved context contains). `CHAT_MODEL` and `AGENT_MODEL` switch them
independently. Treat the last point of accuracy as run-to-run noise on a 26-case
suite rather than a settled difference between models.

Where the time goes at the median. An opening question: retrieval 0.43s, first
token 1.14s, then decoding a ~40-word answer — no decision call in the path at all.
A follow-up: decision call ~1.1s with retrieval **0.00s** running underneath it,
then the same generation. Retrieval having no measurable cost on the turns that do
consult the agent is the point of the whole speculation mechanism — the time that
used to be spent searching is now spent inside the call that runs alongside it.

```bash
cd backend
python -m scripts.evaluate     # exits non-zero if accuracy or latency gates fail
```

**Per-node latency** is logged on every turn, which is how the routing stays
honest — a direct reply must show three nodes and no tool executor, and a
speculation hit must show `tool_executor` at roughly zero:

```
# small talk — decided and answered in one call, no retrieval in the path
node=gate                     0.0ms
agent hop=0 tool=respond_directly query=''
node=agent                 1102.6ms
node=direct_responder         0.1ms
intent=chitchat tools=respond_directly spec=-   total= 1113ms  path: gate → agent → direct_responder

# question — decision call paid, retrieval ran underneath it
node=gate                     0.0ms
agent hop=0 tool=search_knowledge_base query='black hole event horizon'
node=agent                 1133.2ms
speculation hit: reused search for 'What is a black hole? What happens at its event horizon?'
node=tool_executor            0.1ms
node=generator             1108.9ms
intent=new_question tools=search_knowledge_base spec=hit total= 2249ms  path: gate → agent → tool_executor → generator
```

`spec` is the field that says the latency design is working: `hit` means the
turn's retrieval overlapped its decision call. A `-` on a retrieving turn means
it paid for both in series, and is the shape to look for when something is slow.
(A `-` on a chitchat turn is expected — nothing was wanted.) The same information
is on every API response under `agent`, so it can be checked without reading logs.

Timings are also returned on every API response and shown under each answer in the UI.

---

## Tests

```bash
cd backend
python -m pytest
```

247 tests, no network and no API key required. Each graph node is tested in isolation
against a stub services object that records what it was asked to do — which is how
the routing is pinned: `test_a_greeting_reaches_end_without_retrieval_or_generation`
asserts the visited-node list is exactly `["gate", "agent", "direct_responder"]`, and
`test_a_message_about_the_assistant_never_reaches_the_index` runs "do you know me"
and friends through the compiled graph and asserts no chunks and no citations survive
into the response. Both would pass just as happily with a stray retrieval in the path
if they only checked the answer, which is why they check the path. Routing through
the compiled graph is covered for every branch including the hop loop.

What the model actually decides is *not* unit-tested — a stub can only assert that
the graph honours a decision, never that the decision was right. That belongs to
`scripts/evaluate.py`, which runs real turns against the real model. Beyond that: the MediaWiki client
against a mocked API (including `maxlag` and 429 retry behaviour), section parsing,
chunk overlap and token budgets, citation extraction, MMR, conversation persistence
across restarts, and cache expiry.

The failure modes that are easy to regress get their own tests, each pinned to the
real case that motivated it:

* **Speculation reuse, both directions** — that a stripped query ("…what did she
  discover?" → "Marie Curie discoveries") still reuses the parked search, and that a
  resolved pronoun ("what about his wife?" → "Albert Einstein wife") never does.
  The second is the dangerous one: serving pronoun-retrieved chunks for a resolved
  query looks like a successful retrieval.
* **The stitched speculation** — that a referential follow-up searches
  `previous question + message`, and that the agent's resolved query is then
  contained in it, so the parked result is reused rather than repeated.
* **Subject corroboration, all four directions** — a stale subject is overridden
  ("Albert Einstein wife" two turns after the conversation moved to Dhoni), a
  corroborated one is left alone, a chain two questions deep still corroborates
  through the shared word, and a self-contained topic switch is never dragged
  back to the old subject.
* **No double search after a miss** — a turn that has already run a search must
  not speculate again on a query already known to return nothing.
* **Speculation is discarded on every non-retrieving branch** — `direct_responder`
  and `history_answerer` both cancel the parked task, and cancelling it must not
  surface as an unretrieved-task traceback under a turn that succeeded.
* **Agent-compressed queries and ambiguity** — that "photosynthesis products" is not
  mistaken for a bare entity mention, and that a bare "Java" the *user* typed still
  is.
* **Question shape reaching the generator** — that it is asked "Who painted the Mona
  Lisa?" and not "Mona Lisa painter", because the second phrasing draws a copular
  answer that citation verification then drops.
* **Tool availability** — that `answer_from_history` is off the menu on an opening
  message and back on it once there is a conversation, while `respond_directly`
  is offered from the first message, since small talk is usually what opens a
  conversation.
* **`respond_directly` with no reply falls back to searching** — an empty reply
  leaves nothing to say, and an empty turn is worse than a stiff one.
* **The hop loop terminates** — on the clock, on `max_hops`, and when every source
  comes back empty.
* **Cancelled speculations stay quiet** — `.exception()` on a cancelled task raises
  `CancelledError`, a `BaseException`, so suppressing only `Exception` logged a
  stack trace under every successful turn that missed.
* **Claim/citation mismatch** — the island article cited for *"Java is a programming
  language"*, and the mirror case that must survive untouched.
* **Disambiguation source** — that real Wikipedia options are preferred, that the
  nearest-topics fallback is worded as a fallback, and that an indexed disambiguation
  page skips the live lookup entirely.
* **Refusal phrasing** — that a whole question is never echoed back, and that low
  extraction confidence degrades to a generic subject rather than broken English.
* **Login throttling** — lockout after repeated failures, expiry, and that failures
  against one account cannot lock out another.

---

## Wikipedia compliance

This use case is explicitly permitted: Wikipedia content is Creative Commons licensed
(CC BY-SA) and served through an official public API. The client is a well-behaved
consumer of it:

- **User-Agent** — sends a descriptive UA with contact info on every request, per the
  [Wikimedia User-Agent policy](https://meta.wikimedia.org/wiki/User-Agent_policy).
  Set yours in `.env` before running against production Wikipedia.
- **`maxlag=5`** — the API sheds our load automatically when database replicas fall
  behind; the client backs off and retries.
- **Self-throttling** — a token bucket (default 8 req/s) plus a concurrency cap, and
  `Retry-After` is honoured on 429/503.
- **Official API only** — read-only `action=query` calls to `/w/api.php`. No HTML
  scraping, no bypassing rate limits.
- **Attribution** — answers link back to the source articles and sections, satisfying
  CC BY-SA attribution in the place users actually read.

---

## Known limitations

Things this build does not do, stated plainly rather than left to be discovered:

**Coverage is 300 articles, and misses look like retrieval bugs.** Ask "Saturn" and you
get equatorial ridges of its moons, because neither *Saturn* nor *Venus* is among the
seeded articles — the only Saturn text in the index sits inside other articles. This
reads as a ranking problem and is not one: a larger lead-section boost for short
queries was tried, measured, and backed out, because with no *Saturn* lead to promote
it merely lifted unrelated articles' leads over the only on-topic passages, and broke
ambiguity detection for "Venus" on the way. The fix is `--titles "Saturn,Venus"`, not
retrieval tuning. The agent can now route a miss to live Wikipedia instead of
refusing, which softens the symptom — but it only fires when the first search comes
back genuinely empty, and a *partial* miss like Saturn returns weak on-topic-looking
chunks rather than nothing.

**The agent is not consulted on opening questions.** By design, and it is the
reason the latency targets are met — but it does mean the model's source
judgement only ever runs *after* a failed search, never before one. A question the
corpus can answer badly (rather than not at all) will be answered from the corpus,
because the empty-result path is what triggers the agent and a weak-but-nonempty
result does not. `AGENT_FAST_PATH=false` makes every turn agent-decided at a
measured cost of ~0.9s and 8/20 responses over 3s.

**The agent almost never takes a second hop.** 0/20 on the benchmark suite, because
the indexed corpus answers those questions on the first search. The loop is
therefore mostly untested by the metrics above — its behaviour is pinned by unit
tests rather than demonstrated by the suite, and a corpus with more gaps would
exercise it far harder than this one does.

**Latency is quoted from a 20-case suite against a live API.** The p50 and over-3s
figures are stable across runs; p95 is not, and a single stalled generation call
moves it by tens of seconds (observed: 30.7s and 5.5s on consecutive runs of an
identical build). Treat p95 here as an upper-bound sighting rather than a
measurement, and re-run before drawing conclusions from it.

**A stitched query shows up verbatim in the refusal.** When the subject-
corroboration backstop fires and the topic is not covered, the refusal is built
around "ms dhoni and his wife" — the stitched search string rather than a phrase
anyone would write. The model now words the refusal and smooths this over more
often than the template did, but it is still handed the stitched string as the
subject. Fixing it properly means deriving a display subject separately from the
retrieval string, which is not done here.

**The agent's query quality varies by language.** The fallback search is only as
good as the query the agent writes, and that is measurably weaker outside English:
"प्रकाश संश्लेषण के बारे में बताइए" (hi) produced a query whose live search returned
*वैज्ञानिक, क्रम-विकास, चयापचय* rather than *प्रकाश-संश्लेषण*, which the term itself
retrieves correctly. The routing is right and the refusal is in the right language;
the suggested articles are simply worse than they should be.

**One wasted embedding on the `answer_from_history` path.** A follow-up speculates
before the agent has chosen, so a turn that turns out to need no retrieval at all
("what did you just tell me?") throws its speculation away. It never blocks the
turn — it runs alongside a call that was happening regardless — but it is billed.
Rare enough to be worth the win on every other follow-up.

**Answers run long on open-ended questions.** "Who is Albert Einstein" produced a
114-word bulleted list against a prompt asking for 2-3 sentences and 50 words, and
answer length is roughly linear in generation time. Two prompt changes were tried
and both were backed out, which is worth recording because the reasoning
generalises:

* *Forcing prose instead of bullets* made the model open with copular definitions
  — "Public-key cryptography **is a** cryptographic system…" — which is exactly
  the construction citation verification scrutinises. It deleted the sentence,
  leaving an answer that began "It enables…" with no antecedent. Accuracy 100% →
  95%. Bullets had been hiding those sentences from a check they should arguably
  face, but the check's false-positive rate is what made hiding them survivable.
* *A hard 50-word ceiling* ("an answer over 50 words is wrong even when true")
  dropped a required fact from a different case instead. Accuracy 100% → 95%.

Neither moved the benchmark's median answer length (40 → 42 words), because the
verbose answers are open-ended questions the suite barely contains. Trading a
validated accuracy number for a latency win that does not show up in the
measurement is a bad deal, so rule 4 is unchanged. Fixing this properly means a
length policy that varies with question type, not a stricter global cap.

Loosening the verifier was also measured and rejected: the honest paraphrase
scored **0.53** term overlap against its chunk while the real defect it exists to
catch ("Java is a programming language" cited to the island article) scored
**0.75**. The populations do not separate, so an overlap gate would admit the bug
and keep rejecting the paraphrase.

**Citation verification is a word-overlap heuristic, not entailment.** It catches a
citation pointing at unrelated material and a category claim the cited chunk never
makes. It will not catch a subtly wrong date or a plausible-sounding number that the
chunk does not contain. Doing that properly needs an NLI model on the response path,
which the latency budget does not have room for.

**Disambiguation depends on the term being ambiguous *to the index*.** "Python" answers
about the programming language rather than offering choices, because the language
article outranks the indexed disambiguation page. Wikipedia's own primary-topic
convention says that is usually right, but it means the behaviour is not uniform across
ambiguous words.

**Sessions and throttling are per-process.** `JWT_SECRET` defaults to a per-process
random value, and the login throttle and answer cache are in-memory, so multi-worker
deployments need `JWT_SECRET` set and would want Redis behind both.

---

## Project layout

```
backend/
  app/
    main.py              FastAPI app, CORS, lifespan (compiles graph, warms index)
    config.py            settings (.env)
    models.py            request/response schemas
    api/routes.py        chat, stream, conversations, stats, health
    graph/
      state.py           ChatState TypedDict threaded through every node
      nodes.py           the seven nodes + per-node latency instrumentation
      build.py           StateGraph wiring, conditional edges, the agent loop
      services.py        dependencies nodes call into (stubbable in tests)
      runner.py          drives one turn; blocking and SSE-streaming entry points
    core/
      wikipedia.py       MediaWiki client: rate limiting, maxlag, section parsing
      chunker.py         section-aware chunking with token overlap
      embeddings.py      batched OpenAI embeddings with backoff
      vector_store.py    ChromaDB adapter (per-language collections)
      retriever.py       semantic search, MMR, disambiguation, live fallback
      rag.py             prompts, tool schemas, grounding rules, citation binding
      ratelimit.py       failed-login throttling
      conversation.py    multi-turn memory, SQLite-persisted, TTL eviction
      cache.py           TTL answer cache
  scripts/
    ingest.py            crawler + indexer CLI
    seeds.py             seed topics and categories
    evaluate.py          accuracy + latency benchmark
  tests/                 pytest suite (247 tests, no network or API key needed)

frontend/
  app/                   Next.js App Router entry, global styles
  components/            chat window, message bubbles, sources, routing, disambiguation
  lib/                   SSE client and shared types
```
