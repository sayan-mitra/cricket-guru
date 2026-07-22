# Cricket Guru

A cricket question-answering agent that routes each question to the source that actually holds the answer — exact numbers from a structured match database, narrative from an encyclopedic text index, and a web-search freshness check for records. Built to compare, at each pipeline leg, two prominent approaches against a simple baseline rather than reaching for the optimal one first.

Group: IP8, Cricket Guru. Framework: Pydantic AI.

Live demo: <https://cricket-guru-production.up.railway.app>

## The idea

Cricket knowledge lives in two incompatible shapes. Prose holds narrative (why a match mattered, how a rule works); structured records hold exact facts (most runs in IPL 2016). No single retrieval method serves both, and records go stale the moment they break. So Cricket Guru is an agent that reads the question and picks the right tool, and the project measures where each approach wins.

## How it works

A question passes input guardrails (cricket-relevance, a prompt-injection check, safety), the router picks how to answer, one arm answers, and a critic decides whether it ships.

```mermaid
flowchart TB
  Q([cricket question]) --> G[guardrails: cricket-gate · injection · safety]
  G --> RT{router}
  RT -->|ReAct agent · or keyword rule-router| ARMS{{pick an answer arm}}
  ARMS --> S[stats-SQL<br/>Cricsheet Postgres]
  ARMS --> WK[text-RAG · wiki<br/>paragraph chunks · hybrid]
  ARMS --> RB[text-RAG · rules<br/>clause chunks · dense]
  ARMS --> WB[web search<br/>Tavily → Brave → DuckDuckGo]
  S --> C[answer critic · CRAG]
  WK --> C
  RB --> C
  WB --> C
  C -->|ok| SHIP([ship ✓])
  C -->|retrieval_gap| WEB([web + caveat 🌐])
  C -->|hallucination| ABS([abstain ⚠])
```

Three arms answer:

- **stats-SQL** — text-to-SQL over the Cricsheet Postgres. The model writes one read-only query, it runs, and a phraser turns the rows into a sentence. This is the objective oracle for numbers; it can't answer history, records, or captaincy (there is no captain column), so those route to prose.
- **text-RAG** — retrieve chunks from Qdrant (wiki prose or the rules corpus) and answer only from them.
- **web search** — a freshness check for records and facts the corpus can't hold.

The router is either a keyword rule-router (the baseline) or a Pydantic AI tool-calling agent (the serving default). The agent runs a ReAct loop: think about which tool the question needs, call it, read the result, and loop until it can answer, trying another tool when one comes up short.

On the wiki arm a **cross-encoder reranker** (`bge-reranker-base`) re-scores the top 20 retrieved chunks and keeps the top 5, which lifts wiki recall@1 from 60% to 80% — the right passage is usually retrieved but ranked too low. It is wiki-only: rules already rank the right clause first about 90% of the time, where reranking slightly hurts.

After an arm answers, a **CRAG critic** grades the finished answer and, on a bad grade, corrects instead of shipping: `ok` ships it, `retrieval_gap` falls back to web plus a caveat, and `hallucination` (or an all-time record reaching before the data window) abstains with the reason shown.

The full walkthrough — the ReAct and CRAG diagrams, what each leg taught us, and the experiment and gold-set design — is in [docs/how-it-works.md](docs/how-it-works.md).

## What's compared (each leg: baseline → advanced, one leg at a time)

This is an offline eval, not a mode in the app: `python -m cricket_guru.eval.run_experiments` runs each leg's baseline against its advanced variant on the frozen gold set, across combinations. The app serves the one configuration that won.

| Leg | Baseline | Advanced |
|---|---|---|
| L1 Chunking | fixed-size | structural |
| L3 Retrieval | dense | hybrid (dense + BM25) |
| L4 Reranking | bi-encoder ranking | cross-encoder rerank on wiki (`bge-reranker-base`) |
| L5 Routing | rule/keyword | LLM tool-calling agent |
| L6 Judge | same-model | cross-model (different vendor) |

## Layout

```
backend/cricket_guru/
  config.py         env-driven settings + PipelineConfig (the leg "mode" object)
  db.py  qdrant_store.py  llm.py
  ingest/           fetch + load: Cricsheet, Sports SE, Wikipedia
  index/            L1 chunking (fixed|structural), FastEmbed, build_index CLI
  retrieval/        L3 dense | hybrid
  arms/             text_rag, stats_sql (shared answer() interface)
  routing/          L5 rule | agent (Pydantic AI)
  tools/            web-search freshness (frozen at eval time)
  eval/             gold_stats, gold_narrative, judge, harness, run_experiments
frontend/app.py     Streamlit: chat + traces + how-it-works
```

## Data

- Cricsheet ball-by-ball (men's intl + IPL): 8,142 matches, 4.14M deliveries → Postgres (`cricsheet`).
- Sports Stack Exchange `cricket` tag: 839 Q, 459 accepted → Postgres (`sports_se`), the narrative oracle.
- Wikipedia cricket articles: 575 → Qdrant (`wiki_fixed` / `wiki_structural`).

All CC-BY-SA / ODC; raw data is git-ignored and reproduced by the fetch/load scripts. See `data/README.md`.

## Run it (local)

```bash
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt streamlit
cp backend/.env.example backend/.env      # add ANTHROPIC_API_KEY + TAVILY_API_KEY

# one-time ingestion (Postgres running locally)
export PYTHONPATH=backend
.venv/bin/python -m cricket_guru.ingest.fetch_sports_se
.venv/bin/python -m cricket_guru.ingest.fetch_wikipedia
psql -d cricket_guru -f backend/cricket_guru/ingest/schema.sql
.venv/bin/python -m cricket_guru.ingest.load_cricsheet     # after extracting Cricsheet zips to data/cricsheet
.venv/bin/python -m cricket_guru.ingest.load_sports_se
.venv/bin/python -m cricket_guru.index.build_index --chunking fixed
.venv/bin/python -m cricket_guru.index.build_index --chunking structural

# gold sets + experiments
.venv/bin/python -m cricket_guru.eval.gold_stats
.venv/bin/python -m cricket_guru.eval.gold_narrative
.venv/bin/python -m cricket_guru.eval.run_experiments --n 10

# app
.venv/bin/streamlit run frontend/app.py
```

## Deploy (single machine)

`deploy/Dockerfile` bakes Postgres, the Streamlit app, and the on-disk Qdrant index into one image, so the container comes up self-contained. LLM keys come from the host env, never the image. This is what runs the live demo on Railway; `deploy/` holds the entrypoint and `.github/workflows/` the CI that builds the image.

## Models

The answerer, arms, agent, and critic run on `CG_ANSWERER_MODEL` (`anthropic:claude-sonnet-5`); the eval cross-judge on `CG_JUDGE_MODEL` (`anthropic:claude-haiku-4-5`). Web search is Tavily, embeddings are local FastEmbed, so serving needs only the Anthropic key plus Tavily for the web fallback — no OpenAI. Set the models in `.env` as Pydantic AI `provider:model` strings.

## Evaluation

Stats questions are scored objectively (SQL-computed answer must appear in the response). Narrative questions are scored by an LLM-judge against the Sports SE accepted answer, with same-model vs cross-model compared and validated against a hand-labeled sample. Reused projects or fabricated evaluation data are out of scope — every number the harness prints comes from a live run over the frozen gold set.
