# Cricket Guru

A cricket question-answering agent that routes each question to the source that actually holds the answer — exact numbers from a structured match database, narrative from an encyclopedic text index, and a web-search freshness check for records. Built to compare, at each pipeline leg, two prominent approaches against a simple baseline rather than reaching for the optimal one first.

Group: IP8, Cricket Guru. Framework: Pydantic AI.

## The idea

Cricket knowledge lives in two incompatible shapes. Prose holds narrative (why a match mattered, how a rule works); structured records hold exact facts (most runs in IPL 2016). No single retrieval method serves both, and records go stale the moment they break. So Cricket Guru is an agent that reads the question and picks the right tool, and the project measures where each approach wins.

## What's compared (each leg: baseline → advanced, one leg at a time)

| Leg | Baseline | Advanced |
|---|---|---|
| L1 Chunking | fixed-size | structural |
| L3 Retrieval | dense | hybrid (dense + BM25) |
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
frontend/app.py     Streamlit: chat + comparisons dashboard
```

## Data

- Cricsheet ball-by-ball (men's intl + IPL): 8,142 matches, 4.14M deliveries → Postgres (`cricsheet`).
- Sports Stack Exchange `cricket` tag: 839 Q, 459 accepted → Postgres (`sports_se`), the narrative oracle.
- Wikipedia cricket articles: 575 → Qdrant (`wiki_fixed` / `wiki_structural`).

All CC-BY-SA / ODC; raw data is git-ignored and reproduced by the fetch/load scripts. See `data/README.md`.

## Run it (local)

```bash
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt streamlit
cp backend/.env.example backend/.env      # add ANTHROPIC_API_KEY + OPENAI_API_KEY

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

## Run it (Docker)

`docker compose up` brings up Postgres, Qdrant (server mode), and the app. Run the ingestion + index steps once against the containers (they read `CG_*` env). The app reaches Qdrant via `CG_QDRANT_URL`.

## Models

Answerer/arms/agent run on `CG_ANSWERER_MODEL`, the cross-judge on `CG_JUDGE_MODEL` (Pydantic AI `provider:model` strings). Set in `.env`. The intended setup is `anthropic:claude-sonnet-5` answering and an OpenAI cross-judge; both can point at one vendor for testing.

## Evaluation

Stats questions are scored objectively (SQL-computed answer must appear in the response). Narrative questions are scored by an LLM-judge against the Sports SE accepted answer, with same-model vs cross-model compared and validated against a hand-labeled sample. Reused projects or fabricated evaluation data are out of scope — every number the harness prints comes from a live run over the frozen gold set.
