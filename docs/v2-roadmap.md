# Cricket Guru v2

v1 is frozen at the `v1` git tag and stays live at
[cricket-guru-production.up.railway.app](https://cricket-guru-production.up.railway.app). v2 is the
ongoing line on `main`, deployed separately at
[cricket-guru-v2-production.up.railway.app](https://cricket-guru-v2-production.up.railway.app) from
the same repo and the same image pipeline. This doc is what v2 is for and what lands in it.

## Deployment

One repo, one image, two Railway services in the same project:

- **v1** — service `cricket-guru`, at the original URL. Frozen: it keeps running the image it has and is not redeployed, so the demo everyone has the link to never shifts under them.
- **v2** — service `cricket-guru-v2`, at its own URL, tracking `main`. Each merge to `main` that changes the image rebuilds `ghcr.io/sayan-mitra/cricket-guru:latest` in CI, and v2 redeploys from it.

Both run the same single-machine image (Postgres + embedded Qdrant + Streamlit), Sonnet answering and Haiku judging, no OpenAI. Each has its own `/data` volume.

## What v2 is about

v1 answers well but slowly, and the eval surfaced a couple of correctness edges worth closing. v2 keeps the architecture and the accuracy work and goes after two things: latency, and the failure modes below. No new legs, no new sources — the same system, tightened.

## Latency

Every leg is a serial LLM call on Sonnet, and a multi-step question stacks a lot of them. A two-part query runs roughly:

```
gate → agent-reason → [stats: sqlgen + phrase] → agent-reason → [stats: sqlgen×2 + phrase]
     → agent-reason (compose) → groundedness guard → critic
```

That is about ten Sonnet round-trips in sequence, each re-processing a large static prompt. Three structural facts make it worse: `SQL_SYS` is ~2,300 tokens and is re-sent on every sqlgen call (up to three tries per stats call), there is no prompt caching anywhere, and the DB connection sets no `search_path`, so a bare `matches` instead of `cricsheet.matches` errors and forces an extra sqlgen round-trip. On top of that, `parallel_tool_calls=False` (the deadlock fix) serializes independent lookups, so "IPL 2015 vs 2011" runs one after the other.

### Plan, ranked

Tier 1 — high impact, low risk, small change:

1. **`search_path=cricsheet` on connect** (`db.py`, one line). Kills the "relation does not exist" retry class — a wasted Sonnet sqlgen call per affected query. No quality risk.
2. **Prompt-cache the static prompts** (`SQL_SYS` first, then the agent, critic, and guardrail system prompts). Anthropic caching processes the repeated prefix far faster and cheaper, and it is hit four to six times per multi-step query. Verify the cache-control API for pydantic-ai 0.8.1.
3. **Run the mechanical legs on Haiku** via a `CG_FAST_MODEL` knob: the input gate, the phraser, and the groundedness guard are classify-or-format tasks. Keep the router's reasoning on Sonnet; test sqlgen on Haiku before switching it.

Tier 2 — bigger wins, a real refactor:

4. **Merge the groundedness guard and the critic** into one post-answer call, saving a round-trip.
5. **Re-enable parallel tool calls safely** — make the arms async-safe so independent sub-lookups run concurrently instead of serially. Highest payoff, highest risk, since it reopens the deadlock the serial fix closed.
6. **Cheapen the stats path** — the phraser is a second LLM call that turns rows into a sentence the agent then re-reads; return structured rows and let the agent compose, or phrase on Haiku. Make the rename alias-probe (extra DB queries on every team-filter query) conditional.

## Correctness

**False abstention.** The system declines to answer a question it could and should have answered. The worked case: "Rahul Dravid's Test average post-2001, and the top-5 Indian averages" — two `cricket_stats` calls computed Dravid's in-window average two ways (46.29 with no team filter and an explicit date bound; 46.84 with `batting_team='India'`), the answer carried both, and the critic mislabeled that internal clash as `retrieval_gap` and abstained. The data was right; the system talked itself out of shipping it. The opposite failure is a false *ship* — answering wrongly with confidence — and the critic exists to prevent that, so the fix is calibration, not removing the check.

Fixes:
- **Stats-arm query consistency.** One canonical query template for "a batter's in-window average," reused by both the single-player path and the leaderboard, so a lookup and a list can't disagree.
- **Agent composition.** Don't emit a figure no tool returned (the answer also invented a "~52.31 all-time" career number from nowhere), and reconcile when two tool results describe the same quantity.
- **Critic verdicts.** Distinguish an internal inconsistency from a retrieval gap; for an otherwise-answered stats question, prefer ship-with-the-reconciled-figure over abstain. Reserve abstain for genuinely out-of-window or unverifiable records.

## Ownership

These are all leg-level changes — `db.py`, `llm.py`, `arms/`, `guardrails.py`, `serve.py`, `routing/` — the enhancement session's area (see [handoff-enhance-legs.md](handoff-enhance-legs.md)). This doc is the shared plan; the deploy side owns the v1/v2 split, the image pipeline, and the redeploys.
