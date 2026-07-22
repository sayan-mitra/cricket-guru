#!/usr/bin/env python3
"""Stats gold set — templated from Cricsheet with SQL-computed answers.

Objective ground truth, no LLM: each question is a template filled with a real
(format, season), and its answer is computed by the same SQL over Cricsheet.
Season-bounded questions are stable facts (staleness=False). A few all-time
record questions are tagged staleness=True — those are what a years-old prose
corpus gets wrong.

    python -m cricket_guru.eval.gold_stats
"""
import json
import random

from cricket_guru.config import DATA_DIR
from cricket_guru.db import connect

SEED = 42
OUT = DATA_DIR / "gold" / "stats_gold.json"
FMT_NAME = {"Test": "Tests", "ODI": "ODIs", "T20I": "T20Is", "IPL": "the IPL"}

# Bowler credit excludes dismissals a bowler doesn't earn.
NON_BOWLER = "('run out','retired hurt','retired out','obstructing the field')"

# (question template, SQL returning one row of (label, value)); {f}=format {s}=season
SEASON_TEMPLATES = [
    ("Who scored the most runs in {fmt} {season}?",
     "SELECT d.batter, SUM(d.runs_batter) v FROM cricsheet.deliveries d "
     "JOIN cricsheet.matches m ON m.match_id=d.match_id "
     "WHERE m.format=%(f)s AND m.season=%(s)s "
     "GROUP BY d.batter ORDER BY v DESC LIMIT 1"),
    ("Who took the most wickets in {fmt} {season}?",
     "SELECT d.bowler, COUNT(*) v FROM cricsheet.deliveries d "
     "JOIN cricsheet.matches m ON m.match_id=d.match_id "
     f"WHERE m.format=%(f)s AND m.season=%(s)s AND d.wicket_kind IS NOT NULL "
     f"AND d.wicket_kind NOT IN {NON_BOWLER} "
     "GROUP BY d.bowler ORDER BY v DESC LIMIT 1"),
    ("Who hit the most sixes in {fmt} {season}?",
     "SELECT d.batter, COUNT(*) v FROM cricsheet.deliveries d "
     "JOIN cricsheet.matches m ON m.match_id=d.match_id "
     "WHERE m.format=%(f)s AND m.season=%(s)s AND d.runs_batter=6 "
     "GROUP BY d.batter ORDER BY v DESC LIMIT 1"),
    ("What was the highest team innings total in {fmt} {season}?",
     "SELECT i.batting_team, i.runs FROM cricsheet.innings i "
     "JOIN cricsheet.matches m ON m.match_id=i.match_id "
     "WHERE m.format=%(f)s AND m.season=%(s)s ORDER BY i.runs DESC LIMIT 1"),
]

# All-time records — the staleness-prone subset (no season bound).
ALLTIME_TEMPLATES = [
    ("Who has hit the most sixes in {fmt}?",
     "SELECT d.batter, COUNT(*) v FROM cricsheet.deliveries d "
     "JOIN cricsheet.matches m ON m.match_id=d.match_id "
     "WHERE m.format=%(f)s AND d.runs_batter=6 "
     "GROUP BY d.batter ORDER BY v DESC LIMIT 1"),
    ("Who has scored the most career runs in {fmt}?",
     "SELECT d.batter, SUM(d.runs_batter) v FROM cricsheet.deliveries d "
     "JOIN cricsheet.matches m ON m.match_id=d.match_id "
     "WHERE m.format=%(f)s GROUP BY d.batter ORDER BY v DESC LIMIT 1"),
]

# Ad-hoc stats questions rescued from the narrative gold — pre-computed answers
# (only in-window facts Cricsheet can actually reproduce).
EXTRA = [
    {"question": "Which bowler has taken all ten wickets in an innings in a Test match?",
     "answer_label": "Ajaz Patel", "answer_value": 10, "staleness": False,
     "meta": {"note": "AY Patel 10/119 (2021); Kumble's 1999 10/74 predates Cricsheet coverage"}},
]


def seasons_with_data(cur, fmt, min_matches=15):
    cur.execute("SELECT season FROM cricsheet.matches WHERE format=%s AND season IS NOT NULL "
                "GROUP BY season HAVING COUNT(*) >= %s ORDER BY season", (fmt, min_matches))
    return [r[0] for r in cur.fetchall()]


def main():
    rng = random.Random(SEED)
    conn = connect()
    cur = conn.cursor()
    gold, qid = [], 0

    # Season-bounded: a deterministic spread of seasons per format.
    per_format = {"IPL": 4, "ODI": 3, "T20I": 3, "Test": 3}
    for fmt, n in per_format.items():
        avail = seasons_with_data(cur, fmt)
        picks = sorted(rng.sample(avail, min(n, len(avail))))
        for season in picks:
            for qtext, sql in SEASON_TEMPLATES:
                cur.execute(sql, {"f": fmt, "s": season})
                row = cur.fetchone()
                if not row:
                    continue
                label, value = row
                qid += 1
                gold.append({
                    "id": f"stats-{qid}", "qtype": "stats", "staleness": False,
                    "question": qtext.format(fmt=FMT_NAME[fmt], season=season),
                    "answer_label": str(label), "answer_value": value,
                    "answer": f"{label} ({value})",
                    "meta": {"format": fmt, "season": season},
                })

    # All-time records: the staleness-prone subset.
    for fmt in ("T20I", "ODI", "Test", "IPL"):
        for qtext, sql in ALLTIME_TEMPLATES:
            cur.execute(sql, {"f": fmt})
            row = cur.fetchone()
            if not row:
                continue
            label, value = row
            qid += 1
            gold.append({
                "id": f"stats-{qid}", "qtype": "stats", "staleness": True,
                "question": qtext.format(fmt=FMT_NAME[fmt]),
                "answer_label": str(label), "answer_value": value,
                "answer": f"{label} ({value})",
                "meta": {"format": fmt, "all_time": True},
            })

    conn.close()

    for e in EXTRA:
        qid += 1
        gold.append({
            "id": f"stats-{qid}", "qtype": "stats", "staleness": e["staleness"],
            "question": e["question"], "answer_label": e["answer_label"],
            "answer_value": e["answer_value"],
            "answer": f"{e['answer_label']} ({e['answer_value']})", "meta": e.get("meta", {}),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gold, indent=2))
    print(f"wrote {len(gold)} stats questions "
          f"({sum(g['staleness'] for g in gold)} staleness-tagged) -> {OUT}")
    for g in gold[:3] + gold[-2:]:
        print(f"  [{g['id']}] {g['question']}  ->  {g['answer']}")


if __name__ == "__main__":
    main()
