"""Multi-step gold — questions that REQUIRE composing 2+ facts, to verify the ReAct loop's
decomposition, not just its tool choice.

Scored by BOTH the final answer AND the trace (see harness `_multistep`): an answer that is
right but reached in a single hop — or produced by the rule-router, which can't compose — fails
even when the number is correct. That's the point: it measures decomposition depth.

Two flavors:
  multistep-stats  — combine two Cricsheet facts (difference of two seasons' top scorers).
                     Answer is SQL-computed, so it's authoritative.
  multistep-cross  — a stats lookup + a rules lookup; needs two DIFFERENT tools (one SQL can't).

    cd backend && PYTHONPATH=. python -m cricket_guru.eval.gold_multistep
"""
import json

from cricket_guru.config import DATA_DIR
from cricket_guru.db import connect

OUT = DATA_DIR / "gold" / "multistep_gold.json"

# (format, season A, season B) — difference of the two seasons' leading run-scorers.
STATS_PAIRS = [("IPL", "2016", "2007/08"), ("IPL", "2015", "2011"), ("IPL", "2019", "2009")]

# Cross-domain: the stats step identifies an entity, the rules step is the actual answer.
CROSS = [
    {"id": "multistep-cross-1", "qtype": "multistep",
     "question": "Consider the bowler with the most Test wickets in the database. In cricket, is "
                 "underarm bowling permitted?",
     "answer_label": "not permitted", "min_tools": 2,
     "expected_tools": ["cricket_stats", "cricket_rules"],
     "reference": "Underarm bowling is not permitted unless both captains agree before the match "
                  "(MCC Law 21.1.2). The stats step only identifies the bowler; the rule is the answer."},
    {"id": "multistep-cross-2", "qtype": "multistep",
     "question": "For the team that won the most recent India–England Test series, how many fielders "
                 "are allowed outside the fielding circle during the first powerplay of an ODI?",
     "answer_value": 2, "min_tools": 2,
     "expected_tools": ["cricket_stats", "cricket_rules"],
     "reference": "The 2025 series was drawn 2-2 (stats). In ODI Powerplay 1, no more than 2 fielders "
                  "are permitted outside the 30-yard circle (ICC ODI playing conditions, clause 28.7)."},
]


def _top_scorer(cur, fmt, season):
    cur.execute(
        "SELECT batter, SUM(runs_batter) FROM cricsheet.deliveries d "
        "JOIN cricsheet.matches m ON d.match_id = m.match_id "
        "WHERE m.format=%s AND m.season=%s GROUP BY batter ORDER BY 2 DESC LIMIT 1", (fmt, season))
    return cur.fetchone()


def main():
    conn = connect()
    cur = conn.cursor()
    items = []
    for fmt, a, b in STATS_PAIRS:
        (pa, ra), (pb, rb) = _top_scorer(cur, fmt, a), _top_scorer(cur, fmt, b)
        diff = int(ra) - int(rb)
        items.append({
            "id": f"multistep-stats-{fmt}-{a.replace('/', '')}", "qtype": "multistep",
            "question": f"How many more runs did the leading run-scorer of {fmt} {a} make than the "
                        f"leading run-scorer of {fmt} {b}?",
            "answer_value": diff, "min_tools": 2,
            "reference": f"{fmt} {a} top scorer {pa} ({ra}); {fmt} {b} top scorer {pb} ({rb}); "
                         f"difference {diff}."})
        print(f"  {fmt} {a} ({pa} {ra}) vs {b} ({pb} {rb}) -> diff {diff}")
    conn.close()
    items += CROSS
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(items, indent=2))
    print(f"wrote {len(items)} multi-step items -> {OUT}")


if __name__ == "__main__":
    main()
