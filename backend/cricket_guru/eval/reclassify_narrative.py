#!/usr/bin/env python3
"""Route-by-source classifier for the narrative gold.

Instead of keep/drop on Sports-SE-reference quality, decide which of OUR sources
can actually answer each question: rules, stats, wiki, or drop. This rescues good
questions a weak SE answer would otherwise sink, and mines more usable gold.

Output: data/gold/narrative_routes.json  {id: {source, reason}}

    python -m cricket_guru.eval.reclassify_narrative
"""
import json

from pydantic import BaseModel
from pydantic_ai import Agent

from cricket_guru.config import ANSWERER_MODEL, DATA_DIR

SYS = (
    "Pick the ONE source that best answers this cricket question with our system, "
    "judging the QUESTION's answerability (ignore the provided reference's quality):\n"
    "- rules: laws / officiating, answerable from the MCC Laws + ICC/IPL playing conditions\n"
    "- stats: a number / record, computable from a ball-by-ball match database "
    "(runs, wickets, dismissals, sixes, results — by player, season, or format)\n"
    "- wiki: narrative / terminology / history, answerable from an encyclopedia\n"
    "- drop: vague, opinion, or needs data we don't have (org eligibility policy, "
    "boundary-vs-run distinctions, live rankings)\n"
    "Answer with source = one of: rules, stats, wiki, drop."
)
VALID = {"rules", "stats", "wiki", "drop"}


class Route(BaseModel):
    source: str
    reason: str


def main():
    gold = json.loads((DATA_DIR / "gold" / "narrative_gold.json").read_text())
    agent = Agent(ANSWERER_MODEL, output_type=Route, system_prompt=SYS)
    routes, tally = {}, {}
    for g in gold:
        r = agent.run_sync(
            f"Question: {g['question']}\nTags: {g.get('tags','')}").output
        src = r.source.strip().lower()
        src = src if src in VALID else "drop"
        routes[g["id"]] = {"source": src, "reason": r.reason}
        tally[src] = tally.get(src, 0) + 1
        print(f"{src:6} {g['id']:8} {g['question'][:60]}")

    (DATA_DIR / "gold" / "narrative_routes.json").write_text(json.dumps(routes, indent=2))
    print("\ndistribution:", tally)


if __name__ == "__main__":
    main()
