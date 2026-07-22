#!/usr/bin/env python3
"""Phase-B judge validation — generate the material a human labels.

For each narrative gold item: get the agent's answer, then BOTH judges' verdicts
(same=gpt, cross=sonnet). The human later labels correct/incorrect independently,
WITHOUT seeing the judge verdicts (the app hides them) so the call isn't anchored.
Agreement (judge vs human) is computed by phase_b_agreement.py once labels exist.

    cd backend && PYTHONPATH=. python -m cricket_guru.eval.phase_b
"""
import json

from cricket_guru.config import DATA_DIR
from cricket_guru.eval.harness import load_gold
from cricket_guru.eval.judge import make_judge
from cricket_guru.routing import get_router

OUT = DATA_DIR / "results" / "phase_b_items.json"


def main(ids=None):
    gold = load_gold("narrative")
    if ids:                                # regenerate a subset and merge into the file
        gold = [g for g in gold if g["id"] in ids]
    agent = get_router("agent")            # same path L6 judges (gpt answerer)
    same, cross = make_judge("same"), make_judge("cross")

    fresh = []
    for i, g in enumerate(gold, 1):
        q, ref = g["question"], g["reference"]
        ans = agent.answer(q).text
        sv, cv = same(q, ans, ref), cross(q, ans, ref)
        fresh.append({
            "id": g["id"], "question": q, "reference": ref, "candidate": ans,
            "same": {"correct": sv.correct, "reason": sv.reason},
            "cross": {"correct": cv.correct, "reason": cv.reason},
        })
        print(f"[{i}/{len(gold)}] {g['id']}: same={sv.correct} cross={cv.correct}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if ids and OUT.exists():               # patch: replace only the regenerated ids, keep order
        by_id = {x["id"]: x for x in json.loads(OUT.read_text())}
        by_id.update({x["id"]: x for x in fresh})
        merged = list(by_id.values())
        OUT.write_text(json.dumps(merged, indent=2))
        print(f"patched {len(fresh)} items into {OUT} ({len(merged)} total)", flush=True)
    else:
        OUT.write_text(json.dumps(fresh, indent=2))
        print(f"wrote {len(fresh)} items -> {OUT}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="comma-separated gold ids to regenerate and merge")
    a = ap.parse_args()
    main(set(a.ids.split(",")) if a.ids else None)
