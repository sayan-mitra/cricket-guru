#!/usr/bin/env python3
"""Phase-B agreement — compare each judge to the human labels of record.

Reads phase_b_items.json (judge verdicts) + phase_b_labels.json (human calls).
Reports same-vs-human and cross-vs-human agreement over decisive labels, and where
each judge diverges. The higher-agreement judge is the one to trust for L6. No
Qdrant, no LLM — pure comparison, safe to run anytime.

    cd backend && PYTHONPATH=. python -m cricket_guru.eval.phase_b_agreement
"""
import json

from cricket_guru.config import DATA_DIR

ITEMS = DATA_DIR / "results" / "phase_b_items.json"
LABELS = DATA_DIR / "gold" / "phase_b_labels.json"


def main():
    items = {x["id"]: x for x in json.loads(ITEMS.read_text())}
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}

    n = n_same = n_cross = 0
    diverge = []
    for gid, lab in labels.items():
        h = lab.get("human")
        if h not in ("correct", "incorrect") or gid not in items:
            continue                                   # skip unlabeled / borderline
        human = h == "correct"
        s = items[gid]["same"]["correct"]
        c = items[gid]["cross"]["correct"]
        n += 1
        n_same += s == human
        n_cross += c == human
        if s != human or c != human:
            diverge.append((gid, human, s, c))

    if not n:
        print("No decisive human labels yet (correct/incorrect). Label in the app first.")
        return
    print(f"Decisive human labels: {n}")
    print(f"same  (gpt)    agrees with human: {n_same}/{n} = {n_same/n:.0%}")
    print(f"cross (sonnet) agrees with human: {n_cross}/{n} = {n_cross/n:.0%}")
    if diverge:
        print("\nDivergences (id | human | same | cross):")
        for gid, human, s, c in diverge:
            m = lambda b: "✓" if b else "✗"
            print(f"  {gid}: human={m(human)} same={m(s)} cross={m(c)}")


if __name__ == "__main__":
    main()
