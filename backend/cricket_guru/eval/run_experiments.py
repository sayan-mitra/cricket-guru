#!/usr/bin/env python3
"""Run the comparative-analysis legs and save results for the dashboard.

Ablation: baseline = fixed chunking / dense retrieval / rule router / same judge.
Each leg flips one variant against that baseline. Sample size is small by default
to bound LLM cost; raise --n for tighter numbers.

    python -m cricket_guru.eval.run_experiments --n 10
"""
import argparse
import json

from cricket_guru.arms.text_rag import TextRAGArm
from cricket_guru.config import DATA_DIR
from cricket_guru.eval.harness import load_gold, run
from cricket_guru.routing import get_router

OUT = DATA_DIR / "results" / "experiments.json"


def main(n):
    # Corpus-grounded gold on both prose and rules: the reference IS the source passage/clause, not an
    # LLM draft — so the chunking/retrieval legs measure something real instead of the old self-written
    # answers. stats stays the SQL oracle.
    narr = json.loads((DATA_DIR / "gold" / "narrative_gold_corpus.json").read_text())
    stats = load_gold("stats")[:n]
    rules = json.loads((DATA_DIR / "gold" / "rules_gold_corpus.json").read_text())
    multistep = json.loads((DATA_DIR / "gold" / "multistep_gold.json").read_text())
    mixed = narr + stats + rules + multistep   # routing must handle all four types

    def acc(answerer, gold, judge_kind="same"):
        return round(run(answerer, gold, judge_kind=judge_kind)["accuracy"], 3)

    results = {
        # L1/L3 are internal to text-RAG -> tested on the arm over narrative gold.
        "L1_chunking": {
            "fixed": acc(TextRAGArm("dense", "fixed"), narr),
            "structural": acc(TextRAGArm("dense", "structural"), narr),
        },
        "L3_retrieval": {
            "dense": acc(TextRAGArm("dense", "fixed"), narr),
            "hybrid": acc(TextRAGArm("hybrid", "fixed"), narr),
        },
        # L5 routing -> full system over mixed gold.
        "L5_routing": {
            "rule": acc(get_router("rule"), mixed),
            "agent": acc(get_router("agent"), mixed),
        },
        # L6 judge -> same vs cross on the agent's narrative answers.
        "L6_judge": {
            "same": acc(get_router("agent"), narr, judge_kind="same"),
            "cross": acc(get_router("agent"), narr, judge_kind="cross"),
        },
        # L1 chunking again, but on the rules corpus — does structural chunking help
        # authoritative rule text the way it helps encyclopedic prose? Same isolation
        # as L1 (dense retrieval held), just source=rules over the rules gold.
        "rules_chunking": {
            "fixed": acc(TextRAGArm("dense", "fixed", source="rules"), rules),
            "structural": acc(TextRAGArm("dense", "structural", source="rules"), rules),
        },
        # Rules arm answering the authoritative-rules gold (judged vs Law-cited refs).
        "rules_arm": {
            "hybrid": acc(TextRAGArm("hybrid", "fixed", source="rules"), rules),
        },
        "sample_size": n,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="questions per track")
    main(ap.parse_args().n)
