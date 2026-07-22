#!/usr/bin/env python3
"""Run the comparative-analysis legs and save results for the dashboard.

Ablation: baseline = fixed chunking / dense retrieval / rule router / same judge.
Each leg flips one variant against that baseline. Sample size is small by default
to bound LLM cost; raise --n for tighter numbers.

    python -m cricket_guru.eval.run_experiments --n 10
"""
import argparse
import json

from cricket_guru import config
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

    def acc(answerer, gold, judge_kind="same", label=""):
        print(f"[leg] {label} — {len(gold)} questions", flush=True)
        a = round(run(answerer, gold, judge_kind=judge_kind, label=label)["accuracy"], 3)
        print(f"[leg] {label} = {a}", flush=True)
        return a

    results = {
        # L1/L3 are internal to text-RAG -> tested on the arm over narrative gold.
        "L1_chunking": {
            "fixed": acc(TextRAGArm("dense", "fixed"), narr, label="L1.fixed"),
            "structural": acc(TextRAGArm("dense", "structural"), narr, label="L1.structural"),
        },
        "L3_retrieval": {
            "dense": acc(TextRAGArm("dense", "fixed"), narr, label="L3.dense"),
            "hybrid": acc(TextRAGArm("hybrid", "fixed"), narr, label="L3.hybrid"),
        },
        # L5 routing -> full system over mixed gold.
        "L5_routing": {
            "rule": acc(get_router("rule"), mixed, label="L5.rule"),
            "agent": acc(get_router("agent"), mixed, label="L5.agent"),
        },
        # L6 judge -> same vs cross on the agent's narrative answers.
        "L6_judge": {
            "same": acc(get_router("agent"), narr, judge_kind="same", label="L6.same"),
            "cross": acc(get_router("agent"), narr, judge_kind="cross", label="L6.cross"),
        },
        # L1 chunking again, but on the rules corpus — does structural chunking help
        # authoritative rule text the way it helps encyclopedic prose? Same isolation
        # as L1 (dense retrieval held), just source=rules over the rules gold.
        "rules_chunking": {
            "fixed": acc(TextRAGArm("dense", "fixed", source="rules"), rules, label="rules.fixed"),
            "structural": acc(TextRAGArm("dense", "structural", source="rules"), rules, label="rules.structural"),
        },
        # Rules arm answering the authoritative-rules gold (judged vs Law-cited refs).
        "rules_arm": {
            "hybrid": acc(TextRAGArm("hybrid", "fixed", source="rules"), rules, label="rules.arm"),
        },
        "sample_size": n,
        # Which models produced these numbers. Without this a results file can't be compared to the
        # next one — the earlier figures were measured on a different answerer and nothing recorded it.
        "models": {"answerer": config.ANSWERER_MODEL, "critic": config.CRITIC_MODEL,
                   "judge_same": config.ANSWERER_MODEL, "judge_cross": config.JUDGE_CROSS_MODEL},
        # L6 asks whether a model marks its own homework generously. It is only a cross-VENDOR check
        # when judge_cross comes from a different provider; same-vendor pairs still share training, so
        # read the gap as a floor on self-preference, not a measurement of it.
        "judge_note": ("cross-vendor" if config.JUDGE_CROSS_MODEL.split(":")[0]
                       != config.ANSWERER_MODEL.split(":")[0] else "same-vendor, different model"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="questions per track")
    main(ap.parse_args().n)
