"""Run an answerer over a gold set and score it.

Stats questions are scored objectively (the answer label/number must appear in
the response). Narrative questions are scored by the LLM-judge. Any object with
an `.answer(question) -> Answer` method works — an arm, or a router.
"""
import json

from cricket_guru.config import DATA_DIR
from cricket_guru.eval.judge import make_judge


def load_gold(name):
    return json.loads((DATA_DIR / "gold" / f"{name}_gold.json").read_text())


def _stats_correct(text, item):
    t = text.lower()
    return str(item["answer_label"]).lower() in t or str(item["answer_value"]) in text


def _composed(ans, item):
    """Did the trace show the decomposition? `expected_tools` (strong form) requires those distinct
    tool TYPES to all appear — a cross-domain question the rule-router can't do and one SQL can't
    shortcut. Else fall back to a raw min_tools count (weaker: one clever query can combine facts)."""
    used = [s["name"] for s in (ans.trace or []) if s.get("kind") == "tool"]
    if item.get("expected_tools"):
        return set(item["expected_tools"]) <= set(used)
    return len(used) >= item.get("min_tools", 2)


def run(answerer, gold, judge_kind="same", limit=None):
    judge = make_judge(judge_kind)
    items = gold[:limit] if limit else gold
    results = []
    for item in items:
        ans = answerer.answer(item["question"])
        if item["qtype"] == "stats":
            ok = _stats_correct(ans.text, item)
        elif item["qtype"] == "multistep":
            # correct (judged vs reference) AND actually composed (the ReAct verification)
            ok = judge(item["question"], ans.text, item["reference"]).correct and _composed(ans, item)
        else:
            ok = judge(item["question"], ans.text, item["reference"]).correct
        results.append({
            "id": item["id"], "qtype": item["qtype"], "staleness": item.get("staleness"),
            "ok": bool(ok), "answer": ans.text, "trace": ans.tool_trace,
        })
    acc = sum(r["ok"] for r in results) / len(results) if results else 0.0
    return {"accuracy": acc, "n": len(results), "results": results}
