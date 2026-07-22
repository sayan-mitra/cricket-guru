"""Run an answerer over a gold set and score it.

Stats questions are scored objectively (the answer label/number must appear in
the response). Narrative questions are scored by the LLM-judge. Any object with
an `.answer(question) -> Answer` method works — an arm, or a router.
"""
import json
import os
import signal
from contextlib import contextmanager

from cricket_guru.config import DATA_DIR
from cricket_guru.eval.judge import make_judge

ITEM_TIMEOUT = int(os.environ.get("CG_ITEM_TIMEOUT", "300"))


@contextmanager
def _deadline(seconds):
    """A hard ceiling on one gold item, enforced from the main thread.

    The per-request LLM timeout doesn't cover this: the router answers by calling run_sync inside
    sync tools, which pydantic-ai runs on worker threads, and a run has twice frozen mid-leg with
    dead sockets and no CPU burn. SIGALRM lands in the main thread and unwinds the item so the rest
    of the run survives. Main-thread only — that is where run() is called from.
    """
    def _fire(signum, frame):
        raise TimeoutError(f"item exceeded {seconds}s (stuck tool call)")
    old = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


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


def run(answerer, gold, judge_kind="same", limit=None, label=""):
    judge = make_judge(judge_kind)
    items = gold[:limit] if limit else gold
    results = []
    for n, item in enumerate(items, 1):
        # A run is hundreds of network calls; one timeout must not throw the other 99% away.
        # The item counts as wrong and the error rides along in the answer so it's visible after.
        err = None
        try:
            with _deadline(ITEM_TIMEOUT):
                ans = answerer.answer(item["question"])
                if item["qtype"] == "stats":
                    ok = _stats_correct(ans.text, item)
                elif item["qtype"] == "multistep":
                    # correct (judged vs reference) AND actually composed (the ReAct verification)
                    ok = judge(item["question"], ans.text, item["reference"]).correct and _composed(ans, item)
                else:
                    ok = judge(item["question"], ans.text, item["reference"]).correct
                text, trace = ans.text, ans.tool_trace
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {e}"
            text, trace = f"[FAILED] {err}", []
        print(f"  [{label or 'run'}] {n}/{len(items)} {item['id']} "
              f"{'ok' if ok else 'MISS'}{' ' + err if err else ''}", flush=True)
        results.append({
            "id": item["id"], "qtype": item["qtype"], "staleness": item.get("staleness"),
            "ok": bool(ok), "answer": text, "trace": trace, "error": err,
        })
    acc = sum(r["ok"] for r in results) / len(results) if results else 0.0
    return {"accuracy": acc, "n": len(results), "results": results}
