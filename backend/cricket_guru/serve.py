"""Serving layer — the CRAG critic wrapped around a responder, plus the
comparison fan-out. One engine, two renderers (design doc: 'Frontend').

  Engine.serve(q)   -> Mode A: one answer; the critic decides ship / web / abstain
  compare(q, axis)  -> Mode B: the same engine across lanes, display-only

An 'engine' wraps anything with .answer(question) -> Answer — an arm or a router —
so the chunking axis can compare arms directly (no router, matching how the eval
isolates that leg) while the router axis compares full systems.
"""
import time
from dataclasses import dataclass
from functools import lru_cache

from cricket_guru import config, critic, gold_match
from cricket_guru.arms.base import Answer
from cricket_guru.arms.text_rag import TextRAGArm
from cricket_guru.routing import get_router
from cricket_guru.tools.web_search import web_search_frozen

ABSTAIN = ("I'd rather not answer than risk being wrong — I can't verify this "
           "from my cricket sources.")
WEB_CAVEAT = ("Verified from a live web search, not our curated corpus — treat it "
              "with that caveat.")


def _abstain(base, reason):
    """Abstain with the reason surfaced, so the user sees WHY, and record it in the trace."""
    text = f"{ABSTAIN}\n\n**Why:** {reason}"
    base.trace.append({"kind": "action", "name": "abstain", "parent": "critic", "ms": 0,
                       "input": reason, "output": text})
    return text


@dataclass
class ServeResult:
    text: str                 # what the user sees, after the fail-action
    verdict: str              # ok | retrieval_gap | hallucination | blocked
    reason: str
    base: Answer              # the pre-critic answer (tools, sources, trace)
    latency_ms: int = 0       # whole serve incl. critic + any fallback
    label: str = ""           # lane label in the comparison view


class Engine:
    def __init__(self, responder, label=""):
        self.responder = responder
        self.label = label

    def serve(self, question, history=None) -> ServeResult:
        t0 = time.perf_counter()
        # history only to responders that accept it (the agent router); Mode B arms don't.
        base = self.responder.answer(question, history) if history else self.responder.answer(question)
        if base.blocked:
            ms = int((time.perf_counter() - t0) * 1000)
            return ServeResult(base.text, "blocked", base.blocked, base, ms, self.label)

        tc = time.perf_counter()
        v = critic.critique(question, base)
        base.trace.append({                               # the critic is a serving step — trace it
            "kind": "critic", "name": "critic", "ms": int((time.perf_counter() - tc) * 1000),
            "input": f"candidate answer + retrieval_score={base.retrieval_score}",
            "output": f"{v.verdict} — {v.reason}"})

        # The output guard is a gate, not just a metric. Its verdict used to be recorded and then
        # ignored, which is how '242 runs' shipped with the contradicting '31' row sitting in the same
        # evidence blob. A rejected answer abstains whatever the critic makes of its scope. We require
        # evidence to exist first, so 'nothing gathered' stays the critic's call, not an auto-abstain.
        if base.grounded is False and base.evidence.strip():
            v = critic.Verdict(critic.HALLUCINATION, base.grounded_reason or
                               "the evidence doesn't support the answer")
            base.trace.append({"kind": "critic", "name": "output_guard_gate", "ms": 0,
                               "input": "grounded=False", "output": f"forced {v.verdict} — {v.reason}"})

        text = base.text
        # A retrieval_gap with no retrieval score is a stats/all-time record beyond the data window.
        # The web has repeatedly handed back wrong precise figures for these (272, Warne 708, Anderson
        # 676…), so abstain honestly with the reason rather than ship a possibly-wrong number. Web
        # fallback stays for RAG/freshness gaps (a retrieval score is present), where it genuinely fills
        # what the corpus lacks.
        if v.verdict == critic.RETRIEVAL_GAP and base.retrieval_score is None:
            text = _abstain(base, v.reason)
        elif v.verdict == critic.RETRIEVAL_GAP:
            used_web = any(sp.get("name") == "web_search" for sp in base.trace)
            if used_web:
                if "web" not in base.text.lower():        # agent forgot the caveat
                    text = f"{WEB_CAVEAT}\n\n{base.text}"
            else:                                          # simple arm never checks the web itself
                tw = time.perf_counter()
                web = web_search_frozen(question)
                text = f"{WEB_CAVEAT}\n\n{web}"
                base.trace.append({                       # the fallback replaces the answer — trace it
                    "kind": "tool", "name": "web_fallback", "parent": "critic",
                    "ms": int((time.perf_counter() - tw) * 1000),
                    "input": question, "output": f"[replaces the corpus answer]\n{web}"})
        elif v.verdict == critic.HALLUCINATION:
            text = _abstain(base, v.reason)

        ms = int((time.perf_counter() - t0) * 1000)
        return ServeResult(text, v.verdict, v.reason, base, ms, self.label)


@lru_cache(maxsize=1)
def serving_engine():
    """Mode A — the product path on the best-known config (structural + agent).
    Cached: constructing it loads Qdrant collections into memory once."""
    return Engine(get_router("agent", retrieval="hybrid", chunking="structural",
                             rules_retrieval="dense",    # BM25 pulls lexical noise on rulebooks
                             rerank=True),                # cross-encoder rerank on the wiki arm (+20 @1)
                  "structural + agent")


# Mode B lanes. One axis varies; the rest hold at baseline so each lane reads as
# 'exactly what this leg buys' — the same isolation the eval ablation uses.
# Cached per axis so the swimlane doesn't reload collections on every query.
@lru_cache(maxsize=2)
def _lanes(axis):
    if axis == "chunking":        # L1: internal to the text-RAG arm, no router
        return (Engine(TextRAGArm("dense", "fixed", "wiki"), "fixed chunking"),
                Engine(TextRAGArm("dense", "structural", "wiki"), "structural chunking"))
    if axis == "router":          # L5: the full system
        return (Engine(get_router("rule", "dense", "fixed"), "rule router"),
                Engine(get_router("agent", "dense", "fixed"), "agent router"))
    raise ValueError(f"unknown axis: {axis}")


def compare(question, axis="router"):
    """Mode B — run each lane plus the gold lane. Display-only; no winner picked."""
    lanes = [_lane_view(e.serve(question)) for e in _lanes(axis)]
    return {"axis": axis, "lanes": lanes, "gold": gold_match.match(question)}


def _lane_view(r: ServeResult):
    b = r.base
    return {"label": r.label, "verdict": r.verdict, "reason": r.reason, "text": r.text,
            "tools": b.tool_trace, "sources": b.sources,
            "retrieval_score": b.retrieval_score, "latency_ms": r.latency_ms,
            "tokens": b.tokens}


def _cli():
    """Run the serving path from the terminal — no Streamlit, no Qdrant-lock dance.

        python -m cricket_guru.serve "Is there a free hit after a no-ball?"
        python -m cricket_guru.serve --compare router "Who scored most in IPL 2016?"
    """
    import argparse
    ap = argparse.ArgumentParser(description="Serve one question (Mode A) or fan out (Mode B).")
    ap.add_argument("question")
    ap.add_argument("--compare", choices=["router", "chunking"],
                    help="Mode B: show all lanes over this axis instead of one answer")
    args = ap.parse_args()

    if args.compare:
        r = compare(args.question, args.compare)
        print(f"# compare · axis={r['axis']}")
        for L in r["lanes"]:
            meta = [f"{L['latency_ms']/1000:.1f}s"]
            if L["retrieval_score"] is not None:
                meta.append(f"match {L['retrieval_score']:.2f}")
            if L["tokens"]:
                meta.append(f"{L['tokens']} tok")
            print(f"\n[{L['label']}]  {L['verdict']}  ({', '.join(meta)})")
            print("  tools:", " -> ".join(L["tools"]))
            print("  " + L["text"][:400].replace("\n", "\n  "))
        g = r["gold"]
        print("\n[gold]", f"{g['id']} (sim {g['score']:.2f}): {g['reference'][:200]}"
              if g else "no gold hit")
    else:
        r = serving_engine().serve(args.question)
        print(f"# verdict: {r.verdict}  ({r.latency_ms/1000:.1f}s)")
        print(f"# critic:  {r.reason}")
        print(f"# tools:   {' -> '.join(r.base.tool_trace)}")
        if r.base.retrieval_score is not None:
            print(f"# match:   {r.base.retrieval_score:.2f}")
        print("\n" + r.text)


if __name__ == "__main__":
    _cli()
