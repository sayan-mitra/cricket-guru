"""v2 serving baseline: run the full serving path (serving_engine — guards + agent + tools + critic)
over the gold set and record, per query, total latency, the per-leg span times, the critic verdict,
and whether the answer is correct (stats by substring, narrative/rules/multistep by the judge). One
pass gives accuracy AND latency on the true product path, so re-running after each v2 optimization
shows the before/after. Writes data/results/serving_baseline.json (answers included, so re-scoring
never needs another serving run).

    set -a; source backend/.env; set +a
    PYTHONPATH=backend python -m cricket_guru.eval.latency_profile [--n N_PER_TYPE]
"""
import argparse
import json
import statistics as st
from collections import Counter, defaultdict

from cricket_guru.config import DATA_DIR
from cricket_guru.eval.harness import _composed, _stats_correct
from cricket_guru.eval.judge import make_judge
from cricket_guru.serve import serving_engine

GOLD = {
    "stats": "stats_gold.json",
    "narrative": "narrative_gold_corpus.json",
    "rules": "rules_gold_corpus.json",
    "multistep": "multistep_gold.json",
}


def _spans(base):
    return getattr(base, "trace", None) or getattr(base, "spans", None) or []


def main(n, out_name="serving_baseline.json"):
    eng = serving_engine()
    judge = make_judge("cross")   # Haiku — cheap, and Phase-B showed it agrees with the human
    rows = []
    for qtype, fn in GOLD.items():
        items = json.loads((DATA_DIR / "gold" / fn).read_text())
        if n:
            items = items[:n]
        for it in items:
            q = it["question"]
            try:
                r = eng.serve(q, [])
                total = getattr(r, "latency_ms", None) or getattr(r, "ms", 0)
                legs = defaultdict(float)
                for s in _spans(r.base):
                    legs[s.get("name", "?")] += (s.get("ms") or 0)
                abstained = "rather not answer" in (r.text or "").lower()
                if qtype == "stats":
                    ok = _stats_correct(r.text, it)
                elif qtype == "multistep":
                    ok = judge(q, r.text, it["reference"]).correct and _composed(r.base, it)
                else:
                    ok = judge(q, r.text, it["reference"]).correct
                ok = ok and not abstained   # an abstain is a miss — the user didn't get the answer
                spans = [{k: (v[:600] + "…" if isinstance(v, str) and len(v) > 600 else v)
                          for k, v in s.items()} for s in _spans(r.base)]
                rows.append({"qtype": qtype, "id": it.get("id"), "total_ms": total,
                             "verdict": getattr(r, "verdict", "?"), "correct": bool(ok),
                             "abstained": abstained, "tools": [s.get("name") for s in spans
                                                               if s.get("kind") == "tool"],
                             "legs": dict(legs), "spans": spans, "answer": r.text})
                print(f"  {qtype:10} {it.get('id')}: {total/1000:5.1f}s  "
                      f"[{getattr(r,'verdict','?'):11}] {'OK  ' if ok else 'MISS'}", flush=True)
            except Exception as e:
                rows.append({"qtype": qtype, "id": it.get("id"), "total_ms": None,
                             "verdict": "ERROR", "correct": False, "error": f"{type(e).__name__}: {e}"})
                print(f"  {qtype:10} {it.get('id')}: ERROR {type(e).__name__}: {e}", flush=True)

    out = DATA_DIR / "results" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0

    print("\n=== accuracy by type ===")
    for qtype in list(GOLD) + ["ALL"]:
        oks = [r["correct"] for r in rows if r["verdict"] != "ERROR"
               and (qtype == "ALL" or r["qtype"] == qtype)]
        if oks:
            print(f"{qtype:10} {sum(oks):2}/{len(oks):<2} = {sum(oks)/len(oks):.0%}")

    print("\n=== total latency by type (seconds) ===")
    for qtype in list(GOLD) + ["ALL"]:
        ts = [r["total_ms"] / 1000 for r in rows if r["total_ms"]
              and (qtype == "ALL" or r["qtype"] == qtype)]
        if ts:
            print(f"{qtype:10} n={len(ts):3}  mean={st.mean(ts):5.1f}  med={st.median(ts):5.1f}  "
                  f"p95={pct(ts,0.95):5.1f}  max={max(ts):5.1f}")

    legsum = defaultdict(list)
    for r in rows:
        for k, v in (r.get("legs") or {}).items():
            legsum[k].append(v / 1000)
    print("\n=== mean latency per leg (seconds), slowest first ===")
    for k, v in sorted(legsum.items(), key=lambda x: -st.mean(x[1])):
        print(f"{k:18} mean={st.mean(v):5.1f}  n={len(v)}")

    print("\n=== verdict distribution ===")
    for v, c in Counter(r["verdict"] for r in rows).most_common():
        print(f"{v:12} {c}")
    ab = sum(1 for r in rows if r.get("abstained"))
    print(f"\nabstained (counted as misses): {ab}/{len(rows)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="cap items per type (0 = all)")
    ap.add_argument("--out", default="serving_baseline.json", help="output filename under data/results")
    args = ap.parse_args()
    main(args.n, args.out)
