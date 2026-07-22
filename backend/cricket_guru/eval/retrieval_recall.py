"""Direct retrieval metric — recall@k against the known gold clause.

The corpus-grounded rules gold gives us the exact reference clause, so we can measure retrieval
without the LLM/judge downstream: ask the question, take the top-k chunks, check whether the gold
clause is among them. Low-noise and cheap, and it moves when chunking/retrieval actually improve —
unlike end-to-end answer accuracy, where a retrieval gain drowns in the answerer + judge.

    cd backend && PYTHONPATH=. python -m cricket_guru.eval.retrieval_recall --mode loose
"""
import argparse
import json
import re

from cricket_guru.config import DATA_DIR
from cricket_guru.retrieval.base import get_retriever

GOLDS = {"rules": DATA_DIR / "gold" / "rules_gold_corpus.json",
         "wiki": DATA_DIR / "gold" / "narrative_gold_corpus.json"}
OUT = DATA_DIR / "results" / "recall.json"
CLAUSE_ID = re.compile(r"^\s*(\d+\.\d+(?:\.\d+)*)")


def _toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _is_hit(gold_ref, chunk, mode):
    """Did this retrieved chunk surface the gold clause?"""
    if mode == "strict":
        m = CLAUSE_ID.match(gold_ref)                      # the clause number, e.g. 35.2
        if m and m.group(1) in chunk[:80]:
            return True
        g = " ".join(gold_ref.split())                     # or a near-verbatim opening
        return g[:80] in " ".join(chunk.split())
    gt = _toks(gold_ref)                                   # loose: chunk covers most of its words
    return len(gt & _toks(chunk)) / max(1, len(gt)) >= 0.6


def recall(source, chunking, retrieval, mode="loose", ks=(1, 3, 5), rerank_on=False):
    items = json.loads(GOLDS[source].read_text())
    r = get_retriever(retrieval, chunking, source)
    kmax = max(ks)
    fetch = 20 if rerank_on else kmax                      # rerank a wider net back down to the top-k
    got = {k: 0 for k in ks}
    if rerank_on:
        from cricket_guru.retrieval.rerank import rerank
    for it in items:
        hits = r.search(it["question"], k=fetch)
        if rerank_on:
            hits = rerank(it["question"], hits, top_k=kmax)
        ranks = [i for i, h in enumerate(hits) if _is_hit(it["reference"], h.text, mode)]
        first = ranks[0] if ranks else None                # rank of the first correct chunk
        for k in ks:
            if first is not None and first < k:
                got[k] += 1
    n = len(items)
    return {k: got[k] / n for k in ks}, n


def compare_rerank(mode="loose"):
    """Print base vs cross-encoder rerank recall side by side — measure before wiring into serving."""
    for source in ("rules", "wiki"):
        print(f"\nRetrieval recall@k — {source} (match={mode})    base → +reranker")
        for chunking in ("fixed", "structural"):
            for retrieval in ("dense", "hybrid"):
                base, n = recall(source, chunking, retrieval, mode, rerank_on=False)
                rr, _ = recall(source, chunking, retrieval, mode, rerank_on=True)
                cells = "   ".join(f"@{k} {base[k]:.0%}→{rr[k]:.0%}" for k in (1, 3, 5))
                print(f"  {chunking:11s} {retrieval:7s} (n={n})   {cells}", flush=True)


def main(mode="loose"):
    out = {"mode": mode, "sources": []}
    for source in ("rules", "wiki"):
        print(f"\nRetrieval recall@k — {source} corpus (match={mode})")
        grid, n = [], 0
        for chunking in ("fixed", "structural"):
            for retrieval in ("dense", "hybrid"):
                rec, n = recall(source, chunking, retrieval, mode)
                rr, _ = recall(source, chunking, retrieval, mode, rerank_on=True)
                grid.append({"chunking": chunking, "retrieval": retrieval,
                             "recall": {str(k): rec[k] for k in (1, 3, 5)},
                             "reranked": {str(k): rr[k] for k in (1, 3, 5)}})
                cells = "   ".join(f"@{k} {rec[k]:.0%}→{rr[k]:.0%}" for k in (1, 3, 5))
                print(f"  {chunking:11s} {retrieval:7s} (n={n})    {cells}")
        out["sources"].append({"source": source, "n": n, "grid": grid})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["loose", "strict"], default="loose")
    ap.add_argument("--rerank", action="store_true", help="compare base vs cross-encoder rerank")
    args = ap.parse_args()
    (compare_rerank if args.rerank else main)(args.mode)
