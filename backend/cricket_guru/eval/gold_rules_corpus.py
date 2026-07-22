"""Corpus-grounded rules gold — the reference IS the rulebook clause, not an LLM draft.

For each substantive clause we ask a model to write a natural question answerable only from that
clause; the clause text is the authoritative answer. Only the QUESTION is model-written — the
reference is the law, so it can't be a fabrication (the failure mode of the old SE-drafted gold).

    cd backend && PYTHONPATH=. python -m cricket_guru.eval.gold_rules_corpus --n 35
"""
import argparse
import json
import re

from cricket_guru.config import DATA_DIR
from cricket_guru.index.chunking import chunk_structural
from cricket_guru.llm import agent

SOURCES = DATA_DIR / "rules" / "rules.json"
OUT = DATA_DIR / "gold" / "rules_gold_corpus.json"

QGEN = agent(
    "You are given ONE cricket rule clause from an official rulebook. Write ONE natural question "
    "a fan might ask whose answer is fully contained in that clause. The question must be "
    "answerable from the clause alone, must not mention the clause number, and must not quote it "
    "verbatim. Return ONLY the question.")

VERIFY = agent(
    "You are given a QUESTION and a rulebook CLAUSE. Reply 'yes' only if the clause contains the "
    "SPECIFIC fact the question asks for — the actual number, time limit, list, or condition — not "
    "merely the same topic. If the question asks 'within what time / how many / in what situations' "
    "and the clause discusses the subject but does not state that specific detail, reply 'no'. "
    "One word only.")

CLAUSE_START = re.compile(r"^\s*\d+\.\d+")


def _substantive(text):
    """A real rule clause, not a table-of-contents / header / page-number blob."""
    if not CLAUSE_START.match(text) or len(text) < 250:
        return False
    return sum(c.isdigit() for c in text) / len(text) < 0.15   # ToC chunks are mostly numbers


def main(n):
    docs = json.loads(SOURCES.read_text())
    clauses = []
    for d in docs:
        book = d["title"].strip().split(" (")[0]          # 'icc_odi (p49)' -> 'icc_odi'
        for c in chunk_structural(d["text"]):
            if _substantive(c):
                clauses.append((book, " ".join(c.split())))
    step = max(1, len(clauses) // (n * 2))                 # try ~2x candidates; keep n that verify
    candidates = clauses[::step]
    print(f"{len(clauses)} substantive clauses; trying up to {len(candidates)} to keep {n}", flush=True)

    items = []
    for book, clause in candidates:
        if len(items) >= n:
            break
        q = QGEN.run_sync(clause).output.strip()
        ok = VERIFY.run_sync(f"QUESTION: {q}\n\nCLAUSE: {clause[:1200]}").output.strip().lower()
        if not ok.startswith("yes"):                        # drop questions the clause doesn't answer
            continue
        items.append({"id": f"rules-c-{len(items)+1}", "qtype": "rules", "staleness": False,
                      "question": q, "reference": clause[:800], "source": book,
                      "verified": True})                    # reference is the law, not a draft
        print(f"[{len(items)}/{n}] {book}: {q[:70]}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(items, indent=2))
    print(f"wrote {len(items)} corpus-grounded rules items -> {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=35)
    main(ap.parse_args().n)
