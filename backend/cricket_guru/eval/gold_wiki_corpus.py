"""Corpus-grounded narrative gold — the reference IS the Wikipedia passage, not an LLM draft.

Mirror of gold_rules_corpus for prose: for each substantive passage a model writes a question the
passage answers, and the passage is the authoritative reference. Only the QUESTION is model-written,
so the reference can't be a fabrication — the failure mode of the old SE-drafted narrative gold that
kept the wiki chunking/retrieval legs (L1/L3/L6) reading flat.

    cd backend && PYTHONPATH=. python -m cricket_guru.eval.gold_wiki_corpus --n 25
"""
import argparse
import json

from cricket_guru.config import DATA_DIR
from cricket_guru.index.chunking import chunk_structural
from cricket_guru.llm import agent

SOURCES = DATA_DIR / "wikipedia" / "articles.json"
OUT = DATA_DIR / "gold" / "narrative_gold_corpus.json"

QGEN = agent(
    "You are given ONE passage from a Wikipedia article about cricket. Write ONE natural question a "
    "fan might ask whose answer is fully contained in that passage. The question must be answerable "
    "from the passage alone, must name its specific subject (not 'this match' or 'the player'), and "
    "must not quote the passage verbatim. Return ONLY the question.")

VERIFY = agent(
    "You are given a QUESTION and a Wikipedia PASSAGE. Reply 'yes' only if the passage contains the "
    "SPECIFIC fact the question asks for — the actual name, number, date, or outcome — not merely the "
    "same topic. Otherwise reply 'no'. One word only.")


def _substantive(text):
    """Real prose, not a stub, list, or stat table."""
    t = text.strip()
    if len(t) < 350 or t.count(".") < 3:
        return False
    return sum(c.isdigit() for c in t) / len(t) < 0.20   # skip tables / scorecards


def main(n):
    arts = json.loads(SOURCES.read_text())
    passages = []
    for a in arts:
        for c in chunk_structural(a["text"]):
            if _substantive(c):
                passages.append((a["title"], " ".join(c.split())))
    step = max(1, len(passages) // (n * 3))              # spread the candidates; keep n that verify
    candidates = passages[::step]
    print(f"{len(passages)} substantive passages; trying up to {len(candidates)} to keep {n}", flush=True)

    items, seen = [], set()
    for title, passage in candidates:
        if len(items) >= n:
            break
        if title in seen:                                 # at most one question per article, for spread
            continue
        q = QGEN.run_sync(passage).output.strip()
        ok = VERIFY.run_sync(f"QUESTION: {q}\n\nPASSAGE: {passage[:1400]}").output.strip().lower()
        if not ok.startswith("yes"):                      # drop questions the passage doesn't answer
            continue
        seen.add(title)
        items.append({"id": f"narr-c-{len(items)+1}", "qtype": "narrative", "staleness": False,
                      "question": q, "reference": passage[:900], "source": title, "verified": True})
        print(f"[{len(items)}/{n}] {title[:40]}: {q[:70]}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(items, indent=2))
    print(f"wrote {len(items)} corpus-grounded narrative items -> {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    main(ap.parse_args().n)
