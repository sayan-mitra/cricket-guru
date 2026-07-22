#!/usr/bin/env python3
"""Narrative gold from Sports SE accepted answers (the crowd oracle).

Curated to prose-shaped questions: substantial accepted answer, non-trivial
question. Deterministic sample. staleness stays False here — the build-time
web-audit that flags stale accepted answers is a separate, later step.

    python -m cricket_guru.eval.gold_narrative
"""
import json
import random
from html import unescape

from cricket_guru.config import DATA_DIR
from cricket_guru.db import connect

SEED = 42
N = 50
OUT = DATA_DIR / "gold" / "narrative_gold.json"


def main():
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT q.question_id, q.title, a.body, q.link, q.tags
        FROM sports_se.questions q
        JOIN sports_se.answers a ON a.answer_id = q.accepted_answer_id
        WHERE length(a.body) BETWEEN 300 AND 4000 AND q.score >= 1
        ORDER BY q.question_id
    """)
    rows = cur.fetchall()
    conn.close()

    picks = random.Random(SEED).sample(rows, min(N, len(rows)))
    gold = [{
        "id": f"narr-{i+1}", "qtype": "narrative", "staleness": False,
        "question": unescape(title or "").strip(),
        "reference": unescape(ref).strip(),
        "link": link, "tags": tags,
    } for i, (qid, title, ref, link, tags) in enumerate(picks)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gold, indent=2))
    print(f"wrote {len(gold)} narrative questions -> {OUT}")
    for g in gold[:4]:
        print(f"  [{g['id']}] {g['question'][:75]}")


if __name__ == "__main__":
    main()
