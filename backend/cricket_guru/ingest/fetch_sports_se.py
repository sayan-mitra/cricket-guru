#!/usr/bin/env python3
"""Snapshot the Sports Stack Exchange 'cricket' tag (questions + answers) to JSON.

Uses the public Stack Exchange API (no key needed for this volume). We freeze the
data to disk so the corpus is stable for us, and carry attribution fields
(link, owner, creation_date) through for CC-BY-SA compliance.
"""
import gzip
import json
import time
import urllib.request
from cricket_guru.config import DATA_DIR

SITE = "sports"
TAG = "cricket"
BASE = "https://api.stackexchange.com/2.3"
OUT = DATA_DIR / "sports_se"


def get(url):
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
    return json.loads(data)


def paged(url_no_page):
    items, page = [], 1
    while True:
        d = get(f"{url_no_page}&page={page}")
        items.extend(d.get("items", []))
        if d.get("backoff"):
            time.sleep(d["backoff"] + 1)
        if not d.get("has_more"):
            break
        page += 1
        time.sleep(0.3)
        if page > 40:  # safety valve
            break
    return items


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    q_url = (f"{BASE}/questions?pagesize=100&order=asc&sort=creation"
             f"&tagged={TAG}&site={SITE}&filter=withbody")
    questions = paged(q_url)

    qids = [q["question_id"] for q in questions]
    answers = []
    for i in range(0, len(qids), 100):
        ids = ";".join(str(x) for x in qids[i:i + 100])
        a_url = (f"{BASE}/questions/{ids}/answers?pagesize=100&order=asc"
                 f"&sort=creation&site={SITE}&filter=withbody")
        answers.extend(paged(a_url))
        time.sleep(0.3)

    (OUT / "questions.json").write_text(json.dumps(questions, indent=2))
    (OUT / "answers.json").write_text(json.dumps(answers, indent=2))

    accepted = sum(1 for q in questions if q.get("accepted_answer_id"))
    print(f"questions: {len(questions)}  answers: {len(answers)}  "
          f"questions_with_accepted: {accepted}")


if __name__ == "__main__":
    main()
