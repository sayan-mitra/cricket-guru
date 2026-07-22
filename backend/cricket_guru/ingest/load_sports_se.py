#!/usr/bin/env python3
"""Load the Sports SE cricket snapshot (questions + answers) into Postgres.

Narrative eval oracle. Stored, never embedded into Qdrant. Keeps raw HTML plus
a cleaned-text body, and carries attribution (link, owner, date).
"""
import json
import os
import re
from datetime import datetime, timezone
from html import unescape
from psycopg2.extras import execute_values

from cricket_guru.config import DATA_DIR
from cricket_guru.db import connect

DATA = str(DATA_DIR / "sports_se")


def clean_html(raw):
    if not raw:
        return ""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.S)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</p>", "\n\n", text)
    text = re.sub(r"<li>", "\n- ", text)
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()


def ts(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None) if epoch else None


def main():
    questions = json.load(open(os.path.join(DATA, "questions.json")))
    answers = json.load(open(os.path.join(DATA, "answers.json")))

    q_rows = []
    for q in questions:
        owner = q.get("owner") or {}
        q_rows.append((
            q["question_id"], q.get("title"),
            clean_html(q.get("body")), q.get("body"),
            q.get("score"), q.get("view_count"),
            ",".join(q.get("tags", [])),
            q.get("accepted_answer_id"),
            owner.get("user_id"), owner.get("display_name"),
            ts(q.get("creation_date")), q.get("link"),
        ))

    a_rows = []
    for a in answers:
        owner = a.get("owner") or {}
        a_rows.append((
            a["answer_id"], a.get("question_id"),
            clean_html(a.get("body")), a.get("body"),
            a.get("score"), bool(a.get("is_accepted")),
            owner.get("user_id"), owner.get("display_name"),
            ts(a.get("creation_date")),
            f"https://sports.stackexchange.com/a/{a['answer_id']}",
        ))

    conn = connect()
    cur = conn.cursor()
    execute_values(cur,
        "INSERT INTO sports_se.questions "
        "(question_id,title,body,body_html,score,view_count,tags,"
        "accepted_answer_id,owner_user_id,owner_name,creation_date,link) "
        "VALUES %s", q_rows, page_size=500)
    # answers may reference a question outside our set only if paging differed;
    # our snapshot is self-consistent, so insert directly.
    execute_values(cur,
        "INSERT INTO sports_se.answers "
        "(answer_id,question_id,body,body_html,score,is_accepted,"
        "owner_user_id,owner_name,creation_date,link) VALUES %s",
        a_rows, page_size=500)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM sports_se.questions")
    nq = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM sports_se.answers")
    na = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM sports_se.answers WHERE is_accepted")
    nacc = cur.fetchone()[0]
    print(f"done: {nq} questions, {na} answers, {nacc} accepted")
    conn.close()


if __name__ == "__main__":
    main()
