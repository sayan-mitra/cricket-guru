#!/usr/bin/env python3
"""Load Cricsheet match JSON into the cricsheet Postgres schema.

Streams each match file into matches / innings / deliveries / player_lineups.
Deliveries are the big table (~4M rows), so they're flushed in batches.
"""
import glob
import json
import os
from psycopg2.extras import execute_values

from cricket_guru.config import DATA_DIR
from cricket_guru.db import connect

DATA = str(DATA_DIR / "cricsheet")
FOLDERS = {  # extracted dir -> format label
    "tests_male": "Test",
    "odis_male": "ODI",
    "t20s_male": "T20I",
    "ipl_male": "IPL",
}
BATCH = 50_000


def first(x, default=None):
    return x[0] if isinstance(x, list) and x else default


def parse_match(path, fmt):
    with open(path) as f:
        d = json.load(f)
    info = d["info"]
    match_id = int(os.path.splitext(os.path.basename(path))[0])
    teams = info.get("teams", [None, None])
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})
    event = info.get("event")
    event_name = event.get("name") if isinstance(event, dict) else None

    match_row = (
        match_id, fmt, info.get("match_type"), info.get("gender"),
        info.get("season"), event_name, first(info.get("dates")),
        info.get("venue"), info.get("city"),
        teams[0] if len(teams) > 0 else None,
        teams[1] if len(teams) > 1 else None,
        info.get("toss", {}).get("winner"),
        info.get("toss", {}).get("decision"),
        outcome.get("winner"),
        outcome.get("result", "normal") if "winner" not in outcome else "normal",
        by.get("runs"), by.get("wickets"),
        first(info.get("player_of_match")),
        info.get("balls_per_over", 6),
    )

    lineups = []
    for team, players in (info.get("players") or {}).items():
        for p in players:
            lineups.append((match_id, team, p))

    innings_rows, delivery_rows = [], []
    for i, inn in enumerate(d.get("innings", []), start=1):
        bat = inn.get("team")
        bowl = next((t for t in teams if t != bat), None)
        runs = wkts = 0
        for over in inn.get("overs", []):
            over_no = over.get("over")
            for b, ball in enumerate(over.get("deliveries", []), start=1):
                r = ball.get("runs", {})
                runs += r.get("total", 0)
                extras = ball.get("extras") or {}
                extra_type = next(iter(extras), None)
                wickets = ball.get("wickets") or []
                wkts += len(wickets)
                w = wickets[0] if wickets else {}
                fielders = w.get("fielders")
                fielders = ",".join(fl.get("name", "") for fl in fielders) if fielders else None
                delivery_rows.append((
                    match_id, i, over_no, b, bat,
                    ball.get("batter"), ball.get("bowler"), ball.get("non_striker"),
                    r.get("batter", 0), r.get("extras", 0), r.get("total", 0),
                    extra_type, w.get("kind"), w.get("player_out"),
                ))
        innings_rows.append((match_id, i, bat, bowl, runs, wkts))

    return match_row, innings_rows, delivery_rows, lineups


def main():
    conn = connect()
    cur = conn.cursor()
    m_buf, i_buf, l_buf, d_buf = [], [], [], []
    n_matches = 0

    def flush_deliveries():
        if d_buf:
            execute_values(cur,
                "INSERT INTO cricsheet.deliveries "
                "(match_id,innings_no,over_no,ball_no,batting_team,batter,bowler,"
                "non_striker,runs_batter,runs_extras,runs_total,extra_type,"
                "wicket_kind,player_out) VALUES %s", d_buf, page_size=10_000)
            d_buf.clear()

    for folder, fmt in FOLDERS.items():
        files = [f for f in glob.glob(os.path.join(DATA, folder, "*.json"))
                 if "README" not in f]
        for path in files:
            mrow, irows, drows, lrows = parse_match(path, fmt)
            m_buf.append(mrow)
            i_buf.extend(irows)
            l_buf.extend(lrows)
            d_buf.extend(drows)
            n_matches += 1
            if len(d_buf) >= BATCH:
                flush_deliveries()
        print(f"{folder}: parsed {len(files)} matches")

    execute_values(cur,
        "INSERT INTO cricsheet.matches "
        "(match_id,format,match_type,gender,season,event_name,match_date,venue,"
        "city,team1,team2,toss_winner,toss_decision,winner,result,win_by_runs,"
        "win_by_wickets,player_of_match,balls_per_over) VALUES %s",
        m_buf, page_size=1000)
    execute_values(cur,
        "INSERT INTO cricsheet.innings "
        "(match_id,innings_no,batting_team,bowling_team,runs,wickets) VALUES %s",
        i_buf, page_size=1000)
    execute_values(cur,
        "INSERT INTO cricsheet.player_lineups (match_id,team,player) VALUES %s",
        l_buf, page_size=5000)
    flush_deliveries()

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM cricsheet.deliveries")
    n_deliv = cur.fetchone()[0]
    print(f"\ndone: {n_matches} matches, {len(i_buf)} innings, "
          f"{len(l_buf)} lineup rows, {n_deliv:,} deliveries")
    conn.close()


if __name__ == "__main__":
    main()
