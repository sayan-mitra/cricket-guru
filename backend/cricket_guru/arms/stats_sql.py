"""Arm 2 — stats-SQL. Text-to-SQL over Cricsheet, execute, phrase the result.

Read-only: only a single SELECT is ever run against Postgres.
"""
import re
from datetime import date, timedelta

from cricket_guru.arms.base import Answer
from cricket_guru.db import connect
from cricket_guru.llm import agent

SCHEMA = """cricsheet — 4 tables. Row granularity and column meaning both matter; read them before writing SQL.

matches — one row per match
  match_id                     primary key
  format                       'Test' | 'ODI' | 'T20I' | 'IPL'
  season                       e.g. '2023' or '2007/08'
  event_name                   series / tournament, e.g. 'India tour of England', 'ICC World Cup'
  match_date                   date
  venue, city                  ground and city
  team1, team2                 the two sides, full names (e.g. 'India')
  toss_winner, toss_decision
  winner                       winning team; NULL = tie / draw / no result
  result                       'normal' | 'tie' | 'draw' | 'no result'
  win_by_runs, win_by_wickets  victory margin — only one is set, per how they won
  player_of_match

innings — one row per innings (usually two per limited-overs match)
  match_id, innings_no
  batting_team, bowling_team   BOTH live here (deliveries has batting_team only)
  runs                         the FULL innings total, already summed — NOT a per-ball value
  wickets                      wickets lost in the innings

deliveries — one row per ball bowled
  match_id, innings_no, over_no, ball_no    ball location
  batting_team                 (no bowling_team column here — it is on innings)
  batter, bowler, non_striker
  runs_batter                  runs off the bat on THIS ball (0-6). A batter's innings/match total is
                               SUM(runs_batter) GROUP BY (match_id, innings_no, batter). A six = 6.
  runs_extras                  extras conceded on this ball
  runs_total                   runs_batter + runs_extras on this ball
  extra_type                   'wides' | 'noballs' | 'byes' | 'legbyes' | NULL
  wicket_kind                  dismissal type if a wicket fell this ball, else NULL. A bowler is NOT
                               credited for 'run out','retired hurt','retired out','obstructing the field'.
  player_out                   the dismissed batter, else NULL

player_lineups — one row per player per match
  match_id, team, player

NOT in this database: who captained a match (there is NO captain column — player_lineups is who played,
not who led), coaches, and player biography/age. It records match events and player run/wicket/six
tallies, not roles or careers. Questions about captaincy, leadership, or a player's career/biography
cannot be answered here — do not fabricate them from lineups or matches played."""

SQL_SYS = f"""Write ONE read-only PostgreSQL SELECT to answer a cricket question over this schema:
{SCHEMA}

Names: players are stored as initial(s)+surname, e.g. 'JM Anderson', 'V Kohli', 'SR Tendulkar' — when
the user names a player by surname or full name, match with ILIKE (e.g. bowler ILIKE '%Anderson'),
never exact '=', since the stored form carries initials.

Teams and events: international team names are stable (India, England), but franchise and event names
vary and get renamed — match those with ILIKE ('Punjab Kings'/'Kings XI Punjab', 'Royal Challengers
Bengaluru'/'Bangalore'). format already scopes the competition (IPL/ODI/Test/T20I), so do NOT also
filter on event_name for a single match — it over-constrains and can return nothing. To locate a
specific match ('the final', 'the match against X'), identify it by the teams and the relevant/latest
date; do NOT add a winner filter unless the question asks who won.

Phrasing: 'X runs off Y balls' is batting notation for one innings — X runs while facing Y balls. The
ball count DESCRIBES the innings; it is NOT a filter on ball_no or over_no. For a batter's runs in a
match, SUM(runs_batter) for that batter in that match (their balls faced, if asked, is COUNT(*)).

Series/tours: identified by (event_name, season), e.g. ('India tour of England','2025'); to answer who
won a series, aggregate that series' matches by the winner column (winner IS NULL = a draw). For the
'last/most recent' series between two teams, take the (event_name, season) with the latest match_date
where both teams appear, then tally winners.
Example — who won the last India-England Test series:
  WITH s AS (SELECT event_name, season FROM cricsheet.matches WHERE format='Test'
    AND team1 IN ('India','England') AND team2 IN ('India','England')
    GROUP BY event_name, season ORDER BY MAX(match_date) DESC LIMIT 1)
  SELECT m.winner, COUNT(*) FROM cricsheet.matches m
    JOIN s ON m.event_name=s.event_name AND m.season=s.season GROUP BY m.winner;
This returns each team's win count plus a NULL-winner row = drawn Tests; do NOT filter out NULL winners.
Return ONLY the SQL — no markdown fences, no prose."""

PHRASE_SYS = ("Answer the cricket question in one precise sentence using the SQL result. "
              "Include the number. For a series result, give the win tally per team and count "
              "any no-winner rows as draws (e.g. '2-2, with one drawn Test'). "
              "The database covers only Tests from Dec 2001, ODIs from 2002, T20Is from 2005, "
              "and IPL from 2008. If the number is a career or aggregate total for a player or "
              "team whose career began BEFORE that window, give it but add that it counts only "
              "matches within the database's window and undercounts their full career. Do NOT add "
              "that caveat for players or seasons that fall entirely inside the window. "
              "If the result is empty, say no data was found.")

# Cricsheet ball-by-ball coverage — anything older is incomplete.
COVERAGE = ("this database covers Tests from Dec 2001, ODIs from 2002, T20Is from "
            "2005 and IPL from 2008, so records or careers predating these are incomplete")
MAX_SQL_TRIES = 3        # generate → execute → feed the error back and retry

# Coverage floors per format. A career total whose earliest DB match hugs the floor is truncated —
# the player's real career predates the window. Data-aware note for the direct-arm/eval path; in the
# serving path the LLM critic makes the coverage call (it reasons about which formats predate 2001+).
FLOORS = {"test": date(2001, 12, 1), "odi": date(2002, 1, 1),
          "t20i": date(2005, 1, 1), "ipl": date(2008, 1, 1)}
_FMT_RE = re.compile(r"format\s*=\s*'(test|odi|t20i|ipl)'", re.I)
_NAME_RE = re.compile(
    r"\b(?:batter|bowler|player|non_striker|player_out)\s+(?:i?like\s*'%?|=\s*')([^%']+)", re.I)


def _clean(sql):
    sql = re.sub(r"```sql|```", "", sql).strip()
    return sql.split(";")[0].strip()          # single statement only


class StatsSQLArm:
    def __init__(self):
        self.sqlgen = agent(SQL_SYS)
        self.phraser = agent(PHRASE_SYS)

    def answer(self, question):
        steps, prompt, sql, rows, err = [], question, "", None, None
        failures = []
        for i in range(MAX_SQL_TRIES):
            sql = _clean(self.sqlgen.run_sync(prompt).output)
            steps.append({"name": f"sqlgen #{i+1}", "output": sql,
                          "input": f"[system]\n{SQL_SYS}\n\n[user]\n{prompt}"})
            low = sql.lower()
            # Read-only: allow SELECT and WITH…SELECT (CTEs, needed for series aggregation);
            # block any write/DDL. _clean already keeps a single statement.
            if not (low.startswith("select") or low.startswith("with")) or re.search(
                    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke)\b", low):
                rows, err = None, "not a read-only SELECT/WITH query"
            else:
                rows, err = self._execute(sql)
            steps.append({"name": f"execute #{i+1}", "input": sql,
                          "output": f"ERROR: {err}" if err else str(rows)})
            if err is None and rows:
                break
            if err is None:                        # valid query but no rows — usually over-constrained
                if i == MAX_SQL_TRIES - 1:
                    break                          # out of tries; accept it as genuinely no data
                fb = ("returned no rows — filters are likely too strict. Match teams/events on a "
                      "distinctive word with ILIKE (e.g. ILIKE '%Punjab%', ILIKE '%Challengers%'), use "
                      "OR between the two sides of a match, and drop filters the question doesn't "
                      "require (winner, exact ball counts).")
            else:
                fb = err
            # feed back EVERY prior failure (errors AND empty results), so a fix isn't reintroduced
            failures.append((sql, fb))
            past = "\n\n".join(f"Attempt {j+1} SQL:\n{s}\nProblem: {e}"
                               for j, (s, e) in enumerate(failures))
            prompt = (f"{question}\n\nEvery attempt below failed or returned nothing. Return ONE "
                      f"corrected read-only query that avoids ALL of these problems — do not "
                      f"reintroduce a fix you have already made.\n\n{past}")
        if err is not None:
            return Answer(f"I could not form a working query for that ({err}).", "stats_sql",
                          tool_trace=["stats_sql", sql], steps=steps)
        phrase_in = f"Question: {question}\nSQL: {sql}\nResult: {rows}"
        out = self.phraser.run_sync(phrase_in).output
        steps.append({"name": "phrase", "output": out,
                      "input": f"[system]\n{PHRASE_SYS}\n\n[user]\n{phrase_in}"})
        # Data-aware truncation note (direct-arm/eval path; the serving critic handles coverage via
        # its own model call). Fires only when the queried player's career hugs the window floor.
        if "undercount" not in out.lower() and self._boundary_hug(sql):
            out += f" (Note: {COVERAGE}.)"
            steps.append({"name": "coverage (boundary-hug)", "input": sql, "output": "caveat appended"})
        return Answer(out, "stats_sql", sources=[{"sql": sql}], steps=steps,
                      tool_trace=["stats_sql", sql, str(rows)],
                      evidence=f"SQL: {sql}\nResult: {rows}")

    def _execute(self, sql):
        """Run a read-only query; return (rows, None) or (None, error_str)."""
        conn = connect()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            return cur.fetchmany(20), None
        except Exception as e:
            return None, str(e)
        finally:
            conn.close()

    def _boundary_hug(self, sql):
        """B (data backstop): does the player's earliest match in the DB hug the format's
        coverage floor? Best-effort — parses entity+format from the generated SQL, returns
        False if it can't. Catches pre-window players the phraser's world-knowledge misses."""
        fmt_m, name_m = _FMT_RE.search(sql), _NAME_RE.search(sql)
        if not (fmt_m and name_m):
            return False
        fmt, name = fmt_m.group(1).lower(), name_m.group(1).strip()
        if len(name) < 3 or fmt not in FLOORS:
            return False
        conn = connect()
        try:
            cur = conn.cursor()
            cur.execute(                                        # parameterized — name is model-derived
                "SELECT MIN(m.match_date) FROM cricsheet.matches m "
                "JOIN cricsheet.player_lineups pl ON m.match_id = pl.match_id "
                "WHERE pl.player ILIKE %s AND lower(m.format) = %s", (f"%{name}%", fmt))
            earliest = cur.fetchone()[0]
        except Exception:
            return False
        finally:
            conn.close()
        # ~90 days: flag only when the earliest match sits in the DB's very first batch for
        # the format (truncation), not a genuine early-in-window debut (avoids false positives).
        return bool(earliest and earliest <= FLOORS[fmt] + timedelta(days=90))
