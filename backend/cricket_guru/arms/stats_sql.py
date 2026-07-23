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
  team1, team2                 the two sides, full names (e.g. 'New Zealand')
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

Names: players are stored as initial(s)+surname, e.g. 'DW Steyn', 'DL Vettori', 'MG Johnson'. Match on
the SURNAME with a leading wildcard — bowler ILIKE '%Steyn' — never exact '='. Never put the given
name inside the wildcards: '%Dale Steyn%' matches nothing, because no stored value carries the given
name, and worse, it can match an unrelated player who happens to be registered under a full name. A
surname alone can cover several players; when you cannot pin the initials, GROUP BY the name column and
return one row per player rather than summing strangers together.

Teams and events: international team names are stable (New Zealand, Sri Lanka), but franchise and event names
vary and get renamed — match those with ILIKE. A franchise can appear under more than one string across
its history: the city gets respelled, a sponsor is added or dropped, the nickname is kept. Both forms
sit in the table as different values, so a filter on the full registered name silently reads half the
club and the query still returns rows. Match on ONE distinctive word, whichever part of the name is
likely to have survived, never the whole name. format already
scopes the competition (IPL/ODI/Test/T20I), so do NOT also
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
Example — who won the last New Zealand-Bangladesh Test series:
  WITH s AS (SELECT event_name, season FROM cricsheet.matches WHERE format='Test'
    AND team1 IN ('New Zealand','Bangladesh') AND team2 IN ('New Zealand','Bangladesh')
    GROUP BY event_name, season ORDER BY MAX(match_date) DESC LIMIT 1)
  SELECT m.winner, COUNT(*) FROM cricsheet.matches m
    JOIN s ON m.event_name=s.event_name AND m.season=s.season GROUP BY m.winner;
This returns each team's win count plus a NULL-winner row = drawn Tests; do NOT filter out NULL winners.

Trophy names are NOT stored. Cricsheet labels most bilateral series as tours, so the 2024/25
Border-Gavaskar Trophy lives under event_name 'India tour of Australia'. Identify a bilateral series by
format + season + the two teams and leave event_name out of it — never filter event_name on a
colloquial trophy name (Border-Gavaskar, The Ashes, Freedom Trophy, Pataudi). event_name is for
multi-team tournaments ('ICC World Cup', 'Indian Premier League').

A split-year season is written with a slash — '2024/25', never '2024-25' or '2024-2025'. It is ONE
season value: filter `season = '2022/23'`, and NEVER expand it into two calendar years — not
`season IN ('2022','2023')`, not `EXTRACT(year FROM match_date) IN (2022,2023)`, not a
Jan-2022–Dec-2023 date range. Expanding it pulls a second season's matches, and in T20Is it lets
associate-nation players accumulate across both years and outrank the real leader. A bare year in a
question ('most wickets in Tests 2017', 'the IPL 2016') also names the season column, not the calendar
year — filter season, not EXTRACT(year FROM match_date). The two disagree: a season labelled '2017'
holds matches either side of the new year, and a tour that starts in December sits in '2024/25'.

Never mask an empty result. Do not wrap an aggregate in COALESCE(...,0) and do not add a fallback row:
when nothing matches, the query must come back empty. A zero conjured by COALESCE reads downstream as a
real answer — 'scored 0 runs' — and it stops the retry that would have fixed the filters.
Return ONLY the SQL — no markdown fences, no prose."""

PHRASE_SYS = ("Answer the cricket question in one precise sentence using the SQL result. "
              "Include the number. For a series result, give the win tally per team and count "
              "any no-winner rows as draws (e.g. '2-2, with one drawn Test'). "
              "A NULL cell means that part of the query matched nothing — say so for that part, and "
              "never report a NULL or a 0 standing in for it as a real figure. "
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

# Columns whose values are free text the model can only guess at. When a query comes back empty we
# look up what the database actually holds for each filtered literal, so the retry corrects the wrong
# value instead of rewording it — three attempts once died in a row on event_name ILIKE '%Border%'
# for a series stored as 'India tour of Australia'.
_VOCAB = {
    "event_name": ("cricsheet.matches", "event_name"),
    "season": ("cricsheet.matches", "season"),
    "venue": ("cricsheet.matches", "venue"),
    "city": ("cricsheet.matches", "city"),
    "team1": ("cricsheet.matches", "team1"),
    "team2": ("cricsheet.matches", "team2"),
    "winner": ("cricsheet.matches", "winner"),
    "batting_team": ("cricsheet.innings", "batting_team"),
    "bowling_team": ("cricsheet.innings", "bowling_team"),
    "batter": ("cricsheet.deliveries", "batter"),
    "bowler": ("cricsheet.deliveries", "bowler"),
    "non_striker": ("cricsheet.deliveries", "non_striker"),
    "player_out": ("cricsheet.deliveries", "player_out"),
    "player": ("cricsheet.player_lineups", "player"),
}
_PEOPLE = {"batter", "bowler", "non_striker", "player_out", "player"}
# Columns where several stored strings can name the same side. Franchises get renamed mid-history and
# both spellings stay in the table, so a filter can match real rows and still miss half the club.
_TEAMISH = {"team1", "team2", "winner", "toss_winner", "batting_team", "bowling_team", "event_name"}
_FILTER_RE = re.compile(r"\b(" + "|".join(_VOCAB) + r")\s*(?:i?like|=)\s*'([^']*)'", re.I)
MAX_PROBES = 6           # filters we look up per empty result — the probe runs on the failure path


def _clean(sql):
    sql = re.sub(r"```sql|```", "", sql).strip()
    return sql.split(";")[0].strip()          # single statement only


def _words(name):
    return {w.lower() for w in re.split(r"[^A-Za-z0-9]+", name) if len(w) >= 4}


def _shares_word(a, b):
    """A rename keeps part of the name — the city, or the nickname. Substring overlap isn't enough:
    'Chennai Super Kings' and 'Rising Pune Supergiant' share the letters of 'Super' and nothing else,
    and their seasons happen not to overlap because Chennai was suspended in the years Pune played."""
    return bool(_words(a) & _words(b))


def _all_null_or_zero(rows, sql):
    """Is this 'no rows' wearing a disguise?

    An aggregate over zero matching rows still comes back as one row — all NULLs, or all zeros once
    COALESCE has been wrapped round it. Either shape reads downstream as a real answer and skips the
    retry that would have fixed the filters, which is how the arm reported 'Rohit Sharma scored 0 runs'
    off a query whose season and player filters both matched nothing.
    """
    if len(rows) != 1:
        return False
    cells = list(rows[0])
    if all(c is None for c in cells):
        return True
    return "coalesce" in sql.lower() and all(c is None or c == 0 for c in cells)


class StatsSQLArm:
    def __init__(self):
        self.sqlgen = agent(SQL_SYS)
        self.phraser = agent(PHRASE_SYS)

    def answer(self, question):
        steps, prompt, sql, rows, err = [], question, "", None, None
        failures, alias_checked = [], False
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
                if err is None and _all_null_or_zero(rows, sql):
                    rows = []                      # a masked empty result, not a genuine zero
            steps.append({"name": f"execute #{i+1}", "input": sql,
                          "output": f"ERROR: {err}" if err else str(rows)})
            if err is None and rows:
                # Rows came back, but a team filter can still be half-blind to a renamed club. Worth
                # one corrective pass; after that take what we have rather than loop on it.
                alias = [] if alias_checked else self._aliases(sql)
                if alias and i < MAX_SQL_TRIES - 1:
                    alias_checked = True
                    steps.append({"name": f"alias probe #{i+1}", "input": sql,
                                  "output": "\n".join(alias)})
                    failures.append((sql, "returned rows, but the filter misses the same side stored "
                                          "under another name:\n" + "\n".join(alias)))
                    prompt = (f"{question}\n\nThe query below ran and returned rows, but its filters are "
                              f"too narrow, so the answer would cover only part of the data. Return ONE "
                              f"corrected read-only query.\n\nSQL:\n{sql}\n" + "\n".join(alias))
                    continue
                break
            if err is None:                        # valid query but no rows — usually over-constrained
                if i == MAX_SQL_TRIES - 1:
                    break                          # out of tries; accept it as genuinely no data
                fb = ("returned no rows — filters are likely too strict. Match teams/events on a "
                      "distinctive word with ILIKE (e.g. ILIKE '%Delhi%', ILIKE '%Capitals%'), use "
                      "OR between the two sides of a match, and drop filters the question doesn't "
                      "require (winner, exact ball counts).")
                probe = self._probe(sql)
                if probe:
                    fb += "\n" + probe
                    steps.append({"name": f"value probe #{i+1}", "input": sql, "output": probe})
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

    def _probe(self, sql):
        """Ground the retry in real column values: for each literal the empty query filtered on, say
        whether the database holds it and what it holds instead. Without this the model rewords the
        same dead filter — it has no way to see that no event_name contains 'Border' for 2024/25."""
        lines, seen = [], set()
        for col, lit in _FILTER_RE.findall(sql):
            col, core = col.lower(), lit.strip("%").strip()
            if len(core) < 3 or (col, core.lower()) in seen or len(seen) >= MAX_PROBES:
                continue
            seen.add((col, core.lower()))
            table, column = _VOCAB[col]
            if not self._values(table, column, core):
                tok, alts = self._nearest(table, column, core)
                lines.append(f"  {col} '{lit}' matches NOTHING stored" + (
                    f" — values containing '{tok}': {', '.join(alts)}" if alts
                    else " — nothing resembles it either; drop this filter"))
            elif col in _PEOPLE and " " in core:
                # the literal matched, but a full name in a surname column is nearly always the wrong
                # person — the stored form is initials+surname, so anything with a given name is a
                # different registration
                surname = core.split()[-1]
                alts = self._values(table, column, surname)
                lines.append(f"  {col} '{lit}' carries a given name — stored players are "
                             f"initials+surname. Matching '{surname}': {', '.join(alts) or 'none'}")
        # An event_name filter is the prime suspect on any empty result, even when the literal exists:
        # 'Border-Gavaskar Trophy' is a real label for the 2008-2017 editions and a dead one for
        # 2024/25, so checking the value in isolation clears a filter that is still wrong here.
        if any(c.lower() == "event_name" for c, _ in _FILTER_RE.findall(sql)):
            hint = self._series_hint(sql)
            if hint:
                lines.append("  the event_name filter is the likeliest culprit — drop it and identify "
                             "the series by teams + format + season instead")
                lines.append(hint)
        return ("Checked against the database, these filter values are the problem:\n"
                + "\n".join(lines)) if lines else None

    def _aliases(self, sql):
        """Catch the query that returns rows and is still wrong.

        A team filter can match real rows while missing the same club under its other name — a city
        respelled, a sponsor added — and nothing looks broken: the numbers are just computed over part
        of the history. So for each team filter, compare what the whole literal matches against what
        each of its own words matches. If a word reaches stored values the filter doesn't, the filter
        is narrower than the side it names. No list of clubs anywhere; the table says which names go
        together.
        """
        lines = []
        for col, lit in _FILTER_RE.findall(sql):
            col, core = col.lower(), lit.strip("%").strip()
            if col not in _TEAMISH or len(core) < 4:
                continue
            table, column = _VOCAB[col]
            matched = set(self._values(table, column, core, limit=25))
            if not matched:
                continue                      # a dead filter is the empty-result probe's job
            for tok in sorted(set(re.split(r"[^A-Za-z0-9]+", core)), key=len, reverse=True):
                if len(tok) < 4 or tok.lower() == core.lower():
                    continue
                missed = {x for x in self._values(table, column, tok, limit=25) if x not in matched
                          and any(_shares_word(m, x) and self._renamed(m, x) for m in matched)}
                if missed:
                    lines.append(
                        f"  {col} '{lit}' matched {sorted(matched)} but '{tok}' also matches "
                        f"{sorted(missed)} — the same side stored under another name. Match on "
                        f"'%{tok}%' instead, or the answer covers only part of the history.")
                    break
        return lines

    def _renamed(self, a, b):
        """Are these two stored names the same club? A club never plays a season as both names, so
        two names that share a word and never appear in the same season are a rename; two that do
        are separate clubs whose names happen to overlap (plenty of Super Kings and Super Giants)."""
        conn = connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM ("
                "  SELECT season FROM cricsheet.matches WHERE team1 = %s OR team2 = %s"
                "  INTERSECT"
                "  SELECT season FROM cricsheet.matches WHERE team1 = %s OR team2 = %s) t",
                (a, a, b, b))
            return cur.fetchone()[0] == 0
        except Exception:
            return False
        finally:
            conn.close()

    def _values(self, table, column, needle, limit=8):
        """Values of a whitelisted column containing needle, commonest first — a surname search should
        surface RG Sharma ahead of the alphabet. Table/column come from _VOCAB, never from the model;
        only the needle reaches the database, parameterized."""
        conn = connect()
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT {column} FROM {table} WHERE {column} ILIKE %s "
                        f"GROUP BY {column} ORDER BY COUNT(*) DESC LIMIT {limit}", (f"%{needle}%",))
            return [str(r[0]) for r in cur.fetchall() if r[0] is not None]
        except Exception:
            return []
        finally:
            conn.close()

    def _nearest(self, table, column, core):
        """Longest word of a dead literal that does match something — '2024-25' has no hits, but
        '2024' finds '2024/25'. Returns (token, values) or ('', [])."""
        for tok in sorted(re.split(r"[^A-Za-z0-9]+", core), key=len, reverse=True):
            if len(tok) > 3:
                alts = self._values(table, column, tok)
                if alts:
                    return tok, alts
        return "", []

    def _series_hint(self, sql):
        """A dead event_name filter usually means the series is stored under its tour name, so list
        the series that do exist for the teams in the query."""
        teams = [t.strip("%").strip() for c, t in _FILTER_RE.findall(sql)
                 if c.lower() in ("team1", "team2", "batting_team", "bowling_team", "winner")]
        teams = list(dict.fromkeys(t for t in teams if len(t) > 2))[:2]
        fmt = _FMT_RE.search(sql)
        if not (teams and fmt):
            return None
        where = " AND ".join(["lower(format) = %s"]
                             + ["(team1 ILIKE %s OR team2 ILIKE %s)"] * len(teams))
        params = [fmt.group(1).lower()] + [p for t in teams for p in (f"%{t}%", f"%{t}%")]
        conn = connect()
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT DISTINCT event_name, season FROM cricsheet.matches WHERE {where} "
                        f"ORDER BY season DESC LIMIT 10", params)
            rows = cur.fetchall()
        except Exception:
            return None
        finally:
            conn.close()
        return ("  series actually stored for " + " vs ".join(teams) + ": "
                + "; ".join(f"{e} ({s})" for e, s in rows)) if rows else None

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
