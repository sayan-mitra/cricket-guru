-- Cricket Guru schema. Two schemas: cricsheet (structured stats + computed
-- oracle) and sports_se (narrative eval oracle). Drop-and-recreate so re-runs
-- are clean.

DROP SCHEMA IF EXISTS cricsheet CASCADE;
DROP SCHEMA IF EXISTS sports_se CASCADE;
CREATE SCHEMA cricsheet;
CREATE SCHEMA sports_se;

-- ---------------------------------------------------------------------------
-- cricsheet
-- ---------------------------------------------------------------------------
CREATE TABLE cricsheet.matches (
    match_id        BIGINT PRIMARY KEY,
    format          TEXT,        -- Test | ODI | T20I | IPL
    match_type      TEXT,        -- raw info.match_type
    gender          TEXT,
    season          TEXT,
    event_name      TEXT,
    match_date      DATE,        -- first date of the match
    venue           TEXT,
    city            TEXT,
    team1           TEXT,
    team2           TEXT,
    toss_winner     TEXT,
    toss_decision   TEXT,        -- bat | field
    winner          TEXT,        -- null on tie/no-result
    result          TEXT,        -- normal | tie | no result | draw
    win_by_runs     INT,
    win_by_wickets  INT,
    player_of_match TEXT,
    balls_per_over  INT
);

CREATE TABLE cricsheet.innings (
    match_id      BIGINT REFERENCES cricsheet.matches(match_id),
    innings_no    INT,
    batting_team  TEXT,
    bowling_team  TEXT,
    runs          INT,           -- total runs in the innings
    wickets       INT,           -- wickets lost
    PRIMARY KEY (match_id, innings_no)
);

CREATE TABLE cricsheet.deliveries (
    id            BIGSERIAL PRIMARY KEY,
    match_id      BIGINT,
    innings_no    INT,
    over_no       INT,           -- 0-indexed over as in Cricsheet
    ball_no       INT,           -- ball within the over
    batting_team  TEXT,
    batter        TEXT,
    bowler        TEXT,
    non_striker   TEXT,
    runs_batter   INT,
    runs_extras   INT,
    runs_total    INT,
    extra_type    TEXT,          -- wides | noballs | byes | legbyes | null
    wicket_kind   TEXT,          -- bowled | caught | lbw | ... | null
    player_out    TEXT           -- null when no wicket
);

CREATE TABLE cricsheet.player_lineups (
    match_id      BIGINT,
    team          TEXT,
    player        TEXT
);

CREATE INDEX idx_deliveries_match   ON cricsheet.deliveries(match_id, innings_no);
CREATE INDEX idx_deliveries_batter  ON cricsheet.deliveries(batter);
CREATE INDEX idx_deliveries_bowler  ON cricsheet.deliveries(bowler);
CREATE INDEX idx_matches_format     ON cricsheet.matches(format);
CREATE INDEX idx_matches_season     ON cricsheet.matches(season);
CREATE INDEX idx_lineups_match      ON cricsheet.player_lineups(match_id);

-- ---------------------------------------------------------------------------
-- sports_se  (narrative eval oracle; never embedded into Qdrant)
-- ---------------------------------------------------------------------------
CREATE TABLE sports_se.questions (
    question_id        BIGINT PRIMARY KEY,
    title              TEXT,
    body               TEXT,        -- cleaned text
    body_html          TEXT,        -- raw HTML
    score              INT,
    view_count         INT,
    tags               TEXT,        -- comma-joined
    accepted_answer_id BIGINT,
    owner_user_id      BIGINT,
    owner_name         TEXT,
    creation_date      TIMESTAMP,
    link               TEXT
);

CREATE TABLE sports_se.answers (
    answer_id     BIGINT PRIMARY KEY,
    question_id   BIGINT REFERENCES sports_se.questions(question_id),
    body          TEXT,
    body_html     TEXT,
    score         INT,
    is_accepted   BOOLEAN DEFAULT FALSE,
    owner_user_id BIGINT,
    owner_name    TEXT,
    creation_date TIMESTAMP,
    link          TEXT
);

CREATE INDEX idx_answers_question ON sports_se.answers(question_id);
