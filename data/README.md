# Cricket Guru — data manifest

Three downloaded sources, snapshotted 2026-07-13. Raw data is git-ignored; the
fetch/load scripts in `backend/cricket_guru/ingest/` reproduce it. Attribution
fields are carried on every record for CC-BY-SA / ODC compliance. (Rule-book PDFs
are dropped in manually — see `data/rules/README.md`.)

## Cricsheet — structured stats (stats-SQL arm + computed oracle)

Ball-by-ball match data, men's international + IPL, downloaded as per-format JSON zips.

| Format | Files | Source zip |
|---|---|---|
| Test | 886 | `tests_male_json.zip` |
| ODI | 2,552 | `odis_male_json.zip` |
| T20I | 3,461 | `t20s_male_json.zip` |
| IPL | 1,243 | `ipl_male_json.zip` |

Extracted to `data/cricsheet/{tests,odis,t20s,ipl}_male/`. Each match JSON has
`meta`, `info` (teams, dates, venue, players, outcome, season), and `innings`
(deliveries). Source: cricsheet.org. License: attribution required (confirm ODC-BY
on the Cricsheet about page before publishing).

Note: `it20s_male_json.zip` is a different, smaller category (240 matches) — not
men's T20Is. The correct file is `t20s_male_json.zip`.

## Sports Stack Exchange — narrative Q&A (legacy source)

Seeded the early narrative questions. The narrative gold is now corpus-grounded on
the Wikipedia passages, so this set no longer serves as eval ground truth; it stays
documented because the fetch script still reproduces it.

`cricket`-tag Q&A pulled via the Stack Exchange API (no dump/torrent needed at
this volume), frozen to disk.

- `questions.json` — 839 questions (839 with bodies), 459 with an accepted answer.
- `answers.json` — 1,300 answers.

Each row carries `link`, `owner`, `creation_date`, `score`, and (questions)
`accepted_answer_id`. Source: sports.stackexchange.com. License: CC-BY-SA.

## Wikipedia — prose (text-RAG arm)

575 cricket articles (8.26M chars), gathered from a guaranteed core-topics list
plus search expansion, fetched as plain-text extracts.

- `articles.json` — `{title, pageid, url, text}` per article.
- `titles.json` — cached candidate list (lets the fetch resume).

Current snapshot. The staleness rig (aging record articles behind the Cricsheet
cutoff) is applied later at eval-design time, not here. Source: en.wikipedia.org.
License: CC-BY-SA.
