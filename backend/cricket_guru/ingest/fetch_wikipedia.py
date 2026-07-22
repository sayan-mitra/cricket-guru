#!/usr/bin/env python3
"""Fetch a cricket prose corpus from English Wikipedia into data/wikipedia/.

Gathers article titles two ways so coverage is robust: a guaranteed core list of
key cricket topics, plus search expansion over topical queries. Then pulls the
plain-text extract for each and saves {title, pageid, url, text} as JSON.
Current Wikipedia snapshot; the staleness rig (aging record articles) is handled
later at eval-design time, not here.
"""
import gzip
import json
import time
import urllib.parse
import urllib.request
from cricket_guru.config import DATA_DIR

API = "https://en.wikipedia.org/w/api.php"
OUT = DATA_DIR / "wikipedia"
CAP = 1500          # max articles
UA = "CricketGuru/0.1 (learning project; contact via project repo)"

# Guaranteed core topics so the corpus always covers rules, formats, majors.
CORE_TITLES = [
    "Cricket", "Laws of Cricket", "History of cricket", "Test cricket",
    "One Day International", "Twenty20 International", "Twenty20",
    "Indian Premier League", "Cricket World Cup", "ICC Men's T20 World Cup",
    "ICC World Test Championship", "Duckworth–Lewis–Stern method",
    "Leg before wicket", "No-ball", "Wide (cricket)", "Powerplay (cricket)",
    "Wicket", "Over (cricket)", "Batting (cricket)", "Bowling (cricket)",
    "Fielding (cricket)", "Dismissal (cricket)", "Cricket field",
    "Result (cricket)", "Follow-on", "Declaration and forfeiture",
    "Ball tampering", "Underarm bowling incident of 1981",
    "2019 Cricket World Cup Final", "2007 Cricket World Cup",
    "2011 Cricket World Cup", "2023 Cricket World Cup", "Bodyline",
    "Sachin Tendulkar", "Virat Kohli", "MS Dhoni", "Rahul Dravid",
    "Ricky Ponting", "Brian Lara", "Muttiah Muralitharan", "Shane Warne",
    "Jacques Kallis", "Kumar Sangakkara", "AB de Villiers", "Chris Gayle",
    "Decision Review System", "Reverse swing", "Doosra", "Yorker",
]

SEARCH_QUERIES = [
    "cricket world cup final", "cricket controversy", "IPL season",
    "cricket rules explained", "famous test match", "cricket batting record",
    "cricket bowling record", "cricket ground stadium", "cricket captain",
    "cricket format history", "cricket umpire decision", "cricket tournament",
]
PER_QUERY = 60


def get(params, _tries=6):
    params = {**params, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip",
                                               "User-Agent": UA})
    delay = 5
    for attempt in range(_tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
            return json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < _tries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if attempt < _tries - 1:  # transient reset/timeout: back off and retry
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise
    raise RuntimeError("unreachable")


def gather_titles():
    titles = list(dict.fromkeys(CORE_TITLES))  # preserve order, dedupe
    seen = set(titles)
    for q in SEARCH_QUERIES:
        try:
            d = get({"action": "query", "list": "search", "srsearch": q,
                     "srnamespace": 0, "srlimit": PER_QUERY})
        except Exception as e:  # keep the core corpus even if search stays throttled
            print(f"  search '{q}' failed ({e}); continuing")
            continue
        for hit in d.get("query", {}).get("search", []):
            t = hit["title"]
            if t not in seen:
                seen.add(t)
                titles.append(t)
        time.sleep(1.0)
        if len(titles) >= CAP:
            break
    return titles[:CAP]


def fetch_batch(batch):
    # The extracts API returns full text for only a few pages per response and
    # pages the rest via a `continue` token; loop until it's exhausted.
    collected, cont = {}, {}
    for _ in range(25):  # continue-loop guard
        params = {"action": "query", "prop": "extracts|info",
                  "explaintext": 1, "exlimit": 20, "inprop": "url",
                  "titles": "|".join(batch)}
        params.update(cont)
        d = get(params)
        for pid, p in d.get("query", {}).get("pages", {}).items():
            if "extract" in p and pid not in collected:
                collected[pid] = p
        if "continue" in d:
            cont = d["continue"]
            time.sleep(0.5)
        else:
            break
    out = []
    for pid, p in collected.items():
        if int(pid) < 0:
            continue
        text = p.get("extract", "").strip()
        if len(text) < 200:  # skip stubs/disambig
            continue
        out.append({
            "title": p["title"],
            "pageid": p["pageid"],
            "url": p.get("fullurl", f"https://en.wikipedia.org/wiki/"
                         f"{urllib.parse.quote(p['title'].replace(' ', '_'))}"),
            "text": text,
        })
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    titles_path = OUT / "titles.json"
    articles_path = OUT / "articles.json"

    # Cache the candidate title list so gathering runs once across restarts.
    if titles_path.exists():
        titles = json.loads(titles_path.read_text())
    else:
        titles = gather_titles()
        titles_path.write_text(json.dumps(titles, indent=2))
    print(f"{len(titles)} candidate titles")

    # Resume: keep whatever we already fetched, skip those titles.
    articles = []
    if articles_path.exists():
        articles = json.loads(articles_path.read_text())
    have = {a["title"] for a in articles}
    todo = [t for t in titles if t not in have]
    print(f"already have {len(articles)}; {len(todo)} to fetch")

    for i in range(0, len(todo), 20):
        batch = todo[i:i + 20]
        articles.extend(fetch_batch(batch))
        articles_path.write_text(json.dumps(articles, indent=2))  # persist per batch
        print(f"  progress: {len(articles)} saved")
        time.sleep(0.8)

    chars = sum(len(a["text"]) for a in articles)
    print(f"done: {len(articles)} articles  ({chars:,} chars of prose)")


if __name__ == "__main__":
    main()
