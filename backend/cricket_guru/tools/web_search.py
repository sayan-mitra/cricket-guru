"""Freshness tool for the agent — reach the web when the corpus is insufficient
or possibly out of date.

Backends, tried in order of availability:
  Tavily  (TAVILY_API_KEY) — LLM-oriented search API, robust under load
  Brave   (BRAVE_API_KEY)  — keyed web search API
  DuckDuckGo scraping      — no key; works, but DDG throttles rapid sequential calls
                             (we hit the lite/html endpoints over httpx directly; the
                             `ddgs` library's primp backend fails TLS here — "0x304")

A frozen SNAPSHOT can pin specific eval answers ahead of any backend. The agent is
told to flag in its answer that a fact came from the web, not the curated corpus.
"""
from __future__ import annotations   # str | None hint on Python 3.9

from urllib.parse import urlparse

import httpx
from lxml import html as LH

from cricket_guru import config

# Curated overrides for reproducible eval: {query_substring: fact}. Empty = always live.
SNAPSHOT = {}

MARKER = "[WEB]"   # the answer-side caution is driven by this + the system prompt
SUMMARY_TAG = "summary (engine-written, unverified — check it against the sources):"
SOURCES_TAG = "sources:"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")


# Prefer recognised cricket sources where the API supports domain filtering, and cite the sources so
# the answer is attributable — an unnamed bad source is how the wrong '272' fact slipped through.
AUTHORITATIVE = ["espncricinfo.com", "icc-cricket.com", "cricbuzz.com", "en.wikipedia.org"]


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _snippets(items, text_key: str, url_key: str) -> str | None:
    """Raw top snippets, each tagged with its source domain, for the agent to synthesise and cite."""
    out = []
    for x in items:
        c = " ".join((x.get(text_key) or "").split())
        if len(c) > 20:
            dom = _domain(x.get(url_key, ""))
            out.append(f"[{dom}] {c}" if dom else c)
    return "  ".join(out[:5])[:1600] if out else None


def _pack(snippets) -> str | None:
    snippets = [s.strip() for s in snippets if s and len(s.strip()) > 20]
    return f"{MARKER} " + " ".join(snippets[:5])[:1500] if snippets else None


def _tavily_call(query: str, domains) -> str | None:
    # Return BOTH the one-line summary and the raw sourced snippets: the summary is a hint the agent
    # can collapse a record wrongly, so the snippets are the primary material to verify/correct it.
    body = {"api_key": config.TAVILY_API_KEY, "query": query,
            "max_results": 6, "include_answer": True}
    if domains:
        body["include_domains"] = domains
    r = httpx.post("https://api.tavily.com/search", timeout=15, json=body)
    r.raise_for_status()
    d = r.json()
    ans = (d.get("answer") or "").strip()
    snips = _snippets(d.get("results", []), "content", "url")
    if not (ans or snips):
        return None
    parts = []
    if ans:
        parts.append(f"{SUMMARY_TAG} {ans}")
    if snips:
        parts.append(f"{SOURCES_TAG} {snips}")
    return f"{MARKER} " + "  ".join(parts)


def sources_only(text: str) -> str:
    """The evidence half of a web result — the sourced snippets, without the engine's summary.

    Grounding an answer against that summary is circular: the summary is a claim derived from the
    same snippets, and Tavily collapsed two Border-Gavaskar editions into one wrong figure (242 for
    Rohit Sharma, against the 31 sitting in its own ESPNcricinfo row) that the guard then waved
    through. Backends that return snippets only pass through unchanged.
    """
    i = text.find(SOURCES_TAG)
    return f"{MARKER} {text[i:]}" if i != -1 else text


def _tavily(query: str) -> str | None:
    # authoritative cricket sources first, then fall back to the open web
    return _tavily_call(query, AUTHORITATIVE) or _tavily_call(query, None)


def _brave(query: str) -> str | None:
    r = httpx.get("https://api.search.brave.com/res/v1/web/search", timeout=15,
                  params={"q": query, "count": 5},
                  headers={"X-Subscription-Token": config.BRAVE_API_KEY,
                           "Accept": "application/json"})
    r.raise_for_status()
    snips = _snippets(r.json().get("web", {}).get("results", []), "description", "url")
    return f"{MARKER} {snips}" if snips else None


# DDG scraper endpoints, tried in order. Snippet cells differ, so each carries its xpath.
_DDG = [
    ("https://lite.duckduckgo.com/lite/", '//td[contains(@class,"result-snippet")]'),
    ("https://html.duckduckgo.com/html/", '//*[contains(@class,"result__snippet")]'),
]


def _duckduckgo(query: str) -> str | None:
    for url, xp in _DDG:
        try:
            r = httpx.get(url, params={"q": query}, headers={"User-Agent": UA},
                          timeout=12, follow_redirects=True)
            if r.status_code != 200:      # 202 = DDG throttle; try the next endpoint
                continue
            out = _pack(e.text_content() for e in LH.fromstring(r.text).xpath(xp))
            if out:
                return out
        except Exception:
            continue
    return None


def _backends():
    """Available backends, best-first."""
    bs = []
    if config.TAVILY_API_KEY:
        bs.append(_tavily)
    if config.BRAVE_API_KEY:
        bs.append(_brave)
    bs.append(_duckduckgo)                # always available, no key
    return bs


def web_search_frozen(query: str) -> str:
    q = query.lower()
    for key, fact in SNAPSHOT.items():
        if key in q:
            return f"{MARKER} {fact}"
    for backend in _backends():
        try:
            out = backend(query)
            if out:
                return out
        except Exception:
            continue                      # bad key / rate limit — fall through
    return "No results found on the web."
