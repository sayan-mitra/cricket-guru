"""Lightweight local trace store — the substrate for both the frontend Traces
view and the eval metrics (latency, cost, tool path).

Each answer records timed spans (LLM / tool / SQL / judge calls) plus totals, appended to
data/traces/<sid>.jsonl — one file per browser, so the Traces view shows the visitor their own
answers and nobody else's. Without a sid (eval, CLI) it falls back to traces.jsonl, which is where
every historical trace already lives. Self-contained, nothing leaves the box.
An optional Logfire export can sit on top later.
"""
import json
import re
import time
from contextlib import contextmanager

from cricket_guru.config import DATA_DIR

TRACE_DIR = DATA_DIR / "traces"
TRACES = TRACE_DIR / "traces.jsonl"          # the no-sid file: eval, CLI, and the existing history
SID_RE = re.compile(r"\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")


def trace_path(sid=None):
    """Where this session's traces go. The sid reaches us from a query parameter, so it is a
    visitor-controlled string on its way to a filename — anything that isn't a plain UUID is
    dropped rather than sanitised, and lands in the shared file."""
    return TRACE_DIR / f"{sid}.jsonl" if sid and SID_RE.match(sid) else TRACES


class Trace:
    def __init__(self, sid=None):
        self.spans = []
        self.path = trace_path(sid)
        self._t0 = time.perf_counter()

    @contextmanager
    def span(self, name, kind):
        rec = {"name": name, "kind": kind}
        t = time.perf_counter()
        try:
            yield rec
        finally:
            rec["ms"] = round((time.perf_counter() - t) * 1000)
            self.spans.append(rec)

    def elapsed_ms(self):
        return round((time.perf_counter() - self._t0) * 1000)

    def save(self, question, answer, extra=None):
        rec = {"question": question, "answer": answer,
               "total_ms": self.elapsed_ms(), "spans": self.spans, **(extra or {})}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
