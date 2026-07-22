"""Lightweight local trace store — the substrate for both the frontend Traces
view and the eval metrics (latency, cost, tool path).

Each answer records timed spans (LLM / tool / SQL / judge calls) plus totals,
appended to data/traces/traces.jsonl. Self-contained, nothing leaves the box.
An optional Logfire export can sit on top later.
"""
import json
import time
from contextlib import contextmanager

from cricket_guru.config import DATA_DIR

TRACES = DATA_DIR / "traces" / "traces.jsonl"


class Trace:
    def __init__(self):
        self.spans = []
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
        TRACES.parent.mkdir(parents=True, exist_ok=True)
        with open(TRACES, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
