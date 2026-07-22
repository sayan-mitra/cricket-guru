"""Checks that one visitor's chat, traces and answers stay their own.

Plain asserts, no test framework — the repo has none, and adding one means re-freezing
requirements.txt. Run it directly:

    PYTHONPATH=backend python backend/tests/isolation_check.py

Everything here is cheap: no LLM calls, no Postgres. The retriever check loads the index once.
"""
import json
import sys
import tempfile
import threading
from pathlib import Path

from cricket_guru import trace as trace_mod
from cricket_guru.retrieval.base import get_retriever
from cricket_guru.trace import Trace, trace_path

A = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
B = "9c5b94b1-35ad-49bb-b118-8e8fc24abf80"
failures = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        failures.append(name)


print("sid validation — anything that isn't a UUID must not reach a filename")
check("a real uuid gets its own file", trace_path(A).name == f"{A}.jsonl")
for bad in ["../../etc/passwd", "..", "", "a/b", "'; DROP TABLE", "x" * 200, None]:
    check(f"rejected: {bad!r}", trace_path(bad).name == "traces.jsonl")

print("\nno sid falls back to the shared file, so the eval and the existing history keep working")
check("Trace() writes traces.jsonl", Trace().path.name == "traces.jsonl")
check("Trace(sid) writes the visitor's file", Trace(A).path.name == f"{A}.jsonl")

print("\nthe index is shared, so a router per visitor is affordable")
r1 = get_retriever("dense", "structural", "rules")
r2 = get_retriever("dense", "structural", "rules")
check("same config returns the same retriever", r1 is r2)
check("its vectors are one array, not a copy", r1.vecs is r2.vecs)
check("a different config is a different retriever",
      get_retriever("dense", "fixed", "rules") is not r1)

print("\nconcurrent sessions write to their own trace file and don't interleave")
with tempfile.TemporaryDirectory() as d:
    trace_mod.TRACE_DIR = Path(d)
    trace_mod.TRACES = Path(d) / "traces.jsonl"

    def run(sid, tag):
        for i in range(20):
            t = Trace(sid)
            with t.span(f"{tag}-{i}", "tool"):
                pass
            t.save(f"q-{tag}-{i}", f"a-{tag}-{i}")

    threads = [threading.Thread(target=run, args=(A, "alice")),
               threading.Thread(target=run, args=(B, "bob"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for sid, tag, other in [(A, "alice", "bob"), (B, "bob", "alice")]:
        rows = [json.loads(x) for x in (Path(d) / f"{sid}.jsonl").read_text().splitlines() if x.strip()]
        check(f"{tag} wrote all 20 of their own runs", len(rows) == 20)
        check(f"{tag}'s file holds nothing of {other}'s",
              all(tag in r["question"] and other not in r["question"] for r in rows))

print("\nFAILED: " + ", ".join(failures) if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
