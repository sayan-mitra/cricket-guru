"""Shared answer type. Every arm returns this so the harness treats them alike.

The trace/metrics fields are filled by the agent router (observability) and the
guardrails; the simple arms leave them at defaults.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Answer:
    text: str
    arm: str = ""
    sources: list = field(default_factory=list)     # [{title,url}] or [{sql}]
    tool_trace: list = field(default_factory=list)   # what ran, for the router view
    trace: list = field(default_factory=list)        # timed spans (observability)
    latency_ms: int = 0
    tokens: int = 0
    grounded: Optional[bool] = None                  # output guardrail verdict
    blocked: Optional[str] = None                    # reason if a guardrail blocked
    evidence: str = ""                               # context/tool text the critic grounds against
    retrieval_score: Optional[float] = None          # max chunk similarity; None if no corpus retrieval
    steps: list = field(default_factory=list)        # per-leg {name,input,output} for the detailed trace
