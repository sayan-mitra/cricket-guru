"""Serving-path answer critic — a CRAG-style gate (design doc: 'The answer critic').

Grades a finished answer and returns a verdict with a reason:

  ok             — grounded, and its scope sits safely inside the data window. Ship.
  retrieval_gap  — an all-time/career record whose true value could depend on data BEFORE the
                   window (Tests predate 2001, ODIs 2002, …). Verify against the web.
  hallucination  — the evidence doesn't support the answer, and the web wouldn't fix it. Abstain.

A cheap, deterministic pre-check (weak retrieval score) runs first. The coverage + groundedness call
is a single CRITIC_MODEL judgment: instead of a coarse 'is this a superlative?' regex (which flagged
in-window records like the highest India–Australia T20I total and wrongly sent them to the web), the
model reasons about which formats/matchups predate the window and which exist entirely inside it.
"""
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent

from cricket_guru import config
from cricket_guru.arms.base import Answer

OK = "ok"
RETRIEVAL_GAP = "retrieval_gap"
HALLUCINATION = "hallucination"

COVERAGE_WINDOW = ("The stats database covers only: Test matches from Dec 2001, ODIs from 2002, "
                   "T20Is from 2005, IPL from 2008. Anything earlier is absent.")

CRITIC_SYS = f"""You grade a finished cricket answer before it ships. You are given the QUESTION, the
ANSWER, and the EVIDENCE (tool output — SQL result rows with dates, or retrieved passages).

{COVERAGE_WINDOW}

Return exactly one verdict:
- ok: the answer's main claim is supported by the evidence AND its scope sits safely inside the data
  window — ship it. A superlative is safe when the whole period it ranges over exists only within the
  window (for example a T20I record — T20Is began around 2005, so the database holds them all) or when
  it is a specific recent match, season, or series.
- retrieval_gap: the answer is an all-time or career record whose true value could depend on data
  BEFORE the window — for example 'most Test wickets ever': Tests date to 1877 but the database starts
  in 2001, so a pre-window career like Muralitharan's is undercounted and the database's top name may
  be wrong. The answer should be verified against the web.
- hallucination: the evidence does not support the answer's main claim, and a web search would not fix
  it. Abstain.

Reason briefly, judging scope by cricket history: which formats or matchups predate the window, and
which exist entirely inside it."""


@dataclass
class Verdict:
    verdict: str
    reason: str


class _CVerdict(BaseModel):
    verdict: str
    reason: str


_critic = None


def _agent():
    global _critic
    if _critic is None:
        _critic = Agent(config.CRITIC_MODEL, output_type=_CVerdict, system_prompt=CRITIC_SYS)
    return _critic


def critique(question: str, answer: Answer) -> Verdict:
    """Cheap retrieval-gap pre-check (RAG), then one model call for coverage + groundedness."""
    score = answer.retrieval_score
    if score is not None and score < config.CRITIC_THRESHOLD:
        return Verdict(RETRIEVAL_GAP,
                       f"weak retrieval (top similarity {score:.2f} < {config.CRITIC_THRESHOLD})")
    v = _agent().run_sync(
        f"QUESTION: {question}\n\nANSWER: {answer.text}\n\nEVIDENCE:\n{answer.evidence or '(none)'}"
    ).output
    verdict = v.verdict if v.verdict in (OK, RETRIEVAL_GAP, HALLUCINATION) else OK
    return Verdict(verdict, v.reason)
