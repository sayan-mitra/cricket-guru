"""Guardrails, four layers.

  input   — relevance (cricket-only) + safety, and a cheap prompt-injection regex
  tool    — SQL SELECT-only (in the stats arm) + agent loop cap (in the router)
  output  — groundedness: is the answer supported by the tool/context evidence

The input gate and output check are one small LLM call each; injection is a regex.
"""
import re

from pydantic import BaseModel
from pydantic_ai import Agent

from cricket_guru import config
from cricket_guru.llm import SETTINGS

INJECTION = re.compile(
    r"ignore (the |your |all )?(previous|above|prior)|disregard (the|your|all)|"
    r"system prompt|reveal your (instructions|prompt)|you are now|override",
    re.I)


class Gate(BaseModel):
    is_cricket: bool
    is_safe: bool


class Grounded(BaseModel):
    grounded: bool
    reason: str


_gate = None
_ground = {}      # cached per model — the critic may use its own CRITIC_MODEL


def _gate_agent():
    global _gate
    if _gate is None:
        _gate = Agent(config.FAST_MODEL, output_type=Gate, model_settings=SETTINGS,
                      system_prompt=(
            "Classify the user message. is_cricket: is it a question about the "
            "sport of cricket? is_safe: is it free of harmful, abusive, or "
            "clearly out-of-scope content?"))
    return _gate


def _ground_agent(model=None):
    model = model or config.FAST_MODEL
    if model not in _ground:
        _ground[model] = Agent(model, output_type=Grounded, model_settings=SETTINGS,
                               system_prompt=(
            "Decide if the candidate answer is trustworthy given the evidence (tool outputs and "
            "retrieved context). grounded=true when the answer's MAIN claim is supported by the "
            "evidence AND nothing in the answer contradicts it. Extra explanatory detail or general "
            "cricket knowledge that goes beyond the evidence is acceptable as long as it does not "
            "contradict the evidence. grounded=false only when the core answer is unsupported by the "
            "evidence, or the answer contradicts it.\n"
            "When the evidence carries CONFLICTING values for the quantity the answer states — two "
            "different run tallies for the same player and series, two different record holders — one "
            "supporting span is not enough, because the other span says the answer is wrong. Rank the "
            "evidence and check the answer took the strongest: a SQL result from the match database "
            "outranks a web snippet, and a snippet attributed to a named source outranks a search "
            "engine's own summary. If the answer states a value that a stronger piece of evidence "
            "contradicts, grounded=false, and name both values in the reason.\n"
            "Read statistical snippets rather than skimming them for a familiar number. Two snippets "
            "giving a player different totals are covering different periods, and at most one of them "
            "answers the question asked — ranking evidence means picking the row whose period matches "
            "the question, not the first plausible figure."))
    return _ground[model]


def check_input(question, followup=False):
    """Returns (ok, reason). reason is a user-facing refusal when not ok.
    followup=True skips the cricket-relevance gate — a mid-conversation follow-up like
    'and the top scorer?' isn't self-evidently cricket, but the thread already is. Injection
    and safety checks still run."""
    if INJECTION.search(question):
        return False, "That looks like an attempt to override my instructions."
    g = _gate_agent().run_sync(question).output
    if not followup and not g.is_cricket:
        return False, "I only answer questions about cricket."
    if not g.is_safe:
        return False, "I can't help with that request."
    return True, None


def check_output(answer_text, evidence, model=None):
    """Returns a Grounded verdict; feeds the hallucination metric.
    model overrides the judge (the serving critic passes CRITIC_MODEL)."""
    if not evidence.strip():
        return Grounded(grounded=False, reason="no evidence gathered")
    return _ground_agent(model).run_sync(
        f"Evidence:\n{evidence}\n\nCandidate answer:\n{answer_text}").output
