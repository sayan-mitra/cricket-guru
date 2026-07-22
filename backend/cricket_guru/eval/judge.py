"""Leg L6 — LLM-as-judge for narrative answers.

same  = the answerer's own model grades (self-preference bias risk)
cross = a different vendor grades (the honest check)
Validated against human labels separately.
"""
from pydantic import BaseModel
from pydantic_ai import Agent

from cricket_guru import config

JSYS = (
    "You grade a candidate answer to a cricket question against a reference "
    "answer. Mark correct=true if the candidate is factually consistent with the "
    "reference and addresses the question, even if worded differently or shorter. "
    "Mark false if it contradicts the reference, misses the point, or says it "
    "doesn't know. Be strict about facts, lenient about phrasing."
)


class Verdict(BaseModel):
    correct: bool
    reason: str


def make_judge(kind="same"):
    model = config.ANSWERER_MODEL if kind == "same" else config.JUDGE_CROSS_MODEL
    agent = Agent(model, system_prompt=JSYS, output_type=Verdict)

    def judge(question, candidate, reference):
        return agent.run_sync(
            f"Question: {question}\n\nReference answer:\n{reference}\n\n"
            f"Candidate answer:\n{candidate}").output

    return judge
