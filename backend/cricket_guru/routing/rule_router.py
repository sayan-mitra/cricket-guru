"""Leg L5 variant A — rule/keyword router.

Dumb heuristic: stat-shaped cues (counts, records, a season/year) -> stats-SQL;
everything else -> text-RAG. Cheap, transparent, no LLM. The baseline the agent
must beat.
"""
import re

from cricket_guru.arms.stats_sql import StatsSQLArm
from cricket_guru.arms.text_rag import TextRAGArm

RULES_CUES = re.compile(
    r"\b(no.?ball|lbw|leg before|free hit|wide|bye|dead ball|dismiss\w*|umpire|drs|"
    r"review|over rate|powerplay|super over|law|rule|legal|illegal|allowed|penalt\w*|"
    r"officiat\w*)\b", re.I)
STAT_CUES = re.compile(
    r"\b(how many|most|fewest|highest|lowest|average|total|number of|score|runs|"
    r"wickets|sixes|fours|strike rate|economy|won|winner|top)\b|\b(19|20)\d{2}\b",
    re.I)


class RuleRouter:
    def __init__(self, retrieval="dense", chunking="fixed", rules_retrieval=None, rerank=False):
        self.stats = StatsSQLArm()
        self.text = TextRAGArm(retrieval, chunking, source="wiki", rerank=rerank)
        self.rules = TextRAGArm(rules_retrieval or retrieval, chunking, source="rules")

    def answer(self, question):
        if RULES_CUES.search(question):      # rules/officiating -> authoritative source
            arm = self.rules
        elif STAT_CUES.search(question):
            arm = self.stats
        else:
            arm = self.text
        a = arm.answer(question)
        a.tool_trace = [f"rule_router->{a.arm}"] + a.tool_trace
        a.arm = "rule_router"
        return a
