"""Leg L5 variant B — LLM tool-calling agent (the ReAct loop), with
observability (timed spans) and guardrails wired in.

Grounding policy: answer from the tools, fall back to web-search when they're
insufficient, never fabricate, say which source each part came from.
Guardrails: input relevance+safety+injection, a loop cap, and an output
groundedness check. Every step is traced for the frontend and the metrics.
"""
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from cricket_guru import config, guardrails
from cricket_guru.arms.base import Answer
from cricket_guru.arms.stats_sql import StatsSQLArm
from cricket_guru.arms.text_rag import TextRAGArm
from cricket_guru.tools.web_search import web_search_frozen
from cricket_guru.trace import Trace

try:
    from pydantic_ai.exceptions import UsageLimitExceeded
except Exception:  # keep working across pydantic-ai versions
    UsageLimitExceeded = Exception

LOOP_CAP = 6   # max model requests per question (tool guardrail)

SYS = (
    "You are Cricket Guru. Answer cricket questions by calling the right tool:\n"
    "- cricket_stats for numbers COMPUTABLE from the match database — a specific season/match/format "
    "tally WITHIN coverage (Tests 2001+, ODIs 2002+, T20Is 2005+, IPL 2008+), e.g. 'most runs in IPL "
    "2016', 'who won the last India-England series'. NOT for all-time records, tournament history "
    "across years, historical firsts, events before the window, OR captaincy/leadership records and "
    "player careers (the database has no captain data — it knows who played, not who led).\n"
    "- cricket_rules for laws/officiating (no-ball, LBW, DRS, free hit, dismissals, playing conditions)\n"
    "- cricket_prose for history, records, and narrative — why a match mattered, controversies, player "
    "style, tournament history, captaincy and leadership records, player careers, AND all-time/"
    "historical facts EVEN WHEN the answer is a number ('how many times has England hosted the World "
    "Cup', 'most wickets in a single World Cup', \"Kohli's Test captaincy record vs Dhoni's\").\n"
    "- web_search only when the tools are insufficient or the answer may be out of date\n"
    "Never invent facts. Prefer cricket_rules over cricket_prose for anything about how the game "
    "is officiated. If a tool returns nothing useful, try web_search rather than guessing. "
    "Say which source your answer came from. When you rely on web_search, START the answer with a "
    "one-line caution that your curated cricket corpus didn't cover this and you're drawing on a live web search.\n"
    "If cricket_stats notes its data coverage is incomplete for a historical/all-time record, give the "
    "in-database answer first, then WITHOUT asking the user immediately call web_search for the true "
    "all-time record and append it in a SEPARATE section headed 'Historical (web, unverified)' with a "
    "caution — keep the verified database answer distinct from the web facts. Do not offer to check; just check."
)


class AgentRouter:
    def __init__(self, retrieval="dense", chunking="fixed", guard=True, rules_retrieval=None,
                 rerank=False):
        self.guard = guard
        self._t = None
        self._evidence = []
        self._scores = []      # retrieval similarity from each corpus tool call
        stats = StatsSQLArm()
        text = TextRAGArm(retrieval, chunking, source="wiki", rerank=rerank)   # rerank lifts wiki @1
        rules = TextRAGArm(rules_retrieval or retrieval, chunking, source="rules")  # dense, no rerank
        text.retriever      # warm the in-memory indexes in this (main) thread so
        rules.retriever     # tool worker threads never touch the SQLite-backed store
        agent = Agent(config.ANSWERER_MODEL, system_prompt=SYS)

        @agent.tool_plain
        def cricket_stats(question: str) -> str:
            """Exact cricket facts COMPUTABLE from the structured match database (Cricsheet): match
            and series results, winners, margins and scores, plus player aggregates (runs, wickets,
            sixes) — queryable by team, season, format, or date, WITHIN coverage (Tests from Dec
            2001, ODIs 2002, T20Is 2005, IPL 2008). Use for a specific in-window tally like 'most
            runs in IPL 2016' or 'who won the last India-England series'. Do NOT use for all-time
            records, tournament history, historical firsts, events before the window, or captaincy/
            leadership and player careers (no captain data exists) — those go to cricket_prose."""
            with self._t.span("cricket_stats", "tool") as s:
                r = stats.answer(question)
                out = r.text
                s["input"], s["output"], s["steps"], s["parent"] = question, out, r.steps, "agent_run"
            self._evidence.append(out)
            return out

        @agent.tool_plain
        def cricket_prose(question: str) -> str:
            """History, records, and narrative from the encyclopedia: why a match mattered,
            controversies, player style, tournament history across years, captaincy and leadership
            records, player careers, all-time records and historical firsts — EVEN WHEN the answer is
            a number (how many times England has hosted the World Cup; Kohli's Test captaincy record;
            a batsman's score in a 1979 final)."""
            with self._t.span("cricket_prose", "tool") as s:
                r = text.answer(question)
                out = r.text
                s["input"], s["output"], s["steps"], s["parent"] = question, out, r.steps, "agent_run"
            self._evidence.append(out)
            self._scores.append(r.retrieval_score)
            return out

        @agent.tool_plain
        def cricket_rules(question: str) -> str:
            """Authoritative answer from the official rule books (MCC Laws + ICC/IPL playing conditions): laws, officiating, DRS, no-ball, LBW, dismissals, playing conditions."""
            with self._t.span("cricket_rules", "tool") as s:
                r = rules.answer(question)
                out = r.text
                s["input"], s["output"], s["steps"], s["parent"] = question, out, r.steps, "agent_run"
            self._evidence.append(out)
            self._scores.append(r.retrieval_score)
            return out

        @agent.tool_plain
        def web_search(query: str) -> str:
            """Check current cricket facts or records on the web when the corpus is stale or insufficient."""
            with self._t.span("web_search", "tool") as s:
                out = web_search_frozen(query)
                s["input"], s["output"], s["parent"] = query, out, "agent_run"
            self._evidence.append(out)
            return out

        self.agent = agent

    def _blocked(self, question, reason):
        a = Answer(reason, "agent", blocked=reason, trace=self._t.spans,
                   latency_ms=self._t.elapsed_ms())
        self._t.save(question, reason, {"blocked": reason})
        return a

    def answer(self, question, history=None):
        self._t = Trace()
        self._evidence = []
        self._scores = []

        if self.guard:
            with self._t.span("input_guard", "guard") as s:
                ok, reason = guardrails.check_input(question, followup=bool(history))
                s["input"] = question
                s["output"] = "passed" if ok else f"blocked: {reason}"
            if not ok:
                return self._blocked(question, reason)

        asked = question
        if history:               # compact Q&A context (not tool transcripts) so references resolve
            convo = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
            asked = (f"Earlier in this conversation:\n{convo}\n\nNow answer this, resolving any "
                     f"references (it / that / the same …) to the above:\n{question}")

        tokens = 0
        with self._t.span("agent_run", "llm") as s:
            try:
                result = self.agent.run_sync(
                    asked, usage_limits=UsageLimits(request_limit=LOOP_CAP))
                out = result.output
                tokens = getattr(result.usage(), "total_tokens", 0) or 0
            except UsageLimitExceeded:
                return self._blocked(question, "Stopped: hit the tool-call limit.")
            s["tokens"] = tokens
            s["input"] = f"[system]\n{SYS}\n\n[user]\n{asked}"
            s["output"] = out

        grounded = None
        if self.guard:
            with self._t.span("output_guard", "guard") as s:
                evidence = "\n".join(self._evidence)
                v = guardrails.check_output(out, evidence)
                grounded, s["reason"] = v.grounded, v.reason
                s["input"] = f"[evidence]\n{evidence}\n\n[candidate]\n{out}"
                s["output"] = f"grounded={v.grounded}: {v.reason}"

        # Belt-and-suspenders: if web was used and the model forgot the caution, prepend it.
        if any(sp["name"] == "web_search" for sp in self._t.spans) and "web" not in out.lower():
            out = ("Note: my curated cricket corpus didn't have enough on this, so I drew on "
                   "a live web search — treat it with that caveat.\n\n") + out

        tools = [sp["name"] for sp in self._t.spans if sp["kind"] == "tool"]
        scores = [s for s in self._scores if s is not None]
        a = Answer(out, "agent", tool_trace=["agent_router"] + tools,
                   trace=self._t.spans, latency_ms=self._t.elapsed_ms(),
                   tokens=tokens, grounded=grounded,
                   evidence="\n".join(self._evidence),
                   retrieval_score=max(scores) if scores else None)
        self._t.save(question, out, {"tokens": tokens, "grounded": grounded})
        return a
