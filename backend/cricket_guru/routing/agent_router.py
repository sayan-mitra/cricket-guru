"""Leg L5 variant B — LLM tool-calling agent (the ReAct loop), with
observability (timed spans) and guardrails wired in.

Grounding policy: answer from the tools, fall back to web-search when they're
insufficient, never fabricate, say which source each part came from.
Guardrails: input relevance+safety+injection, a loop cap, and an output
groundedness check. Every step is traced for the frontend and the metrics.
"""
from datetime import date

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from cricket_guru import config, guardrails
from cricket_guru.arms.base import Answer
from cricket_guru.arms.stats_sql import StatsSQLArm
from cricket_guru.arms.text_rag import TextRAGArm
from cricket_guru.llm import SETTINGS, agent
from cricket_guru.tools.web_search import sources_only, web_search_frozen
from cricket_guru.trace import Trace

try:
    from pydantic_ai.exceptions import UsageLimitExceeded
except Exception:  # keep working across pydantic-ai versions
    UsageLimitExceeded = Exception

# Max model requests per question (tool guardrail). A three-part question — a total, a margin, and a
# player's score — spends one call per part before it can compose, and retries come out of the same
# budget, so 6 cut off questions that were answering fine.
LOOP_CAP = 10

SALVAGE_SYS = (
    "You are Cricket Guru. The tool budget ran out before the agent finished, so answer the question "
    "from the tool results below and nothing else. Give every part you can support, and say plainly "
    "which parts you could not determine. Never fill a gap with your own knowledge."
)

SYS = (
    "You are Cricket Guru. Answer cricket questions by calling the right tool:\n"
    "- cricket_stats for numbers COMPUTABLE from the match database — a specific season/match/format "
    "tally WITHIN coverage (Tests 2001+, ODIs 2002+, T20Is 2005+, IPL 2008+), e.g. 'how many Test "
    "wickets did DL Vettori take in 2010', 'who won the last New Zealand-Bangladesh Test series'. NOT "
    "for all-time records, tournament history across years, historical firsts, events before the "
    "window, OR captaincy/leadership records and player careers (the database has no captain data — it "
    "knows who played, not who led).\n"
    "- cricket_rules for laws/officiating (no-ball, LBW, DRS, free hit, dismissals, playing conditions)\n"
    "- cricket_prose for history, records, and narrative — why a match mattered, controversies, player "
    "style, tournament history, captaincy and leadership records, player careers, AND all-time/"
    "historical facts EVEN WHEN the answer is a number ('how many World Cup finals has New Zealand "
    "reached', 'which captain has the best Test win rate', 'a bowler's career strike rate across "
    "formats').\n"
    "- web_search only when the tools are insufficient or the answer may be out of date\n"
    "A tally for one named series, season, or match that sits inside coverage is cricket_stats work — "
    "call it FIRST, before prose or the web, even when the series is known by a trophy name.\n"
    "Where cricket_stats and web_search disagree on a number, the database wins: give its figure, and "
    "say the web disagreed rather than quietly picking one. A web result carries a summary line and "
    "the source snippets it was built from — read the snippets, and never repeat a summary figure the "
    "snippets contradict.\n"
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
                 rerank=False, sid=None):
        self.guard = guard
        # One router per browser session, so this instance's per-call state is nobody else's. The
        # arms it builds are cheap; the index behind them is cached in get_retriever and shared.
        self.sid = sid
        self._t = None
        self._evidence = []
        self._scores = []      # retrieval similarity from each corpus tool call
        stats = StatsSQLArm()
        text = TextRAGArm(retrieval, chunking, source="wiki", rerank=rerank)   # rerank lifts wiki @1
        rules = TextRAGArm(rules_retrieval or retrieval, chunking, source="rules")  # dense, no rerank
        text.retriever      # warm the in-memory indexes in this (main) thread so
        rules.retriever     # tool worker threads never touch the SQLite-backed store
        agent = Agent(config.ANSWERER_MODEL, model_settings=SETTINGS,
                      system_prompt=f"Today's date is {date.today().isoformat()}.\n\n{SYS}")

        @agent.tool_plain
        def cricket_stats(question: str) -> str:
            """Exact cricket facts COMPUTABLE from the structured match database (Cricsheet): match
            and series results, winners, margins and scores, plus player aggregates (runs, wickets,
            sixes) — queryable by team, season, format, or date, WITHIN coverage (Tests from Dec
            2001, ODIs 2002, T20Is 2005, IPL 2008). Use for a specific in-window tally like 'how many
            Test wickets did DL Vettori take in 2010' or 'who won the last New Zealand-Bangladesh
            Test series'. Do NOT use for all-time records, tournament history, historical firsts,
            events before the window, or captaincy/leadership and player careers (no captain data
            exists) — those go to cricket_prose."""
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
            a number (how many World Cup finals New Zealand has reached; which captain has the best
            Test win rate; a bowler's career strike rate across formats)."""
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
            # only the sourced snippets become evidence — grounding against the engine's own
            # summary is circular, and that summary is where the wrong numbers come from
            self._evidence.append(sources_only(out))
            return out

        self.agent = agent

    def _blocked(self, question, reason):
        a = Answer(reason, "agent", blocked=reason, trace=self._t.spans,
                   latency_ms=self._t.elapsed_ms())
        self._t.save(question, reason, {"blocked": reason})
        return a

    def answer(self, question, history=None):
        self._t = Trace(self.sid)
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
                # Don't throw away what the tools already returned. Running out of budget on the third
                # part of a three-part question used to discard correct answers to the first two and
                # reply "Stopped: hit the tool-call limit."
                if not self._evidence:
                    return self._blocked(question, "Stopped: hit the tool-call limit.")
                out = agent(SALVAGE_SYS).run_sync(
                    f"Question: {question}\n\nTool results:\n" + "\n\n".join(self._evidence)).output
                s["output"] = out
                s["input"] = f"[system]\n{SALVAGE_SYS}\n\n[user]\n{question}"
                self._t.spans.append({"kind": "action", "name": "salvage", "ms": 0,
                                      "input": "tool budget exhausted",
                                      "output": "answered from the evidence already gathered"})
            s["tokens"] = tokens
            s["input"] = f"[system]\n{SYS}\n\n[user]\n{asked}"
            s["output"] = out

        grounded, grounded_reason = None, ""
        if self.guard:
            with self._t.span("output_guard", "guard") as s:
                evidence = "\n".join(self._evidence)
                v = guardrails.check_output(out, evidence)
                grounded, grounded_reason = v.grounded, v.reason
                s["reason"] = v.reason
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
                   tokens=tokens, grounded=grounded, grounded_reason=grounded_reason,
                   evidence="\n".join(self._evidence),
                   retrieval_score=max(scores) if scores else None)
        self._t.save(question, out, {"tokens": tokens, "grounded": grounded})
        return a
