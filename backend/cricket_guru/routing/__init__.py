"""Routing leg (L5). Variants: rule, agent."""


def get_router(kind, retrieval="dense", chunking="fixed", rules_retrieval=None, rerank=False,
               sid=None):
    """rules_retrieval overrides the retrieval strategy for the rules arm only (BM25 hurts
    rulebook lookup — the serving path uses dense for rules, hybrid for wiki). None = same as
    `retrieval`, so eval comparisons are unaffected. rerank cross-encodes the wiki arm's top-N
    (a big recall@1 win on prose; measured to hurt rules, so it's wiki-only). Off by default so
    the ablation experiments still measure base retrieval. sid scopes the agent router's
    trace file to one browser; None writes to the shared traces.jsonl."""
    from cricket_guru.routing.rule_router import RuleRouter
    from cricket_guru.routing.agent_router import AgentRouter
    if kind == "agent":
        return AgentRouter(retrieval, chunking, rules_retrieval=rules_retrieval, rerank=rerank,
                           sid=sid)
    return RuleRouter(retrieval, chunking, rules_retrieval=rules_retrieval, rerank=rerank)
