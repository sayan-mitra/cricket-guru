"""Pydantic AI agent factory. Importing config loads the .env keys.

Runs on whatever CG_ANSWERER_MODEL points at (openai:gpt-5.4-mini for now,
anthropic:claude-sonnet-5 once that key lands) — the arms don't change.
"""
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from cricket_guru import config

# Every agent carries a request deadline (config.LLM_TIMEOUT) and answers one tool at a time.
#
# Serial tools are not a preference, they are load-bearing. The model can return two tool_use blocks
# in one turn, and pydantic-ai runs sync tools on worker threads — but our tools answer by calling
# run_sync themselves, so two at once means two nested event loops and a hang that no request timeout
# can reach. It cost three eval runs before we caught it: 'how many more runs did the leading scorer of
# IPL 2015 make than IPL 2011' hung past 200s, and answered in 23s with this off.
SETTINGS = ModelSettings(timeout=config.LLM_TIMEOUT, parallel_tool_calls=False)


def agent(system_prompt="", model=None, output_type=str, **kw):
    kw.setdefault("model_settings", SETTINGS)
    return Agent(model or config.ANSWERER_MODEL, system_prompt=system_prompt,
                 output_type=output_type, **kw)
