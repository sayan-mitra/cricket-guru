"""Pydantic AI agent factory. Importing config loads the .env keys.

Runs on whatever CG_ANSWERER_MODEL points at (openai:gpt-5.4-mini for now,
anthropic:claude-sonnet-5 once that key lands) — the arms don't change.
"""
from pydantic_ai import Agent

from cricket_guru import config


def agent(system_prompt="", model=None, output_type=str, **kw):
    return Agent(model or config.ANSWERER_MODEL, system_prompt=system_prompt,
                 output_type=output_type, **kw)
