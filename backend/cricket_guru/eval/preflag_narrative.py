#!/usr/bin/env python3
"""Pre-flag each narrative gold item to speed human curation.

For every question/reference pair, suggest keep or drop with a short reason.
Suggestion only — the human's keep/drop in the Label tab is the label of record.

    python -m cricket_guru.eval.preflag_narrative
"""
import json

from pydantic import BaseModel
from pydantic_ai import Agent

from cricket_guru.config import ANSWERER_MODEL, DATA_DIR

SYS = (
    "Assess a cricket Q&A pair for use as an evaluation item. "
    "keep=true if the question is a clear, answerable cricket question AND the "
    "reference answer is substantive and actually addresses it. "
    "keep=false if the question is vague, opinion-based, or broken, or if the "
    "reference is thin, off-topic, or doesn't answer the question. Give one short reason."
)


class Flag(BaseModel):
    keep: bool
    reason: str


def main():
    gold = json.loads((DATA_DIR / "gold" / "narrative_gold.json").read_text())
    agent = Agent(ANSWERER_MODEL, output_type=Flag, system_prompt=SYS)
    flags = {}
    for g in gold:
        f = agent.run_sync(
            f"Question: {g['question']}\n\nReference answer:\n{g['reference'][:1500]}").output
        flags[g["id"]] = {"keep": f.keep, "reason": f.reason}
        print(("keep" if f.keep else "DROP"), g["id"], "-", f.reason[:70])

    (DATA_DIR / "gold" / "narrative_preflags.json").write_text(json.dumps(flags, indent=2))
    print(f"\nwrote pre-flags: {sum(v['keep'] for v in flags.values())}/{len(flags)} suggested keep")


if __name__ == "__main__":
    main()
