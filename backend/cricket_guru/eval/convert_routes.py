#!/usr/bin/env python3
"""Convert the wiki-routed narrative items into a clean narrative/wiki gold.

Keep the question, replace the weak Sports SE reference with a fresh concise
draft (to be verified in the app — same 'I prep, you decide' pattern). Rewrites
narrative_gold.json to just the wiki-kept set. Rules items already moved to
rules_gold; stats items handled separately; drops removed.

    python -m cricket_guru.eval.convert_routes
"""
import json

from pydantic import BaseModel
from pydantic_ai import Agent

from cricket_guru.config import ANSWERER_MODEL, DATA_DIR

SYS = (
    "Write a concise, factual reference answer (1-3 sentences) to this cricket "
    "question, suitable as an answer key for grading. State the correct answer "
    "plainly, no hedging. If it's a definition/terminology question, define it precisely."
)


class Ref(BaseModel):
    reference: str


def main():
    gd = DATA_DIR / "gold"
    gold = {g["id"]: g for g in json.loads((gd / "narrative_gold.json").read_text())}
    routes = json.loads((gd / "narrative_routes.json").read_text())
    wiki_ids = [i for i, r in routes.items() if r["source"] == "wiki"]

    agent = Agent(ANSWERER_MODEL, output_type=Ref, system_prompt=SYS)
    out = []
    for gid in wiki_ids:
        g = gold[gid]
        ref = agent.run_sync(f"Question: {g['question']}").output.reference
        out.append({
            "id": gid, "qtype": "narrative", "staleness": False,
            "question": g["question"], "reference": ref,
            "tags": g.get("tags", ""), "verified": False,
        })
        print(gid, "->", ref[:75])

    (gd / "narrative_gold.json").write_text(json.dumps(out, indent=2))
    print(f"\nrewrote narrative_gold.json with {len(out)} clean wiki items (verify in app)")


if __name__ == "__main__":
    main()
