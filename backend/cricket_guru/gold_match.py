"""Semantic gold lookup for the comparison view's gold lane.

Embeds the gold questions once (bge-small, cached) and cosine-matches a query.
Returns the curated reference when the match clears a threshold, else None so the
lane greys out as 'no gold hit' — present but honest.
"""
import json
from functools import lru_cache

import numpy as np

from cricket_guru.config import DATA_DIR
from cricket_guru.index.embed import embed_passages, embed_query

GOLD_FILES = ["stats_gold.json", "rules_gold.json", "narrative_gold.json"]
MATCH_THRESHOLD = 0.75


@lru_cache(maxsize=1)
def _gold():
    items = []
    for f in GOLD_FILES:
        p = DATA_DIR / "gold" / f
        if p.exists():
            items.extend(json.loads(p.read_text()))
    vecs = np.array(embed_passages([g["question"] for g in items]), dtype=np.float32)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    return items, vecs


def match(question, threshold=MATCH_THRESHOLD):
    items, vecs = _gold()
    if not items:
        return None
    q = np.array(embed_query(question), dtype=np.float32)
    q /= (np.linalg.norm(q) + 1e-9)
    with np.errstate(all="ignore"):     # macOS Accelerate BLAS raises spurious fp flags on matmul
        sims = vecs @ q
    i = int(np.argmax(sims))
    if sims[i] < threshold:
        return None
    g = items[i]
    return {"id": g["id"], "question": g["question"], "qtype": g.get("qtype"),
            "reference": g.get("reference") or g.get("answer") or "", "score": float(sims[i])}
