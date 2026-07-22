"""Leg L3 variant B — hybrid retrieval: dense (in-memory vectors) + BM25 keyword,
fused with Reciprocal Rank Fusion.

Catches both meaning (dense) and exact terms like rare names, Law numbers, or
specific phrasing (BM25) that pure vectors drift away from. All in memory, so
it's thread-safe under the agent's worker threads.
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from cricket_guru.config import collection
from cricket_guru.index.embed import embed_query
from cricket_guru.retrieval.base import Hit
from cricket_guru.retrieval.dense import load_collection

_TOKEN = re.compile(r"[a-z0-9]+")
RRF_K = 60


def _tok(s):
    return _TOKEN.findall(s.lower())


class HybridRetriever:
    def __init__(self, chunking="fixed", source="wiki"):
        self.payloads, self.vecs = load_collection(collection(source, chunking))
        self.bm25 = BM25Okapi([_tok(p["text"]) for p in self.payloads])

    def search(self, query, k=5, pool=30):
        q = np.array(embed_query(query), dtype=np.float32)
        q /= (np.linalg.norm(q) + 1e-9)
        with np.errstate(all="ignore"):            # macOS Accelerate BLAS raises spurious fp flags on matmul
            sims = self.vecs @ q                   # dense cosine, kept for the score
        dense_top = np.argsort(-sims)[:pool]
        bm25_top = np.argsort(-self.bm25.get_scores(_tok(query)))[:pool]

        fused = {}
        for r, i in enumerate(dense_top):
            fused[i] = fused.get(i, 0) + 1 / (RRF_K + r + 1)
        for r, i in enumerate(bm25_top):
            fused[i] = fused.get(i, 0) + 1 / (RRF_K + r + 1)

        # RRF decides the order; the reported score is the dense cosine so it stays
        # on a 0-1 scale comparable to the dense retriever (RRF values are ~0.03 and
        # rank-based, so they can't signal 'nothing relevant' to the critic's gate).
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.payloads[i]["text"], self.payloads[i]["title"],
                    self.payloads[i]["url"], float(sims[i])) for i in top]
