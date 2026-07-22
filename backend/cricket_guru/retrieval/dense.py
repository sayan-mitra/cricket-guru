"""Leg L3 variant A — dense (vector) retrieval.

Vectors are pulled from Qdrant into memory once, at construction, and searched
with numpy. Query-time never touches the SQLite-backed local Qdrant — which
matters because the agent runs its tools in worker threads (SQLite is thread-
bound). For ~thousands of 384-dim chunks this is a few MB and a single matvec.
"""
import numpy as np

from cricket_guru.config import collection
from cricket_guru.index.embed import embed_query
from cricket_guru.qdrant_store import client
from cricket_guru.retrieval.base import Hit


def load_collection(coll):
    """Scroll all points (payload + vector) into memory. Call from the main thread."""
    qc = client()
    pts, offset = [], None
    while True:
        batch, offset = qc.scroll(coll, limit=1000, offset=offset,
                                  with_payload=True, with_vectors=True)
        pts.extend(batch)
        if offset is None:
            break
    payloads = [p.payload for p in pts]
    vecs = np.array([p.vector for p in pts], dtype=np.float32)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)   # unit-norm for cosine
    return payloads, vecs


def _hit(payload, score):
    return Hit(payload["text"], payload["title"], payload["url"], float(score))


class DenseRetriever:
    def __init__(self, chunking="fixed", source="wiki"):
        self.payloads, self.vecs = load_collection(collection(source, chunking))

    def search(self, query, k=5):
        q = np.array(embed_query(query), dtype=np.float32)
        q /= (np.linalg.norm(q) + 1e-9)
        with np.errstate(all="ignore"):     # macOS Accelerate BLAS raises spurious fp flags on matmul
            sims = self.vecs @ q
        idx = np.argsort(-sims)[:k]
        return [_hit(self.payloads[i], sims[i]) for i in idx]
