"""Retriever interface (leg L3). Variants: dense, hybrid."""
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Hit:
    text: str
    title: str
    url: str
    score: float


@lru_cache(maxsize=None)
def get_retriever(strategy, chunking="fixed", source="wiki"):
    """Factory: pick a retrieval variant by name, over a given source.

    Cached per config, because a retriever IS its index: construction scrolls every point out of
    Qdrant and builds a BM25 table over them (8k+ chunks for wiki), while search only reads that
    state. So one copy can serve every session — which is what lets each browser hold its own
    router without paying for the index again. Bounded by the number of (strategy, chunking,
    source) combinations, not by callers."""
    from cricket_guru.retrieval.dense import DenseRetriever
    from cricket_guru.retrieval.hybrid import HybridRetriever
    return {"dense": DenseRetriever, "hybrid": HybridRetriever}[strategy](chunking, source)
