"""Retriever interface (leg L3). Variants: dense, hybrid."""
from dataclasses import dataclass


@dataclass
class Hit:
    text: str
    title: str
    url: str
    score: float


def get_retriever(strategy, chunking="fixed", source="wiki"):
    """Factory: pick a retrieval variant by name, over a given source."""
    from cricket_guru.retrieval.dense import DenseRetriever
    from cricket_guru.retrieval.hybrid import HybridRetriever
    return {"dense": DenseRetriever, "hybrid": HybridRetriever}[strategy](chunking, source)
