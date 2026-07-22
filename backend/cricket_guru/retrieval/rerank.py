"""Cross-encoder reranker — re-score the top-N bi-encoder hits by reading query + chunk together.

The bi-encoder (bge-small) embeds query and chunk separately, so it ranks fast but coarsely — the
right chunk often lands in the top-5 but not at #1. A cross-encoder reads the pair jointly and scores
relevance directly, which fixes 'right chunk, wrong rank'. Ships with FastEmbed (no extra dependency).
"""
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

RERANK_MODEL = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def _encoder():
    return TextCrossEncoder(model_name=RERANK_MODEL)


def rerank(query, hits, top_k=None):
    """Reorder hits (each carrying .text) by cross-encoder score; return the top_k (or all)."""
    if not hits:
        return hits
    scores = list(_encoder().rerank(query, [h.text for h in hits]))
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    ranked = [hits[i] for i in order]
    return ranked[:top_k] if top_k else ranked
