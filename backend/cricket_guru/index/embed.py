"""Embeddings via FastEmbed (bge-small over ONNX, no torch).

bge-small-en-v1.5 retrieves well without a query instruction prefix, so queries
and passages are embedded the same way. The model loads once and is cached.
"""
from functools import lru_cache

from fastembed import TextEmbedding

from cricket_guru.config import EMBED_MODEL


@lru_cache(maxsize=1)
def _model():
    return TextEmbedding(EMBED_MODEL)


def embed_passages(texts):
    return [v.tolist() for v in _model().embed(list(texts))]


def embed_query(text):
    return next(iter(_model().embed([text]))).tolist()
