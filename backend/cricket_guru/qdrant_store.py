"""Qdrant client — on-disk file mode locally, service URL in Docker.

Local file mode holds a single-process lock, which is fine here (the indexer
and the app never run against it at the same time). Set CG_QDRANT_URL to point
at a Qdrant server instead.
"""
from functools import lru_cache

from qdrant_client import QdrantClient

from cricket_guru.config import QDRANT_PATH, QDRANT_URL


@lru_cache(maxsize=1)
def client():
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=QDRANT_PATH)
