"""Build a vector index for one source and chunking variant.

    python -m cricket_guru.index.build_index --source wiki  --chunking fixed
    python -m cricket_guru.index.build_index --source rules --chunking fixed

Each (source, chunking) lands in its own collection (e.g. wiki_fixed, rules_fixed)
so experiments swap between them without re-indexing.
"""
import argparse
import json
import uuid

from qdrant_client.models import Distance, PointStruct, VectorParams

from cricket_guru.config import DATA_DIR, EMBED_DIM, collection
from cricket_guru.index.chunking import chunk
from cricket_guru.index.embed import embed_passages
from cricket_guru.qdrant_store import client

BATCH = 256
SOURCES = {
    "wiki": DATA_DIR / "wikipedia" / "articles.json",
    "rules": DATA_DIR / "rules" / "rules.json",
}


def build(source, strategy):
    docs = json.loads(SOURCES[source].read_text())
    qc = client()
    coll = collection(source, strategy)
    if qc.collection_exists(coll):
        qc.delete_collection(coll)
    qc.create_collection(
        coll, vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE))

    texts, payloads = [], []
    for d in docs:
        for i, piece in enumerate(chunk(d["text"], strategy)):
            texts.append(piece)
            payloads.append({
                "text": piece, "title": d["title"], "url": d["url"],
                "pageid": d["pageid"], "chunk_index": i, "chunking": strategy,
                "source": source, "section": d.get("section"),
            })

    done = 0
    for s in range(0, len(texts), BATCH):
        vecs = embed_passages(texts[s:s + BATCH])
        points = [PointStruct(id=str(uuid.uuid4()), vector=v, payload=p)
                  for v, p in zip(vecs, payloads[s:s + BATCH])]
        qc.upsert(coll, points=points)
        done += len(points)
        print(f"  upserted {done}/{len(texts)} chunks")

    print(f"done: {len(docs)} docs -> {done} chunks in '{coll}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="wiki")
    ap.add_argument("--chunking", choices=["fixed", "structural"], default="fixed")
    a = ap.parse_args()
    build(a.source, a.chunking)


if __name__ == "__main__":
    main()
