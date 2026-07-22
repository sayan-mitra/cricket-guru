"""Quick retrieval check.

    python -m cricket_guru.retrieval.smoke "your query" --strategy hybrid
"""
import argparse

from cricket_guru.retrieval.base import get_retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--strategy", default="dense", choices=["dense", "hybrid"])
    ap.add_argument("--chunking", default="fixed", choices=["fixed", "structural"])
    a = ap.parse_args()
    for h in get_retriever(a.strategy, a.chunking).search(a.query, k=5):
        print(f"  {h.score:.3f}  {h.title}")
        print(f"         {h.text[:100].strip()}...")


if __name__ == "__main__":
    main()
