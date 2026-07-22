"""Arm 1 — text-RAG. Retrieve prose chunks, answer only from them.

Chunking (L1) and retrieval (L3) variants are injected, so the same arm serves
every cell of those experiments.
"""
from cricket_guru.arms.base import Answer
from cricket_guru.llm import agent
from cricket_guru.retrieval.base import get_retriever

SYS = (
    "You are a cricket expert. Answer the question using ONLY the provided "
    "context passages. If the context does not contain the answer, say you do "
    "not have enough information — do not guess. Keep it to a few sentences."
)


class TextRAGArm:
    def __init__(self, retrieval="dense", chunking="fixed", source="wiki", k=5, rerank=False):
        self._retrieval, self._chunking, self._source, self.k, self._rerank = \
            retrieval, chunking, source, k, rerank
        self._retriever = None            # lazy: don't open Qdrant until asked
        self.llm = agent(SYS)

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = get_retriever(self._retrieval, self._chunking, self._source)
        return self._retriever

    def answer(self, question):
        # With rerank on, pull a wider net and let the cross-encoder pick the top-k — it lifts wiki
        # recall@1 sharply (the right passage is usually retrieved but ranked too low).
        fetch = 20 if self._rerank else self.k
        hits = self.retriever.search(question, k=fetch)
        if self._rerank:
            from cricket_guru.retrieval.rerank import rerank
            hits = rerank(question, hits, top_k=self.k)
        context = "\n\n".join(f"[{h.title}]\n{h.text}" for h in hits)
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        out = self.llm.run_sync(prompt).output
        sources = [{"title": h.title, "url": h.url} for h in hits]
        top = max((h.score for h in hits), default=None)
        tag = f"{self._retrieval}/{self._chunking}/{self._source}{'+rerank' if self._rerank else ''}"
        steps = [
            {"name": f"retrieve ({tag}, k={self.k})",
             "input": question,
             "hits": [{"cosine": round(h.score, 3), "title": h.title,
                       "snippet": " ".join(h.text.split())[:240]} for h in hits],
             "output": "\n".join(f"{h.score:.3f}  {h.title}" for h in hits) or "(no hits)"},
            {"name": "answer", "input": f"[system]\n{SYS}\n\n[user]\n{prompt}", "output": out},
        ]
        return Answer(text=out, arm="text_rag", sources=sources, steps=steps,
                      tool_trace=["semantic_search"], evidence=context,
                      retrieval_score=top)
