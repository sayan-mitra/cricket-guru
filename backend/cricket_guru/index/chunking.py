"""Leg L1 — chunking strategies for Wikipedia prose.

bge-small-en-v1.5 truncates at 512 tokens, so chunks stay well under that
(~1500 chars is roughly 380 tokens). Two variants, selected by name:
  fixed      - equal-size windows with overlap; ignores structure
  structural - packs whole paragraphs up to the cap; respects boundaries
"""
import re

TARGET = 1500      # chars per chunk (~380 tokens, safely under bge's 512)
OVERLAP = 200      # chars shared between adjacent fixed windows


def chunk_fixed(text, target=TARGET, overlap=OVERLAP):
    text = text.strip()
    if len(text) <= target:
        return [text] if text else []
    chunks, start, step = [], 0, target - overlap
    while start < len(text):
        piece = text[start:start + target].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


CLAUSE = re.compile(r"(?=\n\s*\d+\.\d+(?:\.\d+)*[\s\)])")   # a numbered rule clause: 19.4.2, 41.1 …


def chunk_structural(text, target=TARGET):
    # Pack paragraphs up to the cap; never split mid-paragraph unless a single
    # paragraph is itself too big (then fall back to fixed windows for it).
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    # Rulebook PDFs have no blank-line paragraphs but ARE organised by numbered clauses;
    # split on those so each rule lands in its own chunk instead of an arbitrary window.
    if len(paras) <= 1:
        paras = [p.strip() for p in CLAUSE.split(text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(p) > target:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(chunk_fixed(p))
        elif not buf:
            buf = p
        elif len(buf) + len(p) + 2 <= target:
            buf = f"{buf}\n\n{p}"
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


CHUNKERS = {"fixed": chunk_fixed, "structural": chunk_structural}


def chunk(text, strategy):
    return CHUNKERS[strategy](text)
