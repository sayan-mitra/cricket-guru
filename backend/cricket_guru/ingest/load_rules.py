#!/usr/bin/env python3
"""Parse rule-book PDFs in data/rules/ into a chunkable corpus.

Each page becomes a doc with a citation (source + page, plus a leading Law/clause
number when we can spot one). Output shape matches Wikipedia articles.json so the
index builder can reuse the same chunk/embed path.

    python -m cricket_guru.ingest.load_rules
"""
import glob
import json
import re
from pathlib import Path

from pypdf import PdfReader

from cricket_guru.config import DATA_DIR

RULES = DATA_DIR / "rules"
SECTION = re.compile(r"^(Law\s+\d+[A-Za-z.\d]*|\d+\.\d+)")


def parse_pdf(path):
    reader = PdfReader(path)
    src = Path(path).stem
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) < 40:            # skip covers / blank pages
            continue
        m = SECTION.match(text)
        docs.append({
            "title": f"{src} (p{i})",
            "url": f"rulebook:{src}#page{i}",
            "pageid": len(docs),
            "text": text,
            "source": src,
            "page": i,
            "section": m.group(1) if m else None,
        })
    return docs


def main():
    pdfs = sorted(glob.glob(str(RULES / "*.pdf")))
    if not pdfs:
        print(f"No PDFs in {RULES}. See data/rules/README.md for what to drop.")
        return
    docs = []
    for pdf in pdfs:
        d = parse_pdf(pdf)
        docs.extend(d)
        print(f"{Path(pdf).name}: {len(d)} pages")
    (RULES / "rules.json").write_text(json.dumps(docs, indent=2))
    print(f"wrote {len(docs)} rule pages -> {RULES / 'rules.json'}")


if __name__ == "__main__":
    main()
