"""Few-shot loader examples for semantic-layer builds.

A 14B model writes much better pandas loaders by adapting a working loader for a similar export than from scratch.
`loader_examples.json` holds loaders that passed validation on the provided training businesses (generated during
development, see training_code/README.md), each with the raw column layout it was written for.

    pick(store, k=2, max_chars=14000) -> prompt block with the k most similar loaders (empty string if none)

Similarity = Jaccard overlap of folded raw column names, plus a bonus when the industry from the domain summary
matches. An example built from exactly the same data files (same
content hash) is skipped, so evaluating on the training split never shows a business its own loader.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from .parsers import fold_text

LIBRARY = Path(__file__).with_name("loader_examples.json")


def column_signature(store) -> list[str]:
    """Sorted folded column names over every file/sheet of a business (RawStore)."""
    cols: set[str] = set()
    for f in store.list_files():
        for sh in store.sheet_names(f):
            try:
                df = store.read_table(f, sh)
            except Exception:  # noqa: BLE001
                continue
            cols.update(c for c in (fold_text(str(c)) for c in df.columns) if c and not c.startswith("unnamed"))
    return sorted(cols)


def data_hash(biz_dir: str | Path) -> str:
    h = hashlib.sha256()
    for f in sorted(os.listdir(biz_dir)):
        p = Path(biz_dir) / f
        if p.is_file() and not f.lower().startswith("readme"):
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def _library() -> list[dict]:
    return json.loads(LIBRARY.read_text()) if LIBRARY.exists() else []


def pick(store, k: int = 2, max_chars: int = 14000, industry: str | None = None) -> str:
    lib = _library()
    if not lib:
        return ""
    sig = set(column_signature(store))
    own = data_hash(store.dir) if getattr(store, "dir", None) else None
    scored = []
    for ex in lib:
        if own and ex.get("data_hash") == own:
            continue
        other = set(ex["columns"])
        sim = len(sig & other) / max(len(sig | other), 1)
        same_industry = bool(industry) and (fold_text(industry) or "") == (fold_text(ex.get("industry")) or "")
        scored.append((sim + (0.15 if same_industry else 0), sim, ex))
    scored.sort(key=lambda x: -x[0])
    out, used = [], 0
    for score, sim, ex in scored:
        if len(out) >= k or score < 0.05:
            break
        if used + len(ex["code"]) > max_chars:
            continue
        used += len(ex["code"])
        out.append(f"### Example loader ({ex['industry']}, column overlap {sim:.0%})\n"
                   f"Raw columns: {', '.join(ex['columns'][:60])}\n```python\n{ex['code']}\n```")
    if not out:
        return ""
    return ("## Working loaders written for other shops with similar exports\n"
            "They passed validation on their own files. Reuse their structure and helper calls, but map THIS "
            "business's files, sheets, columns and labels exactly as its README and profile say.\n\n"
            + "\n\n".join(out) + "\n\n")
