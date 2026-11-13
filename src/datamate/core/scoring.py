"""Offline re-implementation of the DataMate scoring rules (Penalized Accuracy).

    score_one(pred, ref) -> 1 (correct), 0 (abstain/missing), -1 (wrong)
    score_all(preds, refs) -> dict with PA, counts and per-type breakdown

`pred` is a submission line ({"type", "answer"}), `ref` an answers.jsonl line
({"type", "answer", "alias"}). Aliases: string -> [str]; list -> one [str] per
element; table -> one [str] per row (forms of the row's first cell).
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict

_NON_WORD = re.compile(r"[^\w\s]|_")


def norm_str(x) -> str:
    s = unicodedata.normalize("NFC", str(x)).lower()
    s = _NON_WORD.sub(" ", s)
    return " ".join(s.split())


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def num_eq(a, b, decimals: int | None = None) -> bool:
    if not (_is_num(a) and _is_num(b)):
        return False
    if decimals is not None:
        return round(a, decimals) == round(b, decimals)
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-6)


def str_eq(a, ref, aliases=()) -> bool:
    if not isinstance(a, str):
        return False
    na = norm_str(a)
    return any(na == norm_str(f) for f in [ref, *aliases] if f is not None)


def cell_eq(a, ref, aliases=()) -> bool:
    if _is_num(ref):
        return num_eq(a, ref)
    return str_eq(a, ref, aliases)


def _list_eq(a, ref, alias) -> bool:
    if not isinstance(a, list) or len(a) != len(ref):
        return False
    alias = alias or [[] for _ in ref]
    return all(cell_eq(x, r, al or []) for x, r, al in zip(a, ref, alias))


def _table_eq(a, ref, alias) -> bool:
    if not (isinstance(a, dict) and isinstance(a.get("columns"), list) and isinstance(a.get("rows"), list)):
        return False
    rc, rr = ref["columns"], ref["rows"]
    ac, ar = a["columns"], a["rows"]
    if len(ac) != len(rc) or len(ar) != len(rr) or any(not isinstance(r, list) or len(r) != len(rc) for r in ar):
        return False
    pos = {norm_str(c): i for i, c in enumerate(ac)}
    perm = [pos.get(norm_str(c)) for c in rc]
    if None in perm or len(set(perm)) != len(perm):
        perm = list(range(len(rc)))
    alias = alias or [[] for _ in rr]
    unused = list(range(len(ar)))
    for ref_row, al in zip(rr, alias):
        hit = next((k for k in unused if all(
            cell_eq(ar[k][perm[j]], ref_row[j], (al or []) if j == 0 else ()) for j in range(len(rc)))), None)
        if hit is None:
            return False
        unused.remove(hit)
    return True


def score_one(pred: dict | None, ref: dict) -> int:
    if not pred or pred.get("type") in (None, "abstain"):
        return 0
    t, rt = pred["type"], ref["type"]
    if t != rt:
        return -1
    a, r, al = pred.get("answer"), ref.get("answer"), ref.get("alias") or []
    ok = {
        "unanswerable": lambda: True,
        "number": lambda: num_eq(a, r),
        "percent": lambda: num_eq(a, r, 2),
        "string": lambda: str_eq(a, r, al),
        "date": lambda: isinstance(a, str) and a == r,
        "boolean": lambda: isinstance(a, bool) and a == r,
        "list": lambda: _list_eq(a, r, al),
        "table": lambda: _table_eq(a, r, al),
    }.get(t, lambda: False)()
    return 1 if ok else -1


def score_all(preds: dict[str, dict], refs: dict[str, dict]) -> dict:
    per_type: dict[str, Counter] = defaultdict(Counter)
    scores = {}
    for qid, ref in refs.items():
        s = score_one(preds.get(qid), ref)
        scores[qid] = s
        per_type[ref["type"]][{1: "correct", 0: "abstain", -1: "wrong"}[s]] += 1
    n = len(refs) or 1
    tot = Counter(scores.values())
    return {
        "n": len(refs),
        "PA": round(sum(scores.values()) / n, 4),
        "correct": tot[1], "wrong": tot[-1], "abstain": tot[0],
        "accuracy_answered": round(tot[1] / max(tot[1] + tot[-1], 1), 4),
        "per_type": {k: dict(v) for k, v in sorted(per_type.items())},
        "scores": scores,
    }
