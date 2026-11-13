"""Tie the packaged helpers to the rulebook rows they implement.

The helpers encode rulebook definitions (debt, customer identity, "bán được bao nhiêu <đơn vị>", stock at a day...).
The rulebook is an input that changes between test sets, so a helper is only offered while the row it implements
is still in the rulebook the run is given, with the wording it was written for:

    ok       every anchored row is present with its key phrases          -> helper offered and hinted
    absent   none of its rows exists (the rulebook no longer defines it)  -> helper offered, not hinted as a rule
    changed  a row exists but its key phrases are gone (definition moved) -> helper withdrawn

Rulebook-dependent switches inside the helpers (does trả góp count as debt? do delivering orders count as
revenue?) are read from the rulebook text too (flags()).
"""
from __future__ import annotations

from .qparse import _rule_tokens

# helper -> [(row key, phrases the row text must contain)], all folded (no diacritics, lowercase)
ANCHORS: dict[str, list[tuple[str, list[str]]]] = {
    "debts": [("don duoc tinh vao no", ["khong huy", "tra hang van tinh"]),
              ("no mot don tai ngay", ["tong don"]),
              ("don cod don ban qua san tmdt", ["khong phai khach no"]),
              ("tuoi no", ["ngay dat"])],
    "debt_flow": [("no them tra no thu no trong ky", ["ghi no", "no them", "tra no"])],
    "qty_in": [("ban duoc bao nhieu don vi", ["don vi co ban la"])],
    "with_customer": [("xac dinh khach", ["ma khach", "chi khac cach viet"])],
    "top_customers": [("xac dinh khach", ["ma khach", "chi khac cach viet"])],
    "stock_at": [("den het ngay tinh den hom nay", ["tu dau du lieu"])],
    "avg_price": [("gia nhap binh quan mot mat hang trong thang", ["theo so luong nhap"])],
    "unit_factor": [("ton theo don vi lon", ["don vi lon nguyen"])],
}


def _norm(s: str) -> str:
    return " ".join(_rule_tokens(s))


def _rows(rules: list[dict]) -> dict[str, str]:
    return {_norm(r["key"]): _norm(r["text"]) for r in rules}


def status(rules: list[dict]) -> dict[str, str]:
    """helper -> 'ok' | 'absent' | 'changed' for the given parsed rulebook (qparse.parse_rulebook)."""
    rows = _rows(rules)
    out = {}
    for helper, anchors in ANCHORS.items():
        found = [(rows[k], ph) for k, ph in anchors if k in rows]
        if not found:
            out[helper] = "absent"
        elif all(p in text for text, ph in found for p in ph):
            out[helper] = "ok"
        else:
            out[helper] = "changed"
    return out


def rule_text(rules: list[dict], helper: str) -> str:
    """The current rulebook rows a helper implements, verbatim ('' if none)."""
    keys = {k for k, _ in ANCHORS.get(helper, [])}
    return " | ".join(f"{r['key']}: {r['text']}" for r in rules if _norm(r["key"]) in keys)


def flags(rules: list[dict]) -> dict[str, bool]:
    """Switches the helpers read (helpers.RULE_FLAGS), from the rulebook wording; defaults when undefined."""
    texts = [_norm(r["key"] + " " + r["text"]) for r in rules]
    revenue = _rows(rules).get("don tinh vao doanh thu")
    return {
        # a rule saying instalment orders are debt orders (ghi nợ) -> trả góp counts in debts()/debt_flow()
        "installment_is_credit": any("tra gop" in t and ("ghi no" in t or "khach no" in t) for t in texts),
        "delivering_counts": True if revenue is None else "dang giao van tinh" in revenue,
    }
