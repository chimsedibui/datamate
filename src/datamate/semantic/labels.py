"""Open-vocabulary labels classified by the LLM, once per business, instead of keyword lists.

Shops write their own labels: sales channels ("Shopee", "Sàn TMĐT", a platform launched next year), status texts
("GD thành công", "Khách bom hàng") and walk-in markers ("KL", "khách qua đường"). Keyword lists in code miss any
label they were not written for and fail silently. Here one LLM call per business reads the shop's README and the
distinct values actually present, and maps each value to the canonical enums. The result is stored in
meta["labels"] and applied every time the layer is loaded (apply_labels). Keyword rules remain only as a fallback
for values the LLM did not classify, or when no LLM result is available.

    classify(llm, tables, readme, rulebook) -> dict | None     (meta["labels"])
    apply_labels(tables, meta) -> tables                       (adds orders.channel_type; status / walk-in fixes)
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from ..core.parsers import fold_text, norm_text, status_hint
from .schema import CHANNEL_TYPE, ORDER_STATUS

VERSION = 1
MAX_VALUES = 40          # distinct values sent per column
TOP_CUSTOMERS = 25       # walk-in markers are always among the most frequent customer values
_MARKETPLACE_FALLBACK = re.compile(r"\bsan\b|tmdt|marketplace|e-?commerce|shopee|lazada|tiki|tiktok|sendo")


def _counts(s: pd.Series, n: int) -> list[tuple[str, int]]:
    vc = s.dropna().astype(str).str.strip()
    vc = vc[vc != ""].value_counts().head(n)
    return [(str(k), int(v)) for k, v in vc.items()]


def _rule_rows(rulebook: str, keys: tuple[str, ...]) -> str:
    """Rulebook table rows whose first cell mentions one of `keys` (folded), verbatim."""
    out = []
    for ln in rulebook.splitlines():
        if ln.startswith("|") and any(k in (fold_text(ln.split("|")[1]) or "") for k in keys):
            out.append(ln.strip())
    return "\n".join(out[:8])


def _prompt(o: pd.DataFrame, readme: str, rulebook: str) -> tuple[str, dict] | None:
    parts, asked = [], {}
    if "channel" in o and o["channel"].notna().any():
        vals = _counts(o["channel"], MAX_VALUES)
        asked["channel"] = [v for v, _ in vals]
        parts.append('"channel": map EVERY value below to one of ' + str(list(CHANNEL_TYPE)) +
                     ". marketplace = an e-commerce platform that sells for the shop (any platform name, known or "
                     "new); chat = orders taken over a messaging app or social page; website = the shop's own site; "
                     "delivery_app = food/ride delivery apps; phone = phone orders; store = at the counter.\n"
                     + "\n".join(f"  {v!r} ({n} orders)" for v, n in vals))
    if "status_raw" in o and "status" in o and o["status_raw"].notna().any():
        pairs = (o[["status_raw", "status"]].dropna(subset=["status_raw"]).astype(str)
                 .value_counts().head(MAX_VALUES))
        asked["status"] = sorted({str(r) for (r, _), _ in pairs.items()})
        parts.append('"status": map EVERY raw label below to one of ' + str(list(ORDER_STATUS)) +
                     " following the README and rulebook (current mapping shown; correct it only when the label "
                     "clearly means something else).\n"
                     + "\n".join(f"  {r!r} -> currently {s!r} ({n} orders)" for (r, s), n in pairs.items()))
    who = [c for c in ("customer_name", "customer_id") if c in o and o[c].notna().any()]
    if who:
        vals = _counts(o[who[0]], TOP_CUSTOMERS)
        asked["walk_in"] = [v for v, _ in vals]
        parts.append(f'"walk_in": list the values below (most frequent {who[0]} values) that are NOT a real '
                     "customer but a walk-in / anonymous / placeholder marker (khách lẻ, vãng lai, a dash, "
                     "'no name'...). A real person or company name is never a marker. Empty list if none.\n"
                     + "\n".join(f"  {v!r} ({n} orders)" for v, n in vals))
    if not parts:
        return None
    rules = _rule_rows(rulebook, ("trang thai", "khach", "cod", "san", "doanh thu", "huy", "tra hang"))
    intro = re.split(r"\n## ", readme or "", maxsplit=1)[0][:1500]
    conv = re.search(r"## Quy ước.*", readme or "", re.S)
    text = ("You classify the labels one Vietnamese shop writes in its sales records into fixed categories.\n\n"
            f"## Shop README\n{intro}\n{conv.group(0)[:2500] if conv else ''}\n\n"
            f"## Rulebook rows\n{rules}\n\n"
            "## Tasks\n" + "\n\n".join(parts) +
            "\n\nReturn ONLY a JSON object with the keys asked above: \"channel\" and \"status\" as objects "
            "{raw value: category} covering every listed value (copy each value exactly), \"walk_in\" as a list.")
    return text, asked


def classify(llm, tables: dict, readme: str = "", rulebook: str = "") -> dict | None:
    """One LLM call -> {"version", "channel": {raw: type}, "status": {raw: status}, "walk_in": [...]}, or None."""
    from .mapper import loads_lenient
    o = tables.get("orders")
    if o is None or not len(o):
        return None
    pr = _prompt(o, readme, rulebook)
    if pr is None:
        return {"version": VERSION}
    text, asked = pr
    try:
        res = llm.chat([{"role": "user", "content": text}], max_tokens=3000, thinking=False)
        d = loads_lenient(res.content)
    except Exception as e:  # noqa: BLE001 - optional: keyword fallbacks still apply
        print(f"[labels] classification skipped: {e!r}", file=sys.stderr, flush=True)
        return None
    if not isinstance(d, dict):
        return None
    out: dict = {"version": VERSION}
    for key, enum in (("channel", CHANNEL_TYPE), ("status", ORDER_STATUS)):
        got = d.get(key) if isinstance(d.get(key), dict) else {}
        want = {norm_text(v): v for v in asked.get(key, [])}
        m = {want[norm_text(k)]: norm_text(v) for k, v in got.items()
             if norm_text(k) in want and norm_text(v) in enum}
        if m:
            out[key] = m
    wi = d.get("walk_in") if isinstance(d.get("walk_in"), list) else []
    asked_wi = {norm_text(v): v for v in asked.get("walk_in", [])}
    out["walk_in"] = sorted({asked_wi[norm_text(v)] for v in wi if norm_text(v) in asked_wi})
    return out


def channel_type(value, mapping: dict | None = None) -> str | None:
    """Canonical channel of one raw value: the LLM mapping, else a keyword fallback for marketplaces only."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if mapping:
        nv = norm_text(value)
        for k, v in mapping.items():
            if norm_text(k) == nv:
                return v
    if norm_text(value) in CHANNEL_TYPE:
        return norm_text(value)
    return "marketplace" if _MARKETPLACE_FALLBACK.search(fold_text(value) or "") else None


def apply_labels(tables: dict, meta: dict) -> dict:
    """Apply meta["labels"] to the orders table (no-op without orders). Always adds channel_type when the shop
    records a channel, using the keyword fallback when no LLM labels exist."""
    o = tables.get("orders")
    if o is None:
        return tables
    lab = meta.get("labels") or {}
    o = o.copy()
    changed = {}
    if "channel" in o:
        o["channel_type"] = o["channel"].map(lambda v: channel_type(v, lab.get("channel"))).astype(object)
    if lab.get("status") and "status_raw" in o and "status" in o:
        m = {norm_text(k): v for k, v in lab["status"].items()}
        new = o["status_raw"].map(lambda v: m.get(norm_text(v)) if isinstance(v, str) else None)
        # a label whose keyword is unambiguous keeps it when the LLM disagrees: both are shown in the log
        hint = o["status_raw"].map(status_hint)
        conflict = new.notna() & hint.notna() & (new != hint)
        use = new.notna() & ~conflict & (new != o["status"])
        if use.any():
            changed["status"] = {f"{r}: {a} -> {b}": int(n) for (r, a, b), n in
                                 pd.DataFrame({"r": o.loc[use, "status_raw"], "a": o.loc[use, "status"],
                                               "b": new[use]}).value_counts().items()}
            o.loc[use, "status"] = new[use]
    if lab.get("walk_in"):
        marks = {norm_text(v) for v in lab["walk_in"]}
        for c in ("customer_name", "customer_id"):
            if c in o:
                hit = o[c].map(lambda v: isinstance(v, str) and norm_text(v) in marks)
                if hit.any():
                    changed[f"walk_in.{c}"] = int(hit.sum())
                    o[c] = o[c].astype(object).where(~hit, None)
        who = [c for c in ("customer_name", "customer_id") if c in o]
        if "customer_group" in o and who:
            o["customer_group"] = o["customer_group"].astype(object).where(o[who].notna().any(axis=1), None)
    if changed:
        meta["labels_applied"] = changed
    return {**tables, "orders": o}
