"""Refill product attributes the loader left empty, from the shop's own catalogue sheet.

A loader often loses catalogue attributes (a merge that suffixed the columns, a sheet it did not read): brand,
cost price, base unit, prescription flag... Which raw column holds which attribute is decided by the LLM from the
column headers and sample values (one call per business, only when some attribute is empty and a catalogue sheet
exists); the refill itself is deterministic. Catalogue sheets are found from the data: a sheet with a column whose
values are the layer's product codes or names.

    propose(llm, tables, store) -> {"version", "map": {attr: {"file", "sheet", "header", "true_values"?}}}
    enrich(tables, store, meta, sources=None) -> list of fixes (products updated in place in `tables`)
"""
from __future__ import annotations

import re
import sys

import pandas as pd

from ..core import parsers
from .schema import TABLES

VERSION = 1
_SKIP = ("product_id", "name", "aliases")
# Used only when no LLM mapping exists (layers built before this module): the two pharmacy attributes the
# previous keyword rule handled.
_FALLBACK = {"active_ingredient": re.compile(r"^hoat chat|^thanh phan( chinh)?$|active ingredient"),
             "is_prescription": re.compile(r"ke don|prescription|\betc\b")}
_TRUE = ("true", "1", "x", "co", "yes", "c", "y")


def _attrs() -> dict[str, tuple[str, str]]:
    return {c.name: (c.dtype, c.desc) for c in TABLES["products"].cols if c.name not in _SKIP}


def blank_attrs(p: pd.DataFrame) -> list[str]:
    """Schema attributes that are missing, all null, or (for flags) constant."""
    out = []
    for c, (typ, _) in _attrs().items():
        if c not in p:
            out.append(c)
            continue
        v = p[c].dropna()
        if v.empty or (typ == "bool" and v.astype(str).nunique() <= 1):
            out.append(c)
    return out


def catalogue_sheets(store, p: pd.DataFrame, max_rows_factor: int = 5) -> list[tuple[str, object, pd.DataFrame, pd.Series]]:
    """[(file, sheet, frame, product_id per row)] for raw sheets keyed by the layer's product codes or names."""
    ids = set(p["product_id"].dropna().astype(str).str.strip())
    names = {parsers.fold_text(x): i for x, i in zip(p.get("name", p["product_id"]), p["product_id"])
             if isinstance(x, str)}
    out = []
    for f in store.list_files():
        try:
            sheets = store.sheet_names(f) or [None]
        except Exception:  # noqa: BLE001
            sheets = [None]
        for sh in sheets:
            try:
                df = store.read_table(f, sh)
            except Exception:  # noqa: BLE001
                continue
            if not len(df) or len(df) > max_rows_factor * max(len(p), 1):
                continue  # a sales sheet, not a catalogue
            for c in df.columns:
                col = df[c].astype(object).where(df[c].notna(), None)
                s = col.dropna().astype(str).str.strip()
                if not len(s):
                    continue
                if s.isin(ids).mean() > 0.8:
                    key = col.map(lambda v: str(v).strip() if v is not None else None)
                elif s.map(parsers.fold_text).isin(names).mean() > 0.8:
                    key = col.map(lambda v: names.get(parsers.fold_text(v)) if v is not None else None)
                else:
                    continue
                out.append((f, sh, df.drop(columns=[c]), key))
                break
    return out


def _norm_val(v) -> str | None:
    if v is None or (isinstance(v, float) and v != v):
        return None
    t = parsers.fold_text(str(v).strip()) or ""
    return re.sub(r"\.0$", "", t) or None


def _same_split(a: pd.Series, b: pd.Series) -> bool:
    """Two categorical columns partition the products the same way (each value of one maps to one value of the
    other, both ways), e.g. a 'x'/blank column and a True/False flag."""
    if a.nunique() < 2 or a.nunique() != b.nunique():
        return False
    ct = pd.crosstab(a, b).values
    n = ct.sum()
    return ct.max(axis=1).sum() / n >= 0.95 and ct.max(axis=0).sum() / n >= 0.95


def used_columns(p: pd.DataFrame, df: pd.DataFrame, key: pd.Series) -> set:
    """Raw columns whose per-product values equal a product attribute the layer already has: the loader used them,
    so they must not be offered for another attribute (e.g. 'Hoạt chất' already in active_ingredient is not a
    brand; 'Thuốc kê đơn' already in is_prescription is not 'still on sale')."""
    filled = [c for c in p.columns if c != "product_id" and p[c].notna().any()]
    if not filled:
        return set()
    lay = p.assign(_k=p["product_id"].astype(str).str.strip()).drop_duplicates("_k").set_index("_k")
    flags = {c for c in filled if p[c].dropna().map(type).isin([bool]).all() or str(p[c].dtype) == "boolean"}
    out = set()
    for c in df.columns:
        raw = pd.Series(df[c].values, index=key.values)
        raw = raw[raw.index.notna() & ~raw.index.duplicated()].map(_norm_val)
        for a in filled:
            have = lay[a].reindex(raw.index)
            if a in flags:  # blank cells are a value of a flag column ('x' / blank)
                both = have.notna()
                if both.sum() >= 3 and _same_split(raw[both].fillna("<blank>"), have[both].astype(str)):
                    out.add(c)
                    break
                continue
            hv = have.map(_norm_val)
            both = hv.notna() & raw.notna()
            if both.sum() >= 3 and (raw[both] == hv[both]).mean() >= 0.95:
                out.add(c)
                break
    return out


def _samples(s: pd.Series, n: int = 6) -> list[str]:
    return [str(v)[:40] for v in s.dropna().astype(str).str.strip().loc[lambda x: x != ""].unique()[:n]]


def propose(llm, tables: dict, store) -> dict | None:
    """LLM: which catalogue column holds each empty attribute. None when the call fails."""
    from .mapper import loads_lenient
    p = tables.get("products")
    if p is None or "product_id" not in p:
        return {"version": VERSION, "map": {}}
    blank = blank_attrs(p)
    sheets = catalogue_sheets(store, p) if blank else []
    if not sheets:
        return {"version": VERSION, "map": {}}
    attrs = _attrs()
    cols = []
    for f, sh, df, key in sheets:
        used = used_columns(p, df, key)
        for c in df.columns:
            if c in used:
                continue
            sm = _samples(df[c])
            if sm:
                cols.append({"file": f, "sheet": sh, "header": str(c), "samples": sm})
    text = ("A product catalogue was loaded but these product attributes are empty:\n"
            + "\n".join(f"- {c} ({attrs[c][0]}): {attrs[c][1]}" for c in blank)
            + "\n\nColumns of the shop's raw catalogue sheets (header and sample values):\n"
            + "\n".join(f"- file={c['file']!r} sheet={c['sheet']!r} header={c['header']!r}: {c['samples']}"
                        for c in cols[:80])
            + "\n\nFor each empty attribute that one of these columns clearly records, give its source. Skip an "
              "attribute when no column records exactly that: the column's MEANING (header and values) must be the "
              "attribute's meaning — an ingredient is not a brand, a prescription flag is not 'still on sale', a "
              "promotion date is not a product flag. Most attributes usually have no source; an empty JSON object "
              "is a normal answer. For a bool attribute also give the raw values that mean true. Return ONLY JSON:\n"
              '{"<attribute>": {"file": ..., "sheet": ..., "header": ..., "true_values": [...]}, ...}')
    try:
        d = loads_lenient(llm.chat([{"role": "user", "content": text}], max_tokens=2000, thinking=False).content)
    except Exception as e:  # noqa: BLE001 - optional
        print(f"[attrs] proposal skipped: {e!r}", file=sys.stderr, flush=True)
        return None
    valid = {(c["file"], str(c["sheet"]), c["header"]) for c in cols}
    m = {}
    for a, src in (d or {}).items() if isinstance(d, dict) else []:
        if a in blank and isinstance(src, dict) and \
                (src.get("file"), str(src.get("sheet")), src.get("header")) in valid:
            m[a] = {k: src.get(k) for k in ("file", "sheet", "header")}
            if attrs[a][0] == "bool":
                m[a]["true_values"] = [str(v) for v in src.get("true_values") or []]
    return {"version": VERSION, "map": m}


def _convert(vals: pd.Series, typ: str, true_values=None) -> pd.Series:
    if typ == "float":
        return vals.map(parsers.parse_money)
    if typ == "bool":
        tv = {parsers.fold_text(v) for v in (true_values or [])} or set(_TRUE)
        return vals.map(lambda v: None if v is None or (isinstance(v, float) and v != v)
                        else (parsers.fold_text(str(v)) or "") in tv)
    return vals.map(lambda v: None if v is None or (isinstance(v, float) and v != v) or not str(v).strip()
                    else str(v).strip())


def enrich(tables: dict, store, meta: dict, sources: dict | None = None) -> list[str]:
    """Fill empty product attributes from catalogue sheets: `sources` (propose()['map']) first, else the
    pharmacy keyword fallback. Returns the fixes made (also appended to meta['fixes'])."""
    p = tables.get("products")
    if p is None or "product_id" not in p:
        return []
    blank = blank_attrs(p)
    if not blank:
        return []
    attrs = _attrs()
    sheets = catalogue_sheets(store, p)
    fixes = []
    pid = p["product_id"].astype(str).str.strip()
    for a in blank:
        src = (sources or {}).get(a)
        for f, sh, df, key in sheets:
            if src:
                if (f, str(sh)) != (src["file"], str(src["sheet"])) or src["header"] not in map(str, df.columns):
                    continue
                col = next(c for c in df.columns if str(c) == src["header"])
            else:
                rx = _FALLBACK.get(a)
                used = used_columns(p, df, key) if rx else set()
                col = next((c for c in df.columns if rx and c not in used
                            and rx.search(parsers.fold_text(str(c)) or "")), None)
                if col is None:
                    continue
            vals = pd.Series(df[col].values, index=key.values)
            vals = vals[vals.index.notna() & ~vals.index.duplicated()]
            conv = _convert(vals, attrs[a][0], (src or {}).get("true_values"))
            if conv.dropna().empty or (attrs[a][0] == "bool" and conv.dropna().nunique() < 2):
                continue
            new = pid.map(conv)
            if attrs[a][0] == "bool":
                new = new.astype("boolean")
            p[a] = new
            fixes.append(f"products.{a} filled from raw {f}/{sh} column {col!r}")
            break
    if fixes:
        tables["products"] = p
        meta.setdefault("fixes", []).extend(fixes)
    return fixes
