"""Post-processing and validation of canonical tables produced by a loader.

`finalize(tables, meta)` coerces dtypes, derives helper columns and applies the
file-convention rules that must hold for every shop (so the LLM does not have to
remember them). `validate(tables, meta)` returns (errors, report): errors make the
mapper retry; the report is a human/LLM-readable summary of what was loaded.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..core.parsers import fold_text, norm_text, status_hint
from .schema import ORDER_STATUS, PAYMENT_METHOD, TABLES


def _coerce(df: pd.DataFrame, table: str) -> pd.DataFrame:
    spec = TABLES.get(table)
    if spec is None:
        return df
    df = df.copy()
    for c in spec.cols:
        if c.name not in df.columns:
            continue
        s = df[c.name]
        if c.dtype in ("float", "int"):
            df[c.name] = pd.to_numeric(s, errors="coerce").astype(float)
        elif c.dtype == "datetime":
            df[c.name] = pd.to_datetime(s, errors="coerce")
        elif c.dtype == "date":
            df[c.name] = pd.to_datetime(s, errors="coerce").dt.normalize()
        elif c.dtype == "bool":
            df[c.name] = s.map(lambda v: None if pd.isna(v) else bool(v)).astype("boolean")
        elif c.dtype == "str":
            df[c.name] = s.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v))
                               else str(v).strip() or None).astype(object)
    return df


def _coalesce_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """One column per name: duplicates merged left to right (first non-null value wins)."""
    out = {}
    for name in dict.fromkeys(df.columns):
        sub = df.loc[:, df.columns == name]
        out[name] = sub.bfill(axis=1).iloc[:, 0] if sub.shape[1] > 1 else sub.iloc[:, 0]
    return pd.DataFrame(out, index=df.index)


# Export subtotal rows ("Cộng ngày 01/05/2025", "Tổng cộng", "Total:") read as records double every sum.
_SUBTOTAL = re.compile(r"^(cộng|tổng cộng|tổng số|tổng|total|subtotal|grand total|tong cong|cong ngay|tong ngay)"
                       r"(\s+(ngày|tháng|tuần|kỳ|năm|ca|tiền|hoá đơn|hóa đơn|đơn|phiếu|ngay|thang|tuan|tien)\b"
                       r"|\s*[:\-]|\s*\d|\s*$)")  # with diacritics: "Tống" (a surname) must not match "tổng"
_TXN_TABLES = {"orders", "order_lines", "purchases", "payments", "returns", "stock_moves", "deliveries"}


_TXN_ID = {"orders": "order_id", "order_lines": "order_id", "purchases": "purchase_id", "payments": "payment_id",
           "returns": "return_id", "stock_moves": "move_id", "deliveries": "delivery_id"}


def _id_prefix(v) -> str:
    return re.sub(r"[\d\W_]+", "", str(v)).upper()


def _drop_subtotal_rows(df: pd.DataFrame, table: str = "") -> tuple[pd.DataFrame, int]:
    """Drop rows labelled like a subtotal whose record id is missing or synthesised (prefix unlike the rest).
    A row with a real id is a real record even if a text cell says 'Tổng ngày …' (seen as a junk customer name)."""
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        s = df[c]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        vals = s.dropna().unique()
        hits = {v for v in vals if isinstance(v, str) and _SUBTOTAL.match(norm_text(v) or "")}
        if hits:
            mask |= s.isin(hits)
    if not mask.any():
        return df, 0
    idc = _TXN_ID.get(table)
    if idc in df.columns:
        ids = df[idc]
        pref = ids[~mask & ids.notna()].map(_id_prefix)
        common = set(pref.value_counts().index[:3]) if len(pref) else set()
        real = ids.notna() & ids.map(_id_prefix).isin(common) if common else ids.notna()
        mask &= ~real
    mask |= _footer_lines(df) | _total_rows(df, table)
    return (df[~mask], int(mask.sum())) if mask.any() else (df, 0)


_AMOUNT = {"order_lines": "line_amount", "purchases": "amount"}


def _total_rows(df: pd.DataFrame, table: str) -> pd.Series:
    """Rows that ARE the total of their own order/receipt, whatever their label says (any language, any wording):
    no quantity and no unit price, and an amount equal to the sum of the other rows of the same id (at least two).
    An export that writes the order total under each order otherwise doubles every sum."""
    none = pd.Series(False, index=df.index)
    idc, amt = _TXN_ID.get(table), _AMOUNT.get(table)
    num = [c for c in ("qty", "qty_base", "unit_price", "unit_cost") if c in df.columns]
    if idc not in df.columns or amt not in df.columns or not num:
        return none
    cand = df[num].isna().all(axis=1) & df[amt].notna() & df[idc].notna()
    if not cand.any():
        return none
    a = pd.to_numeric(df[amt], errors="coerce")
    tot = a.groupby(df[idc]).transform("sum")
    n = df.groupby(idc)[idc].transform("size")
    other = tot - a
    return cand & (n >= 3) & (a.abs() > 0) & ((a - other).abs() <= 1)


_NOTE = re.compile(r"^(ghi chú|ghi chu|lưu ý|luu y|note|chú thích|chu thich)\b")


def _footer_lines(df: pd.DataFrame) -> pd.Series:
    """Sheet footers ('Cộng' month total, 'Ghi chú: …') read as an item line of the last order (its id was
    forward-filled): the item name is a total/note label and the line has no quantity and no unit price."""
    none = pd.Series(False, index=df.index)
    if "product_name" not in df.columns:
        return none
    num = [c for c in ("qty", "qty_base", "unit_price") if c in df.columns]
    if not num:
        return none
    label = df["product_name"].map(lambda v: isinstance(v, str) and bool(
        _SUBTOTAL.match(norm_text(v) or "") or _NOTE.match(norm_text(v) or "")))
    return label & df[num].isna().all(axis=1)


def sanitize(tables: dict, meta: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    """Repair loader output that is structurally off but unambiguous, instead of failing the attempt.

    Every repair is listed in meta['sanitized'] (shown to the LLM in the review report):
    duplicate column names are coalesced, non-canonical table names become extra_<name>, unknown columns of
    canonical tables are dropped, empty tables are dropped, missing extra_desc entries are filled in.
    """
    meta = dict(meta or {})
    fixes: list[str] = []
    desc = dict(meta.get("extra_desc") or {})
    out: dict[str, pd.DataFrame] = {}
    footer_orders: set = set()
    for name, df in (tables or {}).items():
        if not isinstance(df, pd.DataFrame):
            continue
        if df.columns.duplicated().any():
            dups = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
            df = _coalesce_duplicates(df)
            fixes.append(f"{name}: duplicate columns {dups} coalesced (first non-null wins)")
        spec = TABLES.get(name)
        if spec is None and not name.startswith("extra_"):
            new = f"extra_{name}"
            fixes.append(f"table '{name}' is not canonical -> renamed {new}")
            if name in desc:
                desc[new] = desc.pop(name)
            name = new
        elif spec is not None:
            unknown = [c for c in df.columns if spec.col(c) is None]
            if unknown:
                df = df.drop(columns=unknown)
                fixes.append(f"{name}: dropped non-schema columns {unknown}")
        if spec is not None and name in _TXN_TABLES:
            before = df
            df, n_sub = _drop_subtotal_rows(df, name)
            if n_sub:
                fixes.append(f"{name}: dropped {n_sub} subtotal/total rows (e.g. 'Cộng ngày …', 'Tổng cộng')")
                if name == "order_lines" and "order_id" in df.columns:
                    footer_orders |= set(before.loc[~before.index.isin(df.index), "order_id"].dropna())
        if len(df) == 0:
            fixes.append(f"{name}: empty table dropped")
            continue
        if name.startswith("extra_") and not desc.get(name):
            desc[name] = f"{name[6:].replace('_', ' ')}: columns {list(df.columns)[:12]}"
            fixes.append(f"{name}: extra_desc auto-filled")
        out[name] = df
    if footer_orders and "orders" in out and "order_lines" in out and "goods_total" in out["orders"]:
        # the loader summed the dropped total lines into goods_total: recompute it for those orders
        o, ln = out["orders"].copy(), out["order_lines"]
        hit = o["order_id"].isin(footer_orders)
        o.loc[hit, "goods_total"] = o.loc[hit, "order_id"].map(ln.groupby("order_id")["line_amount"].sum())
        out["orders"] = o
        fixes.append(f"orders: goods_total recomputed from lines for {int(hit.sum())} orders that had total lines")
    meta["extra_desc"] = desc
    if fixes:
        meta["sanitized"] = fixes
    return out, meta


WALK_IN = {"kl", "le", "khach le", "khach vang lai", "vang lai", "kh le", "khach", "khach le/vang lai",
           "<na>", "nan", "none", "null", "-", "--", "n/a", "na"}


def null_walk_in(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Walk-in markers on orders ('KL', 'Khách lẻ', '<NA>' left by astype(str)...) -> null (rulebook: no group,
    never a ranked customer). Applied at finalize and when loading cached layers."""
    o = tables.get("orders")
    if o is None:
        return tables
    o = o.copy()
    for c in ("customer_name", "customer_id"):
        if c in o:
            walk = o[c].map(lambda v: v is not None and not (isinstance(v, float) and np.isnan(v))
                            and (fold_text(str(v)) or "") in WALK_IN)
            o[c] = o[c].astype(object).where(~walk, None)
    who = [c for c in ("customer_name", "customer_id") if c in o]
    if "customer_group" in o and who:
        known = o[who].notna().any(axis=1)
        o["customer_group"] = o["customer_group"].astype(object).where(known, None)
    return {**tables, "orders": o}


def _receipt_total_amount(pu: pd.DataFrame) -> tuple[bool, float, float]:
    """(looks like receipt totals, share of lines where qty*unit_cost == amount, share of single-amount receipts)."""
    if not {"qty", "unit_cost", "amount", "receipt_id"} <= set(pu.columns):
        return False, 1.0, 0.0
    lines = pu.dropna(subset=["qty", "unit_cost", "amount"])
    if len(lines) < 20:
        return False, 1.0, 0.0
    ok = float(np.isclose(lines["qty"] * lines["unit_cost"], lines["amount"], rtol=0.02, atol=1).mean())
    multi = lines.groupby("receipt_id").size() > 1
    repeated = float((lines.groupby("receipt_id")["amount"].nunique()[multi] == 1).mean()) if multi.any() else 0.0
    return ok < 0.5 and repeated > 0.8, ok, repeated


def repair_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Deterministic fixes applied at finalize and when loading cached layers: walk-in markers -> null, purchase
    line amounts rebuilt as qty*unit_cost when the loader stored the receipt total on every line."""
    tables = null_walk_in(tables)
    pu = tables.get("purchases")
    if pu is not None and _receipt_total_amount(pu)[0]:
        pu = pu.copy()
        line = pu["qty"] * pu["unit_cost"]
        pu["amount"] = line.where(line.notna(), pu["amount"])
        tables = {**tables, "purchases": pu}
    return tables


def _name_key(v) -> str | None:
    if not isinstance(v, str):
        return None
    return re.sub(r"\s+", " ", fold_text(v) or "").strip() or None


def _canon_products(out: dict[str, pd.DataFrame], meta: dict) -> None:
    """One spelling per product. Exports mix 'Gạo ST25' / 'Gao ST25' / 'GẠO ST25' / 'Gạo  ST25' for the same item, so
    a filter on one spelling silently loses the other rows. Canonical = the catalog (products.name) spelling, else the
    variant with diacritics, mixed case, single spaces, most rows. product_id is rewritten too when the loader built
    it from the name (id == name); real codes (SP0049) are kept."""
    cols = [(n, "name" if n == "products" else "product_name") for n, df in out.items()
            if df is not None and ("name" if n == "products" else "product_name") in df.columns]
    if not cols:
        return
    count: dict = {}
    catalog: set = set()
    for n, c in cols:
        for v, k in out[n][c].dropna().astype(str).value_counts().items():
            count[v] = count.get(v, 0) + int(k)
            if n == "products":
                catalog.add(v)
    groups: dict = {}
    for v in count:
        k = _name_key(v)
        if k:
            groups.setdefault(k, []).append(v)

    def score(v):
        return (v in catalog, any(ord(ch) > 127 for ch in v), not v.isupper(), "  " not in v and v == v.strip(),
                count[v], v)

    canon = {v: max(vs, key=score) for vs in groups.values() if len(vs) > 1 for v in vs}
    canon = {v: c for v, c in canon.items() if v != c}
    if not canon:
        return
    merged = 0
    for n, df in list(out.items()):
        c = "name" if n == "products" else "product_name"
        hit = df[c].isin(canon.keys()) if c in df else pd.Series(False, index=df.index)
        # ids built from the name ('Gao ST25') are variants too; real codes never equal a product name
        hid = df["product_id"].isin(canon.keys()) if "product_id" in df else pd.Series(False, index=df.index)
        if not (hit.any() or hid.any()):
            continue
        df = df.copy()
        merged += int(hit.sum())
        if hit.any():
            df.loc[hit, c] = df.loc[hit, c].map(canon)
        if hid.any():
            df.loc[hid, "product_id"] = df.loc[hid, "product_id"].map(canon)
        if n == "products":
            df = df.drop_duplicates(subset=["product_id"] if "product_id" in df else [c])
        elif n == "unit_conversions":
            df = df.drop_duplicates()
        out[n] = df.reset_index(drop=True)
    meta["product_names_merged"] = {"variants": len(canon), "rows": merged}


def finalize(tables: dict[str, pd.DataFrame], meta: dict) -> dict[str, pd.DataFrame]:
    for name, df in tables.items():
        if df is not None and df.columns.duplicated().any():
            dups = sorted(set(df.columns[df.columns.duplicated()]))
            raise ValueError(f"table '{name}' has duplicate column names {dups} (a rename collided with an "
                             "existing column); keep only one column per name")
    out ={name: _coerce(df.reset_index(drop=True), name) for name, df in tables.items() if df is not None}
    o = out.get("orders")
    if o is not None:
        if "order_date" not in o and "order_datetime" in o:
            o["order_date"] = o["order_datetime"].dt.normalize()
        if "order_datetime" not in o and "order_date" in o:
            o["order_datetime"] = o["order_date"]
        if "status" in o:
            o["status"] = o["status"].map(norm_text)
        if "status_raw" in o:
            # Keyword-evident labels override the loader's mapping (e.g. 'Đã hủy đơn' -> cancelled).
            hint = o["status_raw"].map(status_hint)
            cur = o["status"] if "status" in o else pd.Series(None, index=o.index, dtype=object)
            fix = hint.notna() & (hint != cur)
            if fix.any():
                meta["status_autofixed"] = {f"{r} -> {h}": int(n) for (r, h), n in
                                            o.loc[fix].groupby(["status_raw", hint[fix]]).size().items()}
                o["status"] = hint.where(hint.notna(), cur)
        if "payment_method" in o:
            o["payment_method"] = o["payment_method"].map(norm_text)
        o = o.drop_duplicates()
        out["orders"] = o
    ln = out.get("order_lines")
    if ln is not None:
        # Rulebook §8: identical rows inside the same order are an export error -> count once.
        cols = [c for c in ln.columns if c != "line_no" or meta.get("line_no_from_file")]
        before = len(ln)
        if meta.get("order_codes_from_file") is not False:  # synthesised ids: identical rows are separate sales
            ln = ln.drop_duplicates(subset=cols)
        meta["dropped_duplicate_lines"] = before - len(ln)
        if "line_no" not in ln or not meta.get("line_no_from_file"):
            ln["line_no"] = ln.groupby("order_id", sort=False).cumcount() + 1
        if "is_fee" not in ln:
            ln["is_fee"] = ln["product_name"].map(lambda v: norm_text(v) in ("phí phục vụ",)).astype("boolean")
        out["order_lines"] = ln.reset_index(drop=True)
    _canon_products(out, meta)
    return _null_strings(repair_tables(out))


_NULL_LITERALS = ("<NA>", "<NaT>", "nan", "NaN", "None", "none", "null", "NULL")


def _null_strings(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Literal 'nan' / '<NA>' text (from astype(str) on nulls) becomes a real null: else notna() keeps walk-in rows."""
    for name, df in tables.items():
        if df is None:
            continue
        for c in df.columns:
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c].dtype):
                bad = df[c].astype(object).map(lambda v: isinstance(v, str) and v.strip() in _NULL_LITERALS)
                if bad.any():
                    df[c] = df[c].astype(object).where(~bad, None)
    return tables


def _pct(x) -> str:
    return f"{x:.1%}"


def _status_mismaps(o: pd.DataFrame) -> list[str]:
    out = []
    pairs = o[["status_raw", "status"]].dropna().drop_duplicates()
    for raw, st in pairs.itertuples(index=False):
        want = status_hint(raw)
        if want and st != want:
            n = int(((o["status_raw"] == raw) & (o["status"] == st)).sum())
            out.append(f"'{raw}' -> '{st}' ({n} orders; expected '{want}')")
    return out


def _synthetic_id_errors(tables: dict, meta: dict) -> list[str]:
    errs = []
    ln = tables.get("order_lines")
    if ln is None or not {"order_id", "product_name"} <= set(ln.columns) or not len(ln):
        return errs
    s = ln[["order_id", "product_name"]].dropna().head(5000)
    hit = [fold_text(p) in fold_text(o) for o, p in zip(s["order_id"].astype(str), s["product_name"].astype(str))
           if len(str(p)) > 3]
    if hit and sum(hit) / len(hit) > 0.3:
        errs.append("order_id contains the product name — a synthesised order_id must identify the order (code, or "
                    "date+time+customer, or row number), never the product; set meta['order_codes_from_file']=False")
    dropped = meta.get("dropped_duplicate_lines") or 0
    if dropped > 0.05 * (len(ln) + dropped) and meta.get("order_codes_from_file") is not True:
        errs.append(f"finalize dropped {dropped} identical lines as same-order duplicates ({dropped / (len(ln) + dropped):.0%}) — "
                    "if order_id is synthesised set meta['order_codes_from_file']=False (identical rows are separate "
                    "sales), else make sure order_id is the real order code")
    return errs


def validate(tables: dict[str, pd.DataFrame], meta: dict) -> tuple[list[str], str]:
    errs: list[str] = _synthetic_id_errors(tables, meta)
    rep: list[str] = []
    for name, df in tables.items():
        spec = TABLES.get(name)
        rep.append(f"[{name}] {len(df)} rows; columns: {list(df.columns)}")
        if spec is None:
            if not name.startswith("extra_"):
                errs.append(f"table '{name}' is neither canonical nor 'extra_*': map it to a canonical table "
                            f"(e.g. visits/tickets/bookings -> orders + order_lines) or rename it extra_{name}")
            continue
        unknown = [c for c in df.columns if spec.col(c) is None]
        if unknown:
            errs.append(f"{name}: unknown columns {unknown} (use only schema columns, or return them in an extra table)")
        for c in spec.cols:
            if c.required and c.name not in df.columns:
                errs.append(f"{name}: missing required column '{c.name}'")
            elif c.name in df.columns:
                nn = df[c.name].isna().mean() if len(df) else 0
                if c.required and nn > 0.01:
                    errs.append(f"{name}.{c.name}: {_pct(nn)} null (required)")
                elif nn > 0:
                    rep.append(f"  {name}.{c.name}: {_pct(nn)} null")
        if len(df) == 0:
            errs.append(f"{name}: empty table (omit it instead if the shop has no such data)")

    for name, e in (meta.get("table_errors") or {}).items():
        errs.append(f"{name}: failed to build: {str(e)[:300]}")
    o = tables.get("orders")
    if not tables:
        errs.append("no tables returned")
    elif o is None:
        # Non-sales businesses may legitimately have no orders; they are answered from extra_* tables.
        extras = [n for n in tables if n.startswith("extra_")]
        rep.append("  WARNING: no orders table; extra tables: " + (", ".join(extras) or "none"))
        if not extras:
            errs.append(f"no orders table (only {sorted(tables)}): map the sales/transaction files to orders + "
                        "order_lines; extra_* tables are only for data with no canonical home")
        missing = [n for n in extras if n not in meta.get("extra_desc", {})]
        if missing:
            errs.append(f"meta['extra_desc'] missing description for {missing}")
    else:
        if "order_id" in o and o["order_id"].duplicated().any():
            errs.append(f"orders.order_id not unique ({o['order_id'].duplicated().sum()} duplicates) — one row per order")
        if "order_date" in o and len(o):
            d = o["order_date"].dropna()
            if len(d):
                rep.append(f"  order_date range {d.min().date()} .. {d.max().date()}")
                bad = ((d < "2020-01-01") | (d > "2028-12-31")).mean()
                if bad > 0:
                    errs.append(f"orders.order_date: {_pct(bad)} outside 2020-2028 (wrong parse? day/month swapped?)")
                per_month = d.dt.to_period("M").value_counts().sort_index()
                rep.append("  orders per month: " + ", ".join(f"{k}:{v}" for k, v in per_month.items()))
            if "order_datetime" in o:
                hrs = o["order_datetime"].dt.hour
                rep.append(f"  order hours present: {sorted(hrs.dropna().unique().astype(int).tolist())[:24]}")
        if "status" in o:
            vc = o["status"].value_counts(dropna=False).to_dict()
            rep.append(f"  status: {vc}")
            bad = set(vc) - set(ORDER_STATUS)
            if bad:
                errs.append(f"orders.status has values outside {ORDER_STATUS}: {list(bad)[:10]}")
            if "status_raw" in o and len(vc) == 1 and o["status_raw"].nunique() > 1:
                errs.append(f"orders.status is always '{next(iter(vc))}' but status_raw has "
                            f"{o['status_raw'].value_counts().head(8).to_dict()} — map every raw label")
            if "status_raw" in o:
                bad_map = _status_mismaps(o)
                if bad_map:
                    errs.append("orders.status mapping looks wrong for raw labels " + "; ".join(bad_map) +
                                " — map every distinct status_raw label explicitly")
            if not ({"cancelled", "returned"} & set(vc)):
                rep.append("  WARNING: no cancelled or returned orders at all — verify the status/note mapping")
        if "payment_method" in o:
            vc = o["payment_method"].value_counts(dropna=False).to_dict()
            rep.append(f"  payment_method: {vc}")
            bad = {k for k in vc if k is not None and not pd.isna(k)} - set(PAYMENT_METHOD)
            if bad:
                errs.append(f"orders.payment_method has values outside {PAYMENT_METHOD}: {list(bad)[:10]}")
        for c in ("customer_id", "customer_name"):
            if c in o:
                rep.append(f"  {c}: {o[c].nunique()} distinct, {_pct(o[c].isna().mean())} walk-in/null; "
                           f"top: {o[c].value_counts().head(3).to_dict()}")

    ln = tables.get("order_lines")
    if ln is not None and o is not None and "order_id" in ln and "order_id" in o:
        orphan = ~ln["order_id"].isin(o["order_id"])
        if orphan.mean() > 0:
            errs.append(f"order_lines: {_pct(orphan.mean())} of lines reference an order_id missing from orders")
        no_lines = ~o["order_id"].isin(ln["order_id"])
        if no_lines.mean() > 0.001:
            rep.append(f"  orders without lines: {no_lines.sum()}")
        for c in ("qty", "qty_base", "line_amount"):
            if c in ln:
                s = ln[c]
                rep.append(f"  {c}: min {s.min()}, median {s.median()}, max {s.max()}, negatives {int((s < 0).sum())}")
        if "line_amount" in ln:
            bad = ln["line_amount"].isna() & ~ln.get("is_fee", pd.Series(False, index=ln.index)).fillna(False).astype(bool)
            if bad.any():
                ex = ln.loc[bad, [c for c in ("order_id", "product_name", "qty", "unit_price") if c in ln]].head(3)
                errs.append(f"order_lines.line_amount: {int(bad.sum())} lines are null (money string not parsed? "
                            f"check the raw values of those rows): {ex.to_dict('records')}")
        if {"qty", "unit_price", "line_amount"} <= set(ln.columns):
            disc = ln["line_discount"].fillna(0) if "line_discount" in ln else 0
            exp = ln["qty"] * ln["unit_price"] - disc
            ok = np.isclose(exp, ln["line_amount"], rtol=1e-6, atol=1).mean()
            rep.append(f"  qty*unit_price-line_discount == line_amount for {_pct(ok)} of lines")
            ratio = (ln["line_amount"] / exp.where(exp > 0)).abs()
            off = (ratio > 10) | (ratio < 0.1)
            if off.mean() > 0.0005:
                ex = ln.loc[off, [c for c in ("product_name", "qty", "unit_price", "line_amount") if c in ln]].head(3)
                errs.append(f"order_lines: {int(off.sum())} lines where line_amount is >10x off qty*unit_price "
                            f"(money parsed with the wrong scale? e.g. '1,156tr' = 1.156 triệu): "
                            f"{ex.to_dict('records')}")
        if "line_amount" in ln and len(ln) > 100:
            p99, mx = ln["line_amount"].quantile(0.99), ln["line_amount"].max()
            if p99 > 0 and mx > 100 * p99:
                ex = ln.nlargest(3, "line_amount")[[c for c in ("product_name", "qty", "line_amount") if c in ln]]
                errs.append(f"order_lines.line_amount: max {mx:,.0f} is >100x the 99th percentile {p99:,.0f} "
                            f"(parse/scale error?): {ex.to_dict('records')}")
        if {"qty", "qty_base"} <= set(ln.columns):
            ratio = (ln["qty_base"] / ln["qty"]).replace([np.inf, -np.inf], np.nan)
            rep.append(f"  qty_base/qty factors seen: {sorted(ratio.dropna().round(3).unique().tolist())[:15]}")
        if "goods_total" in o and "line_amount" in ln:
            s = ln.groupby("order_id")["line_amount"].sum()
            m = o.set_index("order_id")["goods_total"].dropna()
            common = m.index.intersection(s.index)
            if len(common):
                ok = np.isclose(s[common], m[common], atol=1).mean()
                rep.append(f"  sum(line_amount) == orders.goods_total for {_pct(ok)} of orders")
                if ok < 0.9:
                    errs.append(f"sum(order_lines.line_amount) matches orders.goods_total for only {_pct(ok)} of orders "
                                "— check line_amount / goods_total mapping (or drop goods_total if it means something else)")
        p = tables.get("products")
        if p is not None and "product_id" in ln and "product_id" in p:
            miss = ~ln["product_id"].dropna().isin(p["product_id"])
            if miss.mean() > 0.01:
                errs.append(f"order_lines.product_id: {_pct(miss.mean())} not found in products.product_id "
                            f"e.g. {ln['product_id'][ln['product_id'].notna()][miss].unique()[:5].tolist()}")
        if "line_amount" in ln and "status" in o:
            j = ln.merge(o[["order_id", "status", "order_date"]], on="order_id", how="left")
            ok = j[j["status"] == "completed"]
            if len(ok):
                by_m = ok.groupby(ok["order_date"].dt.to_period("M"))["line_amount"].sum()
                rep.append("  completed revenue per month: " + ", ".join(f"{k}:{v:,.0f}" for k, v in by_m.items()))
                inner = by_m.iloc[1:-1]  # first/last months may be partial
                if len(inner) >= 3 and inner.min() > 0 and inner.max() / inner.min() > 5:
                    errs.append(f"monthly completed revenue varies {inner.max() / inner.min():.0f}x between full months "
                                f"({inner.idxmin()}: {inner.min():,.0f} vs {inner.idxmax()}: {inner.max():,.0f}) — "
                                "check money parsing, unit scale, missing sheets or duplicated rows")

    pu = tables.get("purchases")
    if pu is not None:
        bad, ok, repeated = _receipt_total_amount(pu)
        rep.append(f"  purchases: qty*unit_cost == amount for {_pct(ok)} of lines")
        if bad:
            errs.append(f"purchases.amount looks like the receipt total repeated on every line (qty*unit_cost "
                        f"matches only {_pct(ok)}; one amount per multi-line receipt in {_pct(repeated)}) — "
                        "amount must be the line amount")
    if pu is not None and {"receipt_date"} <= set(pu.columns):
        d = pu["receipt_date"].dropna()
        if len(d):
            rep.append(f"  purchases date range {d.min().date()} .. {d.max().date()}")
    rep.append(f"meta: {meta}")
    return errs, "\n".join(rep)
