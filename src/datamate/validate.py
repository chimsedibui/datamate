"""Post-processing and validation of canonical tables produced by a loader.

`finalize(tables, meta)` coerces dtypes, derives helper columns and applies the
file-convention rules that must hold for every shop (so the LLM does not have to
remember them). `validate(tables, meta)` returns (errors, report): errors make the
mapper retry; the report is a human/LLM-readable summary of what was loaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .parsers import norm_text
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
        if "payment_method" in o:
            o["payment_method"] = o["payment_method"].map(norm_text)
        o = o.drop_duplicates()
        out["orders"] = o
    ln = out.get("order_lines")
    if ln is not None:
        # Rulebook §8: identical rows inside the same order are an export error -> count once.
        cols = [c for c in ln.columns if c != "line_no" or meta.get("line_no_from_file")]
        before = len(ln)
        ln = ln.drop_duplicates(subset=cols)
        meta["dropped_duplicate_lines"] = before - len(ln)
        if "line_no" not in ln or not meta.get("line_no_from_file"):
            ln["line_no"] = ln.groupby("order_id", sort=False).cumcount() + 1
        if "is_fee" not in ln:
            ln["is_fee"] = ln["product_name"].map(lambda v: norm_text(v) in ("phí phục vụ",)).astype("boolean")
        out["order_lines"] = ln.reset_index(drop=True)
    return out


def _pct(x) -> str:
    return f"{x:.1%}"


def validate(tables: dict[str, pd.DataFrame], meta: dict) -> tuple[list[str], str]:
    errs: list[str] = []
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

    o = tables.get("orders")
    if not tables:
        errs.append("no tables returned")
    elif o is None:
        # Non-sales businesses may legitimately have no orders; they are answered from extra_* tables.
        extras = [n for n in tables if n.startswith("extra_")]
        rep.append("  WARNING: no orders table; extra tables: " + (", ".join(extras) or "none"))
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
        if {"qty", "unit_price", "line_amount"} <= set(ln.columns):
            disc = ln["line_discount"].fillna(0) if "line_discount" in ln else 0
            exp = ln["qty"] * ln["unit_price"] - disc
            ok = np.isclose(exp, ln["line_amount"], rtol=1e-6, atol=1).mean()
            rep.append(f"  qty*unit_price-line_discount == line_amount for {_pct(ok)} of lines")
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

    pu = tables.get("purchases")
    if pu is not None and {"receipt_date"} <= set(pu.columns):
        d = pu["receipt_date"].dropna()
        if len(d):
            rep.append(f"  purchases date range {d.min().date()} .. {d.max().date()}")
    rep.append(f"meta: {meta}")
    return errs, "\n".join(rep)
