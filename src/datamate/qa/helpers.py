"""Helpers exposed to LLM-generated answer code (see agent.HELPERS_DOC).

All functions are pure and work on the canonical tables `t` (dict name -> DataFrame).
"""
from __future__ import annotations

import difflib
import re

import numpy as np
import pandas as pd

from ..core.parsers import fold_text

VALID_STATUS = ("completed", "delivering")


def valid_orders(t: dict, as_of=None) -> pd.DataFrame:
    """Orders that count for revenue/order counts (not cancelled, not returned), optionally up to as_of."""
    o = t["orders"]
    o = o[o["status"].isin(VALID_STATUS)] if "status" in o else o
    if as_of is not None:
        o = o[o["order_date"] <= pd.Timestamp(as_of)]
    return o


def sales_lines(t: dict, valid_only: bool = True, as_of=None) -> pd.DataFrame:
    """order_lines joined with their order header (order_date, status, customer, ...)."""
    o = valid_orders(t, as_of) if valid_only else t["orders"]
    if as_of is not None and not valid_only:
        o = o[o["order_date"] <= pd.Timestamp(as_of)]
    ln = t["order_lines"]
    hdr = [c for c in o.columns if c not in ln.columns or c == "order_id"]
    return ln.merge(o[hdr], on="order_id", how="inner")


def _bound(x):
    if x is None or (isinstance(x, str) and not x[:1].isdigit()):
        return None
    return pd.Timestamp(x).normalize()


def orders(t: dict, start=None, end=None, as_of=None, valid_only: bool = True) -> pd.DataFrame:
    """Orders with order_date in [start, end] (inclusive, dates), never after as_of.
    valid_only=True keeps orders that count for revenue/order counts (not cancelled, not returned);
    valid_only=False keeps every status (e.g. cancellation rate). start=None: from the start of the data."""
    o = t["orders"]
    if valid_only and "status" in o:
        o = o[o["status"].isin(VALID_STATUS)]
    s, e, a = _bound(start), _bound(end), _bound(as_of)
    e = a if e is None else (min(e, a) if a is not None else e)
    if s is not None:
        o = o[o["order_date"] >= s]
    if e is not None:
        o = o[o["order_date"] <= e]
    return o


def sales(t: dict, start=None, end=None, as_of=None, valid_only: bool = True) -> pd.DataFrame:
    """order_lines of orders(t, start, end, as_of, valid_only), joined with the order header columns."""
    o = orders(t, start, end, as_of, valid_only)
    if "order_lines" not in t:
        # invoice-list shops (one row per order, no item detail): each order is one line worth its goods total
        amt = o["goods_total"] if "goods_total" in o else o.get("total_payable")
        return o.assign(line_amount=amt, line_no=1)
    ln = t["order_lines"]
    hdr = [c for c in o.columns if c not in ln.columns or c == "order_id"]
    return ln.merge(o[hdr], on="order_id", how="inner")


def month_range(d) -> tuple[pd.Timestamp, pd.Timestamp]:
    d = pd.Timestamp(d)
    start = d.replace(day=1)
    return start, start + pd.offsets.MonthEnd(0)


def week_range(d) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Monday..Sunday week containing d."""
    d = pd.Timestamp(d).normalize()
    start = d - pd.Timedelta(days=d.weekday())
    return start, start + pd.Timedelta(days=6)


def tet_window(lunar_year: int, cal: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """23 tháng Chạp of lunar_year-1 .. mùng 10 tháng Giêng of lunar_year (rulebook default)."""
    c = cal[~cal["lunar_leap"].astype(str).str.lower().eq("true")]
    s = c[(c.lunar_year == lunar_year - 1) & (c.lunar_month == 12) & (c.lunar_day == 23)]["date"]
    e = c[(c.lunar_year == lunar_year) & (c.lunar_month == 1) & (c.lunar_day == 10)]["date"]
    return pd.Timestamp(s.iloc[0]), pd.Timestamp(e.iloc[0])


def match(values, query: str, k: int = 5) -> list:
    """Best fuzzy matches of `query` among `values` (diacritics/case-insensitive). Exact folded hits first."""
    if not isinstance(query, str) and (isinstance(values, str) or (not isinstance(values, pd.Series)
                                                                     and len(list(values)) == 1)):
        values, query = query, values  # tolerate match([name], choices) / match(name, choices)
    if not isinstance(query, str):
        query = list(query)[0]
    vals = pd.Series(pd.unique(pd.Series(list(values)).dropna().astype(str)))
    if vals.empty:
        return []
    fq = fold_text(query)
    folded = vals.map(fold_text)
    exact = vals[folded == fq].tolist()
    if exact:
        return exact[:k]  # an exact (folded) hit is the answer; near names would leak into isin() filters
    contains = vals[folded.str.contains(fq, regex=False)].tolist()
    fuzzy = difflib.get_close_matches(fq, folded.tolist(), n=k, cutoff=0.5)
    fuzzy_vals = [vals[folded == f].iloc[0] for f in fuzzy]
    out = []
    for v in exact + sorted(contains, key=len) + fuzzy_vals:
        if v not in out:
            out.append(v)
    return out[:k]


_MEASURE = ("kg", "g", "gram", "ml", "l", "lit", "m", "m2", "m3", "vien", "goi", "chai", "lon", "hop")


def _unit_factors(t: dict) -> tuple[dict, dict, dict]:
    """(factors[key][unit] = base units per unit, base[key] = base unit, name_of[key]) for product ids and names."""
    factors: dict = {}
    base: dict = {}
    name_of: dict = {}
    p = t.get("products")
    alias: dict = {}
    if p is not None and "product_id" in p:
        for pid, nm in zip(p["product_id"], p.get("name", p["product_id"])):
            if pd.notna(pid):
                alias[str(pid)] = str(nm) if pd.notna(nm) else str(pid)
    keys = lambda k: {str(k), alias.get(str(k), str(k))}  # noqa: E731
    if p is not None and "base_unit" in p:
        for _, r in p.iterrows():
            k = r.get("product_id", r.get("name"))
            if pd.notna(k) and pd.notna(r["base_unit"]):
                for kk in keys(k):
                    base[kk] = fold_text(r["base_unit"])
    for _, r in (p.iterrows() if p is not None else []):
        k = r.get("product_id", r.get("name"))
        if pd.notna(k):
            for kk in keys(k):
                name_of[kk] = str(r.get("name", k))
    ln = t.get("order_lines")
    if ln is not None and {"unit", "qty", "qty_base"} <= set(ln.columns):
        key = "product_id" if "product_id" in ln else "product_name"
        same = ln[(ln["qty"] == ln["qty_base"]) & ln["unit"].notna()]
        for k, u in same.groupby(key)["unit"].agg(lambda s: s.mode().iloc[0]).items():
            for kk in keys(k):
                base.setdefault(kk, fold_text(u))
        ratio = ln[ln["unit"].notna() & (ln["qty"] > 0)].assign(f=lambda d: d["qty_base"] / d["qty"])
        for (k, u), f in ratio.groupby([key, "unit"])["f"].median().items():
            for kk in keys(k):
                factors.setdefault(kk, {}).setdefault(fold_text(u), float(f))
    uc = t.get("unit_conversions")
    if uc is not None and {"product_id", "unit", "factor"} <= set(uc.columns):
        for k, u, f in uc[["product_id", "unit", "factor"]].itertuples(index=False):
            if pd.notna(k) and pd.notna(u) and pd.notna(f):
                for kk in keys(k):
                    factors.setdefault(kk, {})[fold_text(u)] = float(f)
    for kk, b in base.items():
        factors.setdefault(kk, {})[b] = 1.0
    return factors, base, name_of


def qty_in(t: dict, df: pd.DataFrame, unit: str, convert: bool = False) -> pd.Series:
    """Quantity of each row of `df` (sales()/order_lines/purchases rows) expressed in `unit` ('lon', 'kg', 'm3'...).

    convert=False (rulebook §3 "bán được bao nhiêu <đơn vị>"): only products whose BASE unit is `unit`, counted as
    qty_base (lines sold in bigger units included); other products -> NaN.
    convert=True: also products that have `unit` as a larger unit (unit_conversions, e.g. lon -> thùng) or a size
    written in the product name ('bao 50kg'); used for prices per unit ("giá nhập một thùng").
    NaN rows are skipped by .sum()."""
    factors, base, name_of = _unit_factors(t)
    ft = fold_text(unit)
    key = "product_id" if "product_id" in df else "product_name"
    out = []
    for r in df.itertuples(index=False):
        k = str(getattr(r, key, None))
        qty = getattr(r, "qty", np.nan)
        fu = fold_text(getattr(r, "unit", None) or "") or base.get(k)
        if fu == ft and (convert or base.get(k) in (None, ft)):
            out.append(qty)
            continue
        f = factors.get(k, {})
        qb = getattr(r, "qty_base", np.nan)
        if qb is None or pd.isna(qb):
            qb = qty * f[fu] if fu in f else (qty if fu and fu == base.get(k) else np.nan)
        tf = 1.0 if ft == base.get(k) else (f.get(ft) if convert else None)
        if tf is None and convert and ft in _MEASURE:
            m = re.search(rf"(\d+(?:[.,]\d+)?)\s*{re.escape(ft)}\b", fold_text(name_of.get(k, k)) or "")
            if m:
                tf = 1 / float(m.group(1).replace(",", "."))
        out.append(qb / tf if tf else np.nan)
    return pd.Series(out, index=df.index, dtype=float)


def unit_factor(t: dict, product, unit: str) -> float | None:
    """Base units in one `unit` of a product (id or name): unit_factor('SP0003', 'hộp') -> 30.0; None if unknown."""
    factors, base, _ = _unit_factors(t)
    f = factors.get(str(product), {})
    u = fold_text(unit)
    if u in f:
        return f[u]
    if fold_text(base.get(str(product))) == u:
        return 1.0
    near = {v for k, v in f.items() if k.split()[0] == u or k.startswith(u + " ")}  # 'cuộn' ~ 'cuộn 10m'
    return near.pop() if len(near) == 1 else None


def _not_cancelled_lines(t: dict) -> pd.DataFrame:
    o = t["orders"]
    o = o[o["status"] != "cancelled"] if "status" in o else o
    return t["order_lines"].merge(o[["order_id", "order_date"]], on="order_id", how="inner")


def stock_at(t: dict, at=None, as_of=None) -> pd.DataFrame:
    """Stock (base unit) of every product at the END of day `at` (default: today) -> DataFrame[product_id,
    product_name, base_unit, qty_base]. Rulebook: everything from the start of data up to that day.
    Sources, first available: stock ledger `stock_moves` (balance after the last movement), snapshot table `stock`
    (rolled back by the sales and receipts after `at` when the snapshot is later), lots (`batches` in - `batch_sales` out + returned), receipts minus
    sales. Sales leaving stock = lines of every order except cancelled ones (a returned order's goods left; only an
    explicit returned quantity comes back)."""
    d = pd.Timestamp(at if at is not None else as_of).normalize()
    end = d + pd.Timedelta(days=1)
    p = t.get("products")
    qty: pd.Series | None = None
    st, sm, ba, bs, pu = (t.get(k) for k in ("stock", "stock_moves", "batches", "batch_sales", "purchases"))

    def _date(df, c):
        return pd.to_datetime(df[c], errors="coerce")

    if sm is not None and {"product_id", "moved_at", "qty_base"} <= set(sm.columns):
        key = "product_id"
        m = sm[_date(sm, "moved_at") < end]
        if "balance_after" in m and m["balance_after"].notna().mean() > 0.9:
            qty = m.sort_values("moved_at", kind="mergesort").groupby(key)["balance_after"].last()
        else:
            qty = m.groupby(key)["qty_base"].sum()
    elif st is not None and "qty_base" in st and st["qty_base"].notna().any():
        key = "product_id" if "product_id" in st and st["product_id"].notna().any() else "product_name"
        snap_day = _date(st, "as_of_date").max() if "as_of_date" in st else pd.NaT
        if pd.isna(snap_day) and "orders" in t:
            snap_day = pd.to_datetime(t["orders"]["order_date"]).max()
        qty = st.groupby(key)["qty_base"].sum()
        if pd.notna(snap_day) and d < snap_day.normalize() and "order_lines" in t:
            ln = _not_cancelled_lines(t)
            later = ln[(_date(ln, "order_date") >= end) & (_date(ln, "order_date") <= snap_day)]
            qty = qty.add(later.groupby(key)["qty_base"].sum(), fill_value=0)
            if pu is not None and {"receipt_date", "qty_base", key} <= set(pu.columns):
                pu2 = pu[pu["status"] != "cancelled"] if "status" in pu else pu
                rec = pu2[(_date(pu2, "receipt_date") >= end) & (_date(pu2, "receipt_date") <= snap_day)]
                qty = qty.sub(rec.groupby(key)["qty_base"].sum(), fill_value=0)
    elif ba is not None and bs is not None and {"product_id", "qty_in_base"} <= set(ba.columns):
        key = "product_id"
        b = ba[_date(ba, "received_date") < end] if "received_date" in ba else ba
        qty = b.groupby(key)["qty_in_base"].sum()
        o = t["orders"]
        o = o[(o["status"] != "cancelled") & (_date(o, "order_date") < end)] if "status" in o else \
            o[_date(o, "order_date") < end]
        x = bs[bs["order_id"].isin(o["order_id"])]
        if "product_id" not in x and "batch_id" in x:
            x = x.merge(ba[["batch_id", "product_id"]], on="batch_id", how="left")
        out = x["qty_base"] - x.get("qty_returned_base", pd.Series(0.0, index=x.index)).fillna(0)
        qty = qty.sub(out.groupby(x["product_id"]).sum(), fill_value=0)
    elif pu is not None and "order_lines" in t and {"receipt_date", "qty_base"} <= set(pu.columns):
        key = "product_id" if "product_id" in pu and pu["product_id"].notna().any() else "product_name"
        pu2 = pu[pu["status"] != "cancelled"] if "status" in pu else pu
        qty = pu2[_date(pu2, "receipt_date") < end].groupby(key)["qty_base"].sum()
        ln = _not_cancelled_lines(t)
        qty = qty.sub(ln[_date(ln, "order_date") < end].groupby(key)["qty_base"].sum(), fill_value=0)
    if qty is None:
        raise ValueError("no stock data (no stock snapshot, stock ledger, lots or receipts): unanswerable")
    res = qty.rename("qty_base").reset_index()
    key = res.columns[0]
    if p is not None and "product_id" in p:
        cols = [c for c in ("product_id", "name", "base_unit") if c in p]
        pm = p[cols].drop_duplicates("product_id").rename(columns={"name": "product_name"})
        on = "product_id" if key == "product_id" else "product_name"
        if on in pm:
            res = res.merge(pm, on=on, how="left")
    for c in ("product_id", "product_name", "base_unit"):
        if c not in res:
            res[c] = None
    return res[["product_id", "product_name", "base_unit", "qty_base"]]


def avg_price(t: dict, df: pd.DataFrame, unit: str) -> float | None:
    """Average price per `unit` over the rows of `df`: money / qty_in(df, unit), on convertible rows only.
    Money = line_amount for sales rows, qty*unit_cost for purchase rows (amount may be a receipt total)."""
    q = qty_in(t, df, unit, convert=True)
    if {"qty", "unit_cost"} <= set(df.columns):
        money = (df["qty"] * df["unit_cost"]).fillna(df.get("amount"))
    elif "line_amount" in df:
        money = df["line_amount"]
    else:
        money = df["amount"]
    ok = q.notna() & money.notna() & (q > 0)
    return None if not ok.any() else float(money[ok].sum() / q[ok].sum())


_PHONE = re.compile(r"[\s\-:,(]*(?:\+?84|0)(?:[ .]?\d){8,10}\)?\s*$")


def _accents(v: str) -> int:
    return sum(1 for ch in v if ord(ch) > 127)


def customer_names(t: dict) -> tuple[str, dict]:
    """(key column, {key: display name}) per rulebook §3: by customer code when the shop records codes, else by the
    name on the order where names differing only in spelling (case, diacritics, spaces, trailing phone) are one."""
    o = t["orders"]
    if "customer_id" in o and o["customer_id"].notna().any():
        names: dict = {}
        c = t.get("customers")
        if c is not None and {"customer_id", "name"} <= set(c.columns):
            names = {k: v for k, v in zip(c["customer_id"], c["name"]) if pd.notna(k) and pd.notna(v)}
        if "customer_name" in o:
            top = o.dropna(subset=["customer_id", "customer_name"]).groupby("customer_id")["customer_name"].agg(
                lambda s: s.mode().iloc[0])
            for k, v in top.items():
                names.setdefault(k, v)
        return "customer_id", {k: _PHONE.sub("", str(v)).strip() for k, v in names.items()}
    if "customer_name" not in o:
        return "customer_name", {}
    variants: dict = {}
    for v, n in o["customer_name"].dropna().astype(str).value_counts().items():
        clean = _PHONE.sub("", v).strip()
        variants.setdefault(_name_key(v), []).append((_accents(clean), n, clean))
    return "customer_name", {k: max(vs)[2] for k, vs in variants.items()}


def _name_key(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return re.sub(r"\s+", " ", fold_text(_PHONE.sub("", str(v))) or "").strip() or None


def with_customer(t: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Add `cust_key` (one value per customer per rulebook §3; NaN for walk-in orders) and `cust_name` (display name)
    to rows carrying customer_id / customer_name (orders, sales(), debts()...). Group and rank by cust_key."""
    col, names = customer_names(t)
    df = df.copy()
    if col not in df:
        df["cust_key"], df["cust_name"] = None, None
        return df
    key = df[col] if col == "customer_id" else df[col].map(_name_key)
    df["cust_name"] = key.map(names).where(key.notna(), None)
    if col == "customer_name":
        df["cust_name"] = df["cust_name"].fillna(df[col])
        key = df["cust_name"].where(key.notna(), None)  # name-keyed shop: the key IS the display name
    df["cust_key"] = key
    return df


def top_customers(t: dict, df: pd.DataFrame, value: str = "line_amount", n: int | None = None,
                  ascending: bool = False) -> pd.DataFrame:
    """Rank customers (walk-in excluded) of rows `df` by sum of `value`, or by order count with value='orders'.
    Returns DataFrame[cust_key, cust_name, <value>] sorted best first (ties by cust_key)."""
    d = df if "cust_key" in df else with_customer(t, df)
    d = d[d["cust_key"].notna()]
    agg = ("order_id", "nunique") if value == "orders" else (value, "sum")
    g = d.groupby("cust_key").agg(**{value: agg, "cust_name": ("cust_name", "first")}).reset_index()
    g = g.sort_values([value, "cust_key"], ascending=[ascending, True], kind="mergesort")
    g = g[["cust_key", "cust_name", value]].reset_index(drop=True)
    return g.head(n) if n else g


_CREDIT_METHODS = ("debt", "installment")  # trả góp is a credit order (public rulebook, phone shops)
_MARKETPLACE = re.compile(r"\bsan\b|tmdt|shopee|lazada|tiki|tiktok|sendo")


def _payable(o: pd.DataFrame, t: dict) -> pd.Series:
    """Amount the customer owes for the whole order (rulebook §4: after order discount, incl. shipping)."""
    if "total_payable" in o and o["total_payable"].notna().mean() > 0.5:
        return o["total_payable"]
    if "goods_total" in o:
        goods = o["goods_total"]
    else:
        goods = o["order_id"].map(t["order_lines"].groupby("order_id")["line_amount"].sum())
    zero = pd.Series(0.0, index=o.index)
    return goods.fillna(0) - o.get("order_discount", zero).fillna(0) + o.get("shipping_fee", zero).fillna(0)


def _credit_orders(t: dict) -> pd.DataFrame:
    """Orders that can carry customer debt: not cancelled, not COD, not marketplace."""
    o = t["orders"]
    if "status" in o:
        o = o[o["status"] != "cancelled"]
    if "payment_method" in o:
        o = o[o["payment_method"] != "cod"]
    elif "payment_method_raw" in o:
        o = o[~o["payment_method_raw"].map(lambda v: "cod" in (fold_text(v) or ""))]
    if "channel" in o:
        o = o[~o["channel"].map(lambda v: bool(_MARKETPLACE.search(fold_text(v) or "")))]
    o = o.copy()
    o["payable"] = _payable(o, t)
    return o


def _payments(t: dict) -> pd.DataFrame | None:
    p = t.get("payments")
    if p is None or "paid_datetime" not in p:
        return None
    if "kind" in p:
        p = p[~p["kind"].isin(["refund"])]
    p = p.copy()
    p["paid_date"] = pd.to_datetime(p["paid_datetime"]).dt.normalize()
    return p


def debts(t: dict, at, as_of=None) -> pd.DataFrame:
    """Open debt per order at the end of day `at` (rulebook §4 "đến hết ngày X"): orders dated <= at, not cancelled
    (returned still count), not COD / marketplace; remaining = payable - payments received up to `at`; only > 0.
    Columns: order_id, order_date, customer_id/customer_name (if any), payable, paid, remaining, age_days,
    cust_key, cust_name (see with_customer)."""
    at = _bound(at)
    if as_of is not None:
        at = min(at, _bound(as_of))
    o = _credit_orders(t)
    o = o[o["order_date"] <= at]
    p = _payments(t)
    if p is not None:
        paid = p[p["paid_date"] <= at].groupby("order_id")["amount"].sum()
        o["paid"] = o["order_id"].map(paid).fillna(0)
    elif "debt_remaining" in o:
        o["paid"] = o["payable"] - o["debt_remaining"].fillna(0)
    else:
        o["paid"] = o.get("paid_amount", o["payable"]).fillna(0)
    o["remaining"] = (o["payable"] - o["paid"]).round(0)
    o["age_days"] = (at - o["order_date"]).dt.days
    cols = [c for c in ("order_id", "order_date", "customer_id", "customer_name", "customer_group", "payable",
                        "paid", "remaining", "age_days") if c in o]
    return with_customer(t, o.loc[o["remaining"] > 0.5, cols].reset_index(drop=True))


def debt_flow(t: dict, start, end, as_of=None) -> pd.DataFrame:
    """Per credit ("ghi nợ") order, debt added and repaid within [start, end] (rulebook §4 "nợ thêm / trả nợ"):
    added  = unpaid part of orders dated in the period after same-day payments;
    repaid = payments made AFTER the order day, dated in the period, of any credit order.
    Credit orders: payment_method 'debt' or 'installment' (trả góp) when recorded, else orders not fully paid on their
    order day. Installment payments count as repayments; refunds never do.
    Columns: order_id, order_date, customer_id/customer_name, added, repaid, cust_key, cust_name."""
    s, e = _bound(start), _bound(end)
    if as_of is not None:
        e = min(e, _bound(as_of)) if e is not None else _bound(as_of)
    o = _credit_orders(t)
    p = _payments(t)
    if p is None:
        raise ValueError("no payments table: debt added/repaid by period is not recorded")
    p = p.merge(o[["order_id", "order_date"]], on="order_id")
    same_day = p[p["paid_date"] <= p["order_date"]].groupby("order_id")["amount"].sum()
    o["added"] = (o["payable"] - o["order_id"].map(same_day).fillna(0)).clip(lower=0).round(0)
    credit = o["payment_method"].isin(_CREDIT_METHODS) if "payment_method" in o else None
    if credit is not None and credit.any():
        o = o[credit]
    else:
        o = o[o["added"] > 0.5]
    later = p[(p["paid_date"] > p["order_date"]) & p["order_id"].isin(o["order_id"])]
    if s is not None:
        later = later[later["paid_date"] >= s]
    if e is not None:
        later = later[later["paid_date"] <= e]
    o["repaid"] = o["order_id"].map(later.groupby("order_id")["amount"].sum()).fillna(0)
    in_period = (o["order_date"] >= s if s is not None else True) & (o["order_date"] <= e if e is not None else True)
    o["added"] = o["added"].where(in_period, 0)
    cols = [c for c in ("order_id", "order_date", "customer_id", "customer_name", "added", "repaid") if c in o]
    return with_customer(t, o.loc[(o["added"] > 0) | (o["repaid"] > 0), cols].reset_index(drop=True))


def due_dates(purchase_date, due_day, n) -> list[pd.Timestamp]:
    """Due dates of n monthly instalments: instalment i (1..n) falls on day `due_day` of the i-th month after the
    purchase month (day clipped to the month's last day)."""
    p = pd.Timestamp(purchase_date)
    out = []
    for i in range(1, int(n) + 1):
        first = (p.to_period("M") + i).to_timestamp()
        out.append(first + pd.Timedelta(days=min(int(due_day), first.days_in_month) - 1))
    return out


def pct(part, whole) -> float | None:
    """part/whole as a percentage rounded to 2 decimals."""
    return None if not whole else round(float(part) / float(whole) * 100, 2)
