"""Question kinds of generator v2: named metrics from the split's rule set, cost and profit, purchases, stock, lots, price
changes, event tables, catalog attributes, and question shapes the train split lacks.

Each kind computes the answer from the simulated tables AND writes the SQL that answers it on what the pipeline shows the
model: the canonical views (v_sales, v_orders, v_orders_all) and the shop's own tables for everything else. synth/build2.py
runs every SQL on the rendered files and keeps a question only when the SQL result equals the computed answer, so each kept
row has two independent derivations that agree. The reply (notes, TYPE, SQL) follows the format of the pipeline's samples.
"""
import hashlib
from datetime import date, timedelta

import numpy as np
import pandas as pd

from datamate.ingest import norm
from synth.questions import Facts, money, noisy, pct, period

WD = {1: "thứ Hai", 2: "thứ Ba", 3: "thứ Tư", 4: "thứ Năm", 5: "thứ Sáu", 6: "thứ Bảy", 7: "Chủ nhật"}
q = lambda c: '"' + str(c).replace('"', '""') + '"'                                # noqa: E731
lit = lambda s: "'" + str(s).replace("'", "''") + "'"                              # noqa: E731
low = lambda expr: f"lower(trim(CAST({expr} AS VARCHAR)))"                         # noqa: E731
win = lambda col, a, b: f"{col} >= DATE '{a}' AND {col} < DATE '{b + timedelta(days=1)}'"   # noqa: E731
cap = lambda s: s[:1].upper() + s[1:]                                              # noqa: E731
lc = lambda s: s[:1].lower() + s[1:]                                               # noqa: E731
vnd = lambda v: f"{v:,}".replace(",", ".")                                         # noqa: E731


def plain(s: str) -> str:
    """Lower-case, no accents, đ as d: what replace(strip_accents(lower(x)), 'đ', 'd') gives in DuckDB."""
    import unicodedata
    s = s.lower().strip().replace("đ", "d")
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")


def strip_md(text: str) -> str:
    return text.replace("**", "")


class Ctx:
    def __init__(self, shop, meta: dict, rules: dict, rng: np.random.Generator, today: date):
        self.shop, self.meta, self.rules, self.rng, self.today = shop, meta, rules, rng, today
        self.f = Facts(shop, today)
        lines = self.f.lines
        orev = lines.groupby("order_id", as_index=False).agg(revenue=("amount", "sum"), n_lines=("line", "count"))
        self.orders = self.f.orders.merge(orev, on="order_id")                     # valid orders with their revenue
        self.lines = lines
        self.X = meta.get("extra", {})
        self.flags = meta["flags"]
        self.at_end = today == shop.end

    def tab(self, kind: str) -> str:
        return q(norm(self.X[kind]["file"].rsplit(".", 1)[0]))

    def col(self, kind: str, role: str) -> str:
        return q(self.X[kind]["cols"][role])

    def pick(self, items):
        return items[int(self.rng.integers(0, len(items)))]

    def P(self, kinds: list):
        """A period fully inside the shop's data and not after today: (start, end, phrase)."""
        for _ in range(6):
            p = period(self.rng, self.shop, self.today, str(self.pick(kinds)))
            if p and p[0] >= self.shop.start and p[1] <= min(self.shop.end, self.today):
                return p[0], p[1], p[2]
        return None


def top1(s: pd.Series):
    """Index of the single largest value; None on an empty series or a tie at the top (the rulebook excludes ties)."""
    s = s.sort_values(ascending=False, kind="stable")
    if len(s) == 0 or (len(s) > 1 and s.iloc[0] == s.iloc[1]) or s.iloc[0] <= 0:
        return None
    return s.index[0]


def out(kind, text, typ, answer, sql, notes, alias=None):
    return {"kind": kind, "text": text, "type": typ, "answer": answer, "alias": alias or [], "sql": sql, "notes": notes}


def unans(kind, text, why):
    return out(kind, text, "unanswerable", None, "SELECT 1", [why, "Rulebook §7: information the files do not record is unanswerable, not 0."])


def per(o, a, b):
    return o[(o["date"] >= a) & (o["date"] <= b)]


def rule_note(r: dict) -> str:
    return f"Rule: {r['term']} = {strip_md(r['text'])}."


# ---------- named metrics on the canonical views ----------

def k_cust_rank(c: Ctx):
    r = c.rules["cust_rank"]
    p = c.P(["month", "quarter", "ytd", "last_week", "this_month"])
    if not p:
        return None
    a, b, ph = p
    o, l = per(c.orders, a, b).dropna(subset=["ckey"]), per(c.lines, a, b).dropna(subset=["ckey"])
    v, src, having = r["variant"], "v_orders", ""
    if v == "days":
        s, m = o.groupby("ckey")["date"].nunique(), "COUNT(DISTINCT order_date)"
    elif v == "avg":
        g = o.groupby("ckey")["revenue"].agg(["sum", "count"])
        g = g[g["count"] >= r["n"]]
        s, m, having = g["sum"] / g["count"], "SUM(revenue) / COUNT(*)", f" HAVING COUNT(*) >= {r['n']}"
    elif v == "orders":
        s, m = o.groupby("ckey").size(), "COUNT(*)"
    elif v == "max_order":
        s, m = o.groupby("ckey")["revenue"].max(), "MAX(revenue)"
    elif v == "qty":
        s, m, src = l.groupby("ckey")["qty_base"].sum(), "SUM(qty)", "v_sales"
    else:
        s, m, src = l.groupby("ckey")["code"].nunique(), "COUNT(DISTINCT product_name)", "v_sales"
    key = top1(s)
    if key is None:
        return None
    name = o.loc[o["ckey"] == key, "cname"].iloc[0]
    sql = (f"SELECT customer_name FROM (SELECT customer_key, any_value(customer_name) AS customer_name, {m} AS m FROM {src} "
           f"WHERE {win('order_date', a, b)} AND customer_key IS NOT NULL GROUP BY customer_key{having}) ORDER BY m DESC, customer_key LIMIT 1")
    text = c.pick([f"{r['term']} {ph} là ai?", f"Ai là {lc(r['term'])} {ph}?", f"{cap(ph)} {lc(r['term'])} là khách nào?"])
    notes = [rule_note(r), f"{src}: valid orders only, one customer per customer_key, walk-ins (NULL key) left out.", f"Period {ph} = [{a}, {b + timedelta(days=1)})."]
    return out("cust_rank", text, "string", name, sql, notes, [key] if c.flags["customer_code"] else [])


def k_item_rank(c: Ctx):
    r = c.rules["item_rank"]
    p = c.P(["month", "quarter", "last_week", "this_month", "recent"])
    if not p:
        return None
    a, b, ph = p
    l = per(c.lines, a, b)
    v, extra = r["variant"], ""
    if v == "orders":
        s, m = l.groupby("code")["order_id"].nunique(), "COUNT(DISTINCT order_id)"
    elif v == "customers":
        s, m, extra = l.dropna(subset=["ckey"]).groupby("code")["ckey"].nunique(), "COUNT(DISTINCT customer_key)", " AND customer_key IS NOT NULL"
    elif v == "days":
        s, m = l.groupby("code")["date"].nunique(), "COUNT(DISTINCT order_date)"
    else:
        s, m = l.groupby("code")["amount"].sum(), "SUM(amount)"
    key = top1(s)
    if key is None:
        return None
    name = l.loc[l["code"] == key, "name"].iloc[0]
    sql = (f"SELECT product_name FROM (SELECT product_name, {m} AS m FROM v_sales WHERE {win('order_date', a, b)}{extra} "
           "GROUP BY product_name) ORDER BY m DESC, product_name LIMIT 1")
    text = c.pick([f"{r['term']} {ph} là gì?", f"{cap(ph)} {lc(r['term'])} là mặt hàng nào?"])
    notes = [rule_note(r), "v_sales: lines of valid orders; one item per product_name.", f"Period {ph} = [{a}, {b + timedelta(days=1)})."]
    return out("item_rank", text, "string", name, sql, notes, [key])


def k_day_rank(c: Ctx):
    r = c.rules["day_rank"]
    p = c.P(["month", "quarter"])
    if not p:
        return None
    a, b, ph = p
    o, l = per(c.orders, a, b), per(c.lines, a, b)
    v, src, extra = r["variant"], "v_orders", ""
    if v == "orders":
        s, m = o.groupby("date").size(), "COUNT(*)"
    elif v == "customers":
        s, m, extra = o.dropna(subset=["ckey"]).groupby("date")["ckey"].nunique(), "COUNT(DISTINCT customer_key)", " AND customer_key IS NOT NULL"
    elif v == "qty":
        s, m, src = l.groupby("date")["qty_base"].sum(), "SUM(qty)", "v_sales"
    else:
        s, m, src = l.groupby("date")["amount"].sum(), "SUM(amount)", "v_sales"
    key = top1(s)
    if key is None:
        return None
    sql = (f"SELECT strftime(order_date, '%Y-%m-%d') FROM (SELECT order_date, {m} AS m FROM {src} WHERE {win('order_date', a, b)}{extra} "
           "GROUP BY order_date) ORDER BY m DESC, order_date LIMIT 1")
    text = c.pick([f"{r['term']} {ph} là ngày nào?", f"{cap(ph)} ngày nào là {lc(r['term'])}?"])
    return out("day_rank", text, "date", key.isoformat(), sql, [rule_note(r), f"{src}, grouped by order_date.", f"Period {ph} = [{a}, {b + timedelta(days=1)})."])


def k_hour_rank(c: Ctx):
    r = c.rules["hour_rank"]
    p = c.P(["month", "quarter", "last_week"])
    if not p:
        return None
    a, b, ph = p
    text = c.pick([f"{r['term']} {ph} là mấy giờ?", f"{cap(ph)} {lc(r['term'])} là khung giờ nào?"])
    if not c.flags["time"]:
        return unans("hour_rank", text, "order_ts is NULL on every line: this shop records no time of day, and the rule needs the order's hour.")
    o = per(c.orders, a, b)
    hours = o["time"].map(lambda t: t.hour)
    s = o.groupby(hours).size() if r["variant"] == "orders" else o.groupby(hours)["revenue"].sum()
    key = top1(s)
    if key is None:
        return None
    m = "COUNT(*)" if r["variant"] == "orders" else "SUM(revenue)"
    sql = f"SELECT hour(order_ts) FROM v_orders WHERE {win('order_date', a, b)} GROUP BY 1 ORDER BY {m} DESC, 1 LIMIT 1"
    return out("hour_rank", text, "number", int(key), sql, [rule_note(r), "v_orders.order_ts holds the time the order was created; answer the hour h.", f"Period {ph} = [{a}, {b + timedelta(days=1)})."])


def k_staff_rank(c: Ctx):
    r = c.rules["staff_rank"]
    p = c.P(["month", "quarter", "last_week", "ytd"])
    if not p:
        return None
    a, b, ph = p
    text = c.pick([f"{r['term']} {ph} là ai?", f"Ai là {lc(r['term'])} {ph}?"])
    if not c.flags["staff"]:
        return unans("staff_rank", text, "staff is NULL on every line: this shop does not record who sold each order.")
    o = per(c.orders, a, b)
    v, extra = r["variant"], ""
    if v == "days":
        s, m = o.groupby("staff")["date"].nunique(), "COUNT(DISTINCT order_date)"
    elif v == "orders":
        s, m = o.groupby("staff").size(), "COUNT(*)"
    elif v == "avg":
        s, m = o.groupby("staff")["revenue"].mean(), "AVG(revenue)"
    else:
        s, m, extra = o.dropna(subset=["ckey"]).groupby("staff")["ckey"].nunique(), "COUNT(DISTINCT customer_key)", " AND customer_key IS NOT NULL"
    key = top1(s)
    if key is None:
        return None
    sql = f"SELECT staff FROM v_orders WHERE {win('order_date', a, b)}{extra} GROUP BY staff ORDER BY {m} DESC, staff LIMIT 1"
    return out("staff_rank", text, "string", key, sql, [rule_note(r), "v_orders has one row per valid order with its staff.", f"Period {ph} = [{a}, {b + timedelta(days=1)})."])


def k_big_order(c: Ctx):
    r = c.rules["big_order"]
    p = c.P(["month", "quarter", "last_week", "this_month", "recent"])
    if not p:
        return None
    a, b, ph = p
    o = per(c.orders, a, b)
    if o.empty:
        return None
    v = r["variant"]
    mask, cond = (o["revenue"] >= r["x"], f"revenue >= {r['x']}") if v == "amount" else (o["n_lines"] >= r["k"], f"n_lines >= {r['k']}") if v == "lines" \
        else (o["revenue"] > r["x"], f"revenue > {r['x']}")
    w, t = win("order_date", a, b), lc(r["term"])
    form = c.pick(["count", "count", "revenue", "share"])
    notes = [rule_note(r), "v_orders: one row per valid order with revenue and n_lines.", f"Period {ph} = [{a}, {b + timedelta(days=1)})."]
    if form == "count":
        return out("big_order", c.pick([f"{cap(ph)} có bao nhiêu {t}?", f"Số {t} {ph} là bao nhiêu?"]), "number", int(mask.sum()),
                   f"SELECT COUNT(*) FROM v_orders WHERE {w} AND {cond}", notes)
    if form == "revenue":
        return out("big_order", f"Doanh thu từ các {t} {ph} là bao nhiêu?", "number", money(o.loc[mask, "revenue"].sum()),
                   f"SELECT COALESCE(ROUND(SUM(revenue)), 0) FROM v_orders WHERE {w} AND {cond}", notes)
    return out("big_order", f"{r['term']} chiếm bao nhiêu % số đơn {ph}?", "percent", pct(mask.sum() / len(o) * 100),
               f"SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE {cond}) / COUNT(*), 2) FROM v_orders WHERE {w}", notes)


def k_loyal(c: Ctx):
    r = c.rules["loyal"]
    p = c.P(["month", "quarter", "ytd", "recent"])
    if not p:
        return None
    a, b, ph = p
    o = per(c.orders, a, b).dropna(subset=["ckey"])
    v = r["variant"]
    if v == "orders":
        s, cond = o.groupby("ckey").size() >= r["n"], f"COUNT(*) >= {r['n']}"
    elif v == "days":
        s, cond = o.groupby("ckey")["date"].nunique() >= r["n"], f"COUNT(DISTINCT order_date) >= {r['n']}"
    else:
        s, cond = o.groupby("ckey")["revenue"].sum() >= r["x"], f"SUM(revenue) >= {r['x']}"
    keys = set(s[s].index)
    sub = f"SELECT customer_key FROM v_orders WHERE {win('order_date', a, b)} AND customer_key IS NOT NULL GROUP BY customer_key HAVING {cond}"
    t = lc(r["term"])
    notes = [rule_note(r), "v_orders, customers by customer_key, walk-ins (NULL key) are no customers.", f"Period {ph} = [{a}, {b + timedelta(days=1)})."]
    if c.rng.random() < 0.65:
        return out("loyal", c.pick([f"{cap(ph)} có bao nhiêu {t}?", f"Số {t} {ph} là bao nhiêu?"]), "number", len(keys), f"SELECT COUNT(*) FROM ({sub})", notes)
    return out("loyal", f"Doanh thu từ {t} {ph} là bao nhiêu?", "number", money(o.loc[o["ckey"].isin(keys), "revenue"].sum()),
               f"SELECT COALESCE(ROUND(SUM(revenue)), 0) FROM v_orders WHERE {win('order_date', a, b)} AND customer_key IN ({sub})", notes)


def k_weekday_avg(c: Ctx):
    r = c.rules["weekday_avg"]
    p = c.P(["month", "quarter"])
    if not p:
        return None
    a, b, ph = p
    k = int(c.rng.integers(1, 8))
    o = per(c.orders, a, b)
    o = o[o["date"].map(lambda d: d.isoweekday() == k)]
    n_all = sum(1 for i in range((b - a).days + 1) if (a + timedelta(days=i)).isoweekday() == k)
    n = n_all if r["variant"] == "all_days" else o["date"].nunique()
    if n == 0 or o.empty:
        return None
    den = (f"(SELECT COUNT(*) FROM generate_series(DATE '{a}', DATE '{b}', INTERVAL 1 DAY) AS g(d) WHERE isodow(g.d) = {k})"
           if r["variant"] == "all_days" else "COUNT(DISTINCT order_date)")
    w = f"{win('order_date', a, b)} AND isodow(order_date) = {k}"
    notes = [f"Rule: {r['term']} = {strip_md(r['text'])}.", f"{WD[k]} = isodow {k}; v_orders for valid orders.", f"Period {ph} = [{a}, {b + timedelta(days=1)})."]
    if c.rng.random() < 0.6:
        return out("weekday_avg", f"Trung bình mỗi {WD[k]} {ph} bán được bao nhiêu tiền?", "number", money(o["revenue"].sum() / n),
                   f"SELECT ROUND(SUM(revenue) / {den}) FROM v_orders WHERE {w}", notes)
    return out("weekday_avg", f"Trung bình mỗi {WD[k]} {ph} có bao nhiêu đơn?", "number", pct(len(o) / n),
               f"SELECT ROUND(COUNT(*) * 1.0 / {den}, 2) FROM v_orders WHERE {w}", notes)


def k_bad_rate(c: Ctx):
    r = c.rules["bad_rate"]
    p = c.P(["month", "quarter", "last_week"])
    if not p:
        return None
    a, b, ph = p
    text = c.pick([f"{r['term']} {ph} là bao nhiêu %?", f"{cap(ph)} {lc(r['term'])} là bao nhiêu phần trăm?"])
    if not c.flags["status"]:
        return unans("bad_rate", text, "The shop records no order status, so cancelled and returned orders cannot be told apart.")
    o = per(c.f.orders_all, a, b)
    if o.empty:
        return None
    can, ret = (o["status"] == "huỷ").sum(), (o["status"] == "trả hàng").sum()
    v = r["variant"]
    if v == "both":
        val, expr = (can + ret) / len(o), "COUNT(*) FILTER (WHERE is_cancelled OR is_returned) / COUNT(*)"
    elif v == "returned":
        val, expr = ret / len(o), "COUNT(*) FILTER (WHERE is_returned AND NOT is_cancelled) / COUNT(*)"
    else:
        done = len(o) - can - ret
        if done == 0:
            return None
        val, expr = (can + ret) / done, "COUNT(*) FILTER (WHERE is_cancelled OR is_returned) / COUNT(*) FILTER (WHERE is_valid)"
    return out("bad_rate", text, "percent", pct(val * 100), f"SELECT ROUND(100.0 * {expr}, 2) FROM v_orders_all WHERE {win('order_date', a, b)}",
               [rule_note(r), "v_orders_all: every order placed in the period with is_cancelled / is_returned / is_valid.", f"Period {ph} = [{a}, {b + timedelta(days=1)})."])


def k_new_cust(c: Ctx):
    r = c.rules["new_cust"]
    p = c.P(["month", "quarter", "last_week", "this_month"])
    if not p:
        return None
    a, b, ph = p
    first = c.orders.dropna(subset=["ckey"]).groupby("ckey")["date"].min()
    n = int(((first >= a) & (first <= b)).sum())
    sql = ("SELECT COUNT(*) FROM (SELECT customer_key, MIN(order_date) AS d FROM v_orders WHERE customer_key IS NOT NULL GROUP BY customer_key) "
           f"WHERE {win('d', a, b)}")
    text = c.pick([f"{cap(ph)} có bao nhiêu {lc(r['term'])}?", f"Số {lc(r['term'])} {ph} là bao nhiêu?"])
    return out("new_cust", text, "number", n, sql, [rule_note(r), "First valid order of each customer_key over the whole data, then keep those inside the period.",
                                                     f"Period {ph} = [{a}, {b + timedelta(days=1)})."])


# ---------- cost and profit ----------

def k_profit(c: Ctx):
    p = c.P(["month", "quarter", "last_week", "this_month"])
    if not p:
        return None
    a, b, ph = p
    form = c.pick(["gross", "gross", "top", "margin"])
    r = c.rules["margin"]
    text = {"gross": c.pick([f"{cap(ph)} cửa hàng lãi gộp bao nhiêu tiền?", f"Lãi gộp {ph} là bao nhiêu?"]),
            "top": c.pick([f"Hàng nào lời nhất {ph}?", f"{cap(ph)} mặt hàng nào có tổng lãi gộp cao nhất?"]),
            "margin": f"{r['term']} {ph} là bao nhiêu %?"}[form]
    if not c.flags.get("cost"):
        return unans("profit", text, "cost is NULL on every line: the shop writes no cost of goods on its sale lines, and gross profit needs it (rulebook §5).")
    l = per(c.lines, a, b)
    if l.empty:
        return None
    w = win("order_date", a, b)
    base = ["Rulebook §5: gross profit = revenue minus cost of goods sold; the line cost is quantity times the cost per sale unit written on the sale line.",
            "v_sales: amount and cost of every line of valid orders.", f"Period {ph} = [{a}, {b + timedelta(days=1)})."]
    rev, cost = l["amount"].sum(), l["cost"].sum()
    if form == "gross":
        return out("profit", text, "number", money(rev - cost), f"SELECT ROUND(SUM(amount) - SUM(cost)) FROM v_sales WHERE {w}", base)
    if form == "top":
        g = l.groupby("code").agg(p=("amount", "sum"), k=("cost", "sum"))
        key = top1(g["p"] - g["k"])
        if key is None:
            return None
        return out("profit", text, "string", l.loc[l["code"] == key, "name"].iloc[0],
                   f"SELECT product_name FROM v_sales WHERE {w} GROUP BY product_name ORDER BY SUM(amount) - SUM(cost) DESC, product_name LIMIT 1", base, [key])
    den, d = (rev, "SUM(amount)") if r["variant"] == "revenue" else (cost, "SUM(cost)")
    if den == 0:
        return None
    return out("profit", text, "percent", pct((rev - cost) / den * 100), f"SELECT ROUND(100.0 * (SUM(amount) - SUM(cost)) / {d}, 2) FROM v_sales WHERE {w}",
               [rule_note(r)] + base[1:])


# ---------- purchases, stock, lots (the shop's own tables) ----------

def k_purchase(c: Ctx):
    p = c.P(["month", "quarter", "this_month", "ytd"])
    if not p:
        return None
    a, b, ph = p
    r = c.rules["main_supplier"]
    form = c.pick(["total", "total", "receipts", "top_supplier", "main", "avg_price"])
    if not c.flags.get("purchases"):
        text = c.pick([f"Tổng tiền nhập hàng {ph} là bao nhiêu?", f"{cap(ph)} cửa hàng nhập hàng của nhà cung cấp nào nhiều tiền nhất?"])
        return unans("purchase", text, "No table records purchase receipts or suppliers.")
    P = c.shop.extra["purchases"]
    ok = per(P[P["status"] != "huỷ"], a, b)
    if ok.empty:
        return None
    T, col = c.tab("purchases"), lambda role: c.col("purchases", role)             # noqa: E731
    D = f"CAST({col('ngày nhập')} AS DATE)"
    # spellings are compared without accents: the loader may rewrite 'Huỷ' as another spelling of the same word
    live = (f"replace(strip_accents({low(col('trạng thái phiếu nhập'))}), 'đ', 'd') NOT IN "
            f"({', '.join(lit(plain(s)) for s in c.X['purchases']['status']['huỷ'])})")
    w = f"{win(D, a, b)} AND {live}"
    base = [f"Purchases are in {T}: one row per receipt line, receipt date {col('ngày nhập')}; cancelled receipts are left out (rulebook §5).",
            f"Period {ph} = [{a}, {b + timedelta(days=1)})."]
    if form == "total":
        return out("purchase", c.pick([f"Tổng tiền nhập hàng {ph} là bao nhiêu?", f"{cap(ph)} cửa hàng nhập hàng hết bao nhiêu tiền?"]), "number",
                   money(ok["amount"].sum()), f"SELECT ROUND(SUM({col('tiền nhập')})) FROM {T} WHERE {w}", base)
    if form == "receipts":
        return out("purchase", f"{cap(ph)} có bao nhiêu phiếu nhập hàng?", "number", int(ok["receipt_id"].nunique()),
                   f"SELECT COUNT(DISTINCT {col('mã phiếu nhập')}) FROM {T} WHERE {w}", base)
    if form in ("top_supplier", "main"):
        if form == "top_supplier":
            s, m, text, note = ok.groupby("supplier")["amount"].sum(), f"SUM({col('tiền nhập')})", f"{cap(ph)} cửa hàng nhập nhiều tiền hàng nhất từ nhà cung cấp nào?", \
                "Rulebook §5: the supplier bought from most = the one with the highest purchase amount."
        else:
            v = r["variant"]
            s, m = (ok.groupby("supplier")["receipt_id"].nunique(), f"COUNT(DISTINCT {col('mã phiếu nhập')})") if v == "receipts" else \
                (ok.groupby("supplier")["code"].nunique(), f"COUNT(DISTINCT {col('tên hàng')})") if v == "items" else \
                (ok.groupby("supplier")["date"].nunique(), f"COUNT(DISTINCT {D})")
            text, note = c.pick([f"{r['term']} {ph} là ai?", f"{cap(ph)} {lc(r['term'])} là nhà cung cấp nào?"]), rule_note(r)
        key = top1(s)
        if key is None:
            return None
        return out("purchase", text, "string", key, f"SELECT {col('nhà cung cấp')} FROM {T} WHERE {w} GROUP BY 1 ORDER BY {m} DESC, 1 LIMIT 1", [note] + base)
    g = ok.groupby("code").agg(amount=("amount", "sum"), qty=("qty", "sum"), name=("name", "first"), unit=("unit", "first"))
    row = g.iloc[int(c.rng.integers(0, len(g)))]
    text = f"Giá nhập bình quân một {row['unit']} {row['name']} {ph} là bao nhiêu?"
    return out("purchase", text, "number", money(row["amount"] / row["qty"]),
               f"SELECT ROUND(SUM({col('tiền nhập')}) / SUM({col('số lượng nhập')})) FROM {T} WHERE {w} AND {col('tên hàng')} = {lit(row['name'])}",
               ["Rulebook §5: average purchase price = weighted by purchased quantity, per purchase unit, receipts dated in the period, cancelled ones left out."] + base)


def k_stock(c: Ctx):
    if not c.at_end:
        return None
    goods = c.shop.products[c.shop.products["group"] != "Dịch vụ"]
    if len(goods) == 0:
        return None
    form = c.pick(["item", "item", "units", "value", "out"])
    if not c.flags.get("stock"):
        g = goods.iloc[int(c.rng.integers(0, len(goods)))]
        return unans("stock", f"Tồn kho {g['name']} cuối ngày hôm nay là bao nhiêu {g['base_unit']}?", "No table records stock on hand.")
    S = c.shop.extra["stock"]
    T, col = c.tab("stock"), lambda role: c.col("stock", role)                    # noqa: E731
    base = [f"{T} holds the stock at the end of the last day of data, in base units; today is that day."]
    row = S.iloc[int(c.rng.integers(0, len(S)))]
    if form == "item":
        return out("stock", c.pick([f"Tồn kho {row['name']} cuối ngày hôm nay là bao nhiêu {row['base_unit']}?", f"Hết hôm nay {row['name']} còn lại mấy {row['base_unit']} trong kho?"]),
                   "number", int(row["ending"]), f"SELECT {col('tồn cuối kỳ')} FROM {T} WHERE {col('tên hàng')} = {lit(row['name'])}", base)
    if form == "units":
        big = S[S["units"].map(len) > 1]
        if big.empty:
            return None
        row = big.iloc[int(c.rng.integers(0, len(big)))]
        unit, factor = row["units"][int(c.rng.integers(1, len(row["units"])))]
        return out("stock", f"Tồn kho {row['name']} cuối ngày hôm nay được mấy {unit} chẵn và mấy {row['base_unit']} lẻ?", "list",
                   [int(row["ending"] // factor), int(row["ending"] % factor)],
                   f"SELECT CAST(FLOOR({col('tồn cuối kỳ')} / {factor}) AS BIGINT), CAST({col('tồn cuối kỳ')} % {factor} AS BIGINT) FROM {T} WHERE {col('tên hàng')} = {lit(row['name'])}",
                   base + [f"Rulebook §5: stock in a big unit = [whole big units, leftover base units]; 1 {unit} = {factor} {row['base_unit']}."])
    if form == "out":
        return out("stock", "Cuối ngày hôm nay có bao nhiêu mặt hàng hết hàng?", "number", int((S["ending"] == 0).sum()),
                   f"SELECT COUNT(*) FROM {T} WHERE {col('tồn cuối kỳ')} = 0", base + ["Rulebook §5: out of stock = stock 0 at the end of that day."])
    C, cc = c.tab("catalog"), lambda role: c.col("catalog", role)                  # noqa: E731
    val = int((S["ending"] * S["cost"]).sum())
    sql = (f"SELECT ROUND(SUM(s.{col('tồn cuối kỳ')} * k.{cc('giá vốn danh mục')})) FROM {T} s JOIN {C} k ON {low('s.' + col('tên hàng'))} = {low('k.' + cc('tên hàng'))} "
           f"WHERE s.{col('tồn cuối kỳ')} > 0")
    return out("stock", c.pick(["Hết ngày hôm nay, số hàng còn trong kho đáng bao nhiêu tiền, tính theo giá vốn?", "Giá trị hàng tồn kho cuối ngày hôm nay theo giá vốn là bao nhiêu?"]),
               "number", val, sql, base + ["Rulebook §5: stock value = stock of each item times the cost per base unit in the catalog, items in stock only."])


def k_lots(c: Ctx):
    if not c.at_end or not c.flags.get("lots"):
        return None
    L = c.shop.extra["lots"]
    live = L[(L["remaining"] > 0)]
    T, col = c.tab("lots"), lambda role: c.col("lots", role)                      # noqa: E731
    today = c.today
    E = f"CAST({col('hạn sử dụng')} AS DATE)"
    form = c.pick(["nearest", "count", "capital", "items"])
    rule = "Rule: a lot in stock has a remaining quantity above 0; counted at today D."
    if form == "nearest":
        cand = live[live["expiry"] >= today]
        if cand.empty:
            return None
        name = str(c.pick(sorted(cand["name"].unique())))
        d = cand.loc[cand["name"] == name, "expiry"].min()
        return out("lots", c.pick([f"Lô {name} gần hết hạn nhất có hạn dùng ngày nào?", f"Theo tồn kho hôm nay, lô {name} còn hàng gần hết hạn nhất hết hạn ngày nào?"]),
                   "date", d.isoformat(),
                   f"SELECT strftime(MIN({E}), '%Y-%m-%d') FROM {T} WHERE {col('tên hàng')} = {lit(name)} AND {col('tồn lô')} > 0 AND {E} >= DATE '{today}'",
                   [rule, "Nearest-expiry lot = among lots in stock and not yet expired at D, the earliest expiry."])
    n = int(c.pick([30, 60, 90, 120]))
    end = today + timedelta(days=n)
    hit = live[(live["expiry"] >= today) & (live["expiry"] <= end)]
    w = f"{col('tồn lô')} > 0 AND {E} >= DATE '{today}' AND {E} <= DATE '{end}'"
    span = f"từ {today:%d/%m/%Y} đến hết {end:%d/%m/%Y}"
    notes = [rule, f"Expiring within {n} days = expiry from D = {today} to the end of D + {n} = {end}."]
    if form == "count":
        return out("lots", f"Có mấy lô hàng còn tồn sẽ hết hạn trong {n} ngày tới, {span}?", "number", len(hit), f"SELECT COUNT(*) FROM {T} WHERE {w}", notes)
    if form == "capital":
        val = money(float((hit["remaining"] * hit["unit_cost_base"].round(2)).sum()))
        return out("lots", f"Tiền vốn đang nằm ở các lô còn tồn có hạn dùng {span} là bao nhiêu?", "number", val,
                   f"SELECT COALESCE(ROUND(SUM({col('tồn lô')} * {col('giá nhập cơ bản')})), 0) FROM {T} WHERE {w}",
                   notes + ["Capital in a lot = remaining quantity times that lot's own purchase price per base unit."])
    g = hit.groupby("name")["expiry"].min().sort_values(kind="stable")
    if not 1 <= len(g) <= 5 or g.duplicated().any():
        return None
    return out("lots", f"Những mặt hàng nào còn tồn có lô hết hạn trong {n} ngày tới, {span}?", "list", list(g.index),
               f"SELECT {col('tên hàng')} FROM {T} WHERE {w} GROUP BY 1 ORDER BY MIN({E}), 1", notes + ["List by earliest expiry first (rulebook §9)."])


def k_price_effect(c: Ctx):
    P = c.shop.extra["prices"]
    if P.empty:
        return None
    n = int(c.pick([7, 14, 30]))
    limit = min(c.shop.end, c.today)
    cand = P[[(d - timedelta(days=n) >= c.shop.start) and (d + timedelta(days=n - 1) <= limit) for d in P["date"]]]
    if cand.empty:
        return None
    row = cand.iloc[int(c.rng.integers(0, len(cand)))]
    d = row["date"]
    l = c.lines[c.lines["code"] == row["code"]]
    after, before = per(l, d, d + timedelta(days=n - 1))["qty_base"].sum(), per(l, d - timedelta(days=n), d - timedelta(days=1))["qty_base"].sum()
    if before == 0:
        return None
    unit = c.shop.products.loc[c.shop.products["code"] == row["code"], "base_unit"].iloc[0]
    text = c.pick([f"Ngày {d:%d/%m/%Y} {row['name']} tăng giá từ {vnd(row['old'])}đ lên {vnd(row['new'])}đ. Số {unit} bán được trong {n} ngày kể từ ngày đó so với {n} ngày ngay trước thay đổi bao nhiêu %?",
                   f"{row['name']} lên giá {vnd(row['new'])}đ từ ngày {d:%d/%m/%Y}; so {n} ngày kể từ ngày tăng giá với {n} ngày liền trước, lượng bán tăng hay giảm bao nhiêu %?"])
    name = lit(row["name"])
    sql = (f"SELECT ROUND(100.0 * (a.q - b.q) / b.q, 2) FROM (SELECT COALESCE(SUM(qty), 0) AS q FROM v_sales WHERE product_name = {name} AND {win('order_date', d, d + timedelta(days=n - 1))}) a, "
           f"(SELECT SUM(qty) AS q FROM v_sales WHERE product_name = {name} AND {win('order_date', d - timedelta(days=n), d - timedelta(days=1))}) b")
    return out("price_effect", text, "percent", pct((after - before) / before * 100), sql,
               [f"Rulebook §5: compare the base-unit quantity of the {n} days from D = {d} (D included) with the {n} days right before D; both without cancelled and returned orders.",
                "v_sales.qty is already in base units.", f"Windows: [{d}, {d + timedelta(days=n)}) and [{d - timedelta(days=n)}, {d})."])


# ---------- event tables ----------

def k_event(c: Ctx):
    if not c.flags.get("events"):
        return None
    flavor = c.X["events"]
    E = c.shop.extra["events"]["df"]
    T, col = c.tab(flavor), lambda role: c.col(flavor, role)                      # noqa: E731
    word = "phiếu" if flavor == "ticket" else "lượt hẹn"
    p = c.P(["month", "quarter", "last_week", "this_month", "recent"])
    if not p:
        return None
    a, b, ph = p
    e = per(E, a, b)
    if e.empty:
        return None
    rc = c.rules["ev_count"]
    paid = rc["variant"] == "paid"
    counted = e[e["amount"] > 0] if paid else e
    D = f"CAST({col('ngày nhận')} AS DATE)"
    w = win(D, a, b)
    wc = w + (f" AND {col('tiền thu')} > 0" if paid else "")
    base = [f"Rule: {word} = {strip_md(rc['text'])}; a row belongs to the period by its {col('ngày nhận')}.", f"Table {T}: one row per {word}.",
            f"Period {ph} = [{a}, {b + timedelta(days=1)})."]
    form = c.pick(["count", "count", "entities", "amount", "staff", "free_rate", "repeat", "weekday"] + (["duration"] if flavor == "ticket" else []))
    if form == "count":
        return out("event", c.pick([f"{cap(ph)} có bao nhiêu {word}?", f"Đếm giúp số {word} {ph}."]), "number", len(counted), f"SELECT COUNT(*) FROM {T} WHERE {wc}", base)
    if form == "entities":
        return out("event", f"{cap(ph)} có bao nhiêu khách khác nhau có {word}?", "number", int(counted["entity"].nunique()),
                   f"SELECT COUNT(DISTINCT {col('khách')}) FROM {T} WHERE {wc}", base + ["One customer = one phone number."])
    if form == "amount":
        return out("event", f"{cap(ph)} thu được bao nhiêu tiền từ các {word}?", "number", int(e["amount"].sum()), f"SELECT ROUND(SUM({col('tiền thu')})) FROM {T} WHERE {w}", base[1:])
    if form == "staff":
        r = c.rules["ev_staff"]
        s, m, cond = (counted.groupby("staff").size(), "COUNT(*)", wc) if r["variant"] == "count" else (e.groupby("staff")["amount"].sum(), f"SUM({col('tiền thu')})", w)
        key = top1(s)
        if key is None:
            return None
        return out("event", c.pick([f"{r['term']} {ph} là ai?", f"Ai là {lc(r['term'])} {ph}?"]), "string", key,
                   f"SELECT {col('người làm')} FROM {T} WHERE {cond} GROUP BY 1 ORDER BY {m} DESC, 1 LIMIT 1", [rule_note(r)] + base)
    if form == "free_rate":
        r = c.rules["ev_free_rate"]
        return out("event", f"{r['term']} {ph} là bao nhiêu %?", "percent", pct((e["amount"] == 0).sum() / len(e) * 100),
                   f"SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE {col('tiền thu')} = 0) / COUNT(*), 2) FROM {T} WHERE {w}", [rule_note(r)] + base[1:])
    if form == "repeat":
        r = c.rules["ev_repeat"]
        n = int((counted.groupby("entity").size() >= r["n"]).sum())
        return out("event", c.pick([f"{cap(ph)} có bao nhiêu {lc(r['term'])}?", f"Số {lc(r['term'])} {ph} là bao nhiêu?"]), "number", n,
                   f"SELECT COUNT(*) FROM (SELECT {col('khách')} FROM {T} WHERE {wc} GROUP BY 1 HAVING COUNT(*) >= {r['n']})", [rule_note(r)] + base)
    if form == "duration":
        done = e.dropna(subset=["date2"])
        if done.empty:
            return None
        days = [(d2 - d1).days for d1, d2 in zip(done["date"], done["date2"])]
        return out("event", f"Thời gian xử lý trung bình của các phiếu nhận {ph} là bao nhiêu ngày?", "number", pct(sum(days) / len(days)),
                   f"SELECT ROUND(AVG(date_diff('day', {D}, CAST({col('ngày trả')} AS DATE))), 2) FROM {T} WHERE {w} AND {col('ngày trả')} IS NOT NULL",
                   ["Rule: processing time = days from the received date to the returned date, tickets with a return date only, by received date; average to 2 decimals."] + base[1:])
    rw = c.rules["weekday_avg"]
    k = int(c.rng.integers(1, 8))
    on = counted[counted["date"].map(lambda d: d.isoweekday() == k)]
    n_all = sum(1 for i in range((b - a).days + 1) if (a + timedelta(days=i)).isoweekday() == k)
    n = n_all if rw["variant"] == "all_days" else on["date"].nunique()
    if n == 0 or on.empty:
        return None
    den = (f"(SELECT COUNT(*) FROM generate_series(DATE '{a}', DATE '{b}', INTERVAL 1 DAY) AS g(d) WHERE isodow(g.d) = {k})"
           if rw["variant"] == "all_days" else f"COUNT(DISTINCT {D})")
    return out("event", f"Trung bình mỗi {WD[k]} {ph} có bao nhiêu {word}?", "number", pct(len(on) / n),
               f"SELECT ROUND(COUNT(*) * 1.0 / {den}, 2) FROM {T} WHERE {wc} AND isodow({D}) = {k}", [f"Rule: {rw['term']} = {strip_md(rw['text'])}."] + base)


# ---------- side books: orders paid in periods, amounts tied to orders ----------

def k_plan(c: Ctx):
    if not c.flags.get("plans"):
        return None
    pl = c.shop.extra["plans"]
    P, S, plan, pw, layout = pl["plans"], pl["sched"], pl["plan"], pl["p"], c.X["plans"]["layout"]
    T, col = c.tab("plans"), lambda role: c.col("plans", role)                    # noqa: E731
    kind2 = "plan_pay" if layout == "two" else "plan_sched"
    T2, c2 = c.tab(kind2), lambda role: c.col(kind2, role)                        # noqa: E731
    pid, pid2 = col("mã trả dần"), c2("mã trả dần (sổ thu)")
    D, PD, owed = c.today, f"CAST({col('ngày mua')} AS DATE)", f"({col('giá trị đơn')} - {col('trả trước')})"
    if layout == "two":      # every row of the second book is one payment
        paid_d, paid_amt, is_paid = f"CAST({c2('ngày thu')} AS DATE)", c2("tiền thu kỳ"), ""
        due_d = f"CAST(date_trunc('month', CAST(p.{col('ngày mua')} AS DATE)) + to_months(CAST(g.k AS INTEGER)) + to_days(CAST(p.{col('ngày hạn')} AS INTEGER) - 1) AS DATE)"
        sched = (f"SELECT p.{pid} AS pid, {due_d} AS due_date, p.{col('tiền mỗi kỳ')} AS amount FROM {T} p, generate_series(1, CAST(p.{col('số kỳ')} AS BIGINT)) AS g(k)")
        how = (f"{T}: one row per {plan} order; period k is due on day {col('ngày hạn')} of the k-th month after the purchase month, {col('tiền mỗi kỳ')} each. "
               f"{T2}: one row per payment.")
    else:                    # the second book has one row per period, with the day it was paid
        paid_d, paid_amt, is_paid = f"CAST({c2('ngày trả kỳ')} AS DATE)", c2("tiền kỳ"), f"{c2('ngày trả kỳ')} IS NOT NULL AND "
        how = f"{T}: one row per {plan} order. {T2}: one row per period with its due date; {c2('ngày trả kỳ')} is empty while unpaid."
    paid_to = f"SELECT {pid2} AS pid, SUM({paid_amt}) AS paid FROM {T2} WHERE {is_paid}{paid_d} <= DATE '{D}' GROUP BY 1"
    asof = f"Today D = {D}: only {plan} orders bought and payments made up to D count."
    Pd = P[P["date"] <= D]
    Sd = S[S["pid"].isin(Pd["pid"])]
    paid_now = Sd[Sd["paid"].notna() & (Sd["paid"] <= D)].groupby("pid")["amount"].sum()
    bal = (Pd.set_index("pid")["total"] - Pd.set_index("pid")["down"]).sub(paid_now, fill_value=0)
    today = c.pick(["Tính tới hôm nay", "Đến hết hôm nay", "Tại hôm nay"])
    form = c.pick(["opened", "financed", "collected", "collected", "balance", "balance", "overdue", "overdue", "overdue", "overdue_n", "top_debtor", "share", "due_in",
                   "done"] + (["late", "late"] if layout == "sched" else []))
    if form in ("opened", "financed", "collected", "share", "due_in", "late"):
        p = c.P(["month", "quarter", "this_month", "recent", "month"])
        if not p:
            return None
        a, b, ph = p
        per_note = f"Period {ph} = [{a}, {b + timedelta(days=1)})."
        if form == "opened":
            n = len(per(P, a, b))
            return out("plan", c.pick([f"{cap(ph)} có bao nhiêu đơn {plan}?", f"Đếm giúp số đơn {plan} mua {ph}."]), "number", n,
                       f"SELECT COUNT(*) FROM {T} WHERE {win(PD, a, b)}", [f"Rule: a {plan} order belongs to the period by its purchase date.", how, per_note])
        if form == "financed":
            e = per(P, a, b)
            if e.empty:
                return None
            return out("plan", f"Các đơn {plan} mua {ph} còn phải trả sau lúc mua tổng cộng bao nhiêu tiền (giá trị đơn trừ trả trước)?", "number",
                       int((e["total"] - e["down"]).sum()), f"SELECT ROUND(SUM({owed})) FROM {T} WHERE {win(PD, a, b)}",
                       [f"Rule: a {plan} order belongs to the period by its purchase date.", how, per_note])
        if form == "collected":
            e = S[S["paid"].notna() & (S["paid"] >= a) & (S["paid"] <= b)]
            if e.empty:
                return None
            return out("plan", c.pick([f"{cap(ph)} thu được bao nhiêu tiền {pw} của các đơn {plan}?", f"Tiền {pw} {plan} thu về {ph} là bao nhiêu?"]), "number",
                       int(e["amount"].sum()), f"SELECT ROUND(SUM({paid_amt})) FROM {T2} WHERE {is_paid}{win(paid_d, a, b)}",
                       [f"Rule: money collected for a period belongs to the report period by the day it was paid.", how, per_note])
        if form == "share":
            o = per(c.orders, a, b)
            if o.empty or not o["revenue"].sum():
                return None
            hit = o[o["order_id"].isin(P["oid"])]["revenue"].sum()
            return out("plan", f"Các đơn {plan} chiếm bao nhiêu % doanh thu {ph}?", "percent", pct(hit / o["revenue"].sum() * 100),
                       f"SELECT ROUND(100.0 * COALESCE(SUM(revenue) FILTER (WHERE order_id IN (SELECT CAST({col('mã đơn bán')} AS VARCHAR) FROM {T})), 0) / SUM(revenue), 2) "
                       f"FROM v_orders WHERE {win('order_date', a, b)}",
                       [f"{T} names the sales order of each {plan} order in {col('mã đơn bán')}; the sales book keeps the full amount of these orders.",
                        "Revenue = valid orders (v_orders), by order date.", per_note])
        if form == "due_in":
            e = S[(S["due"] >= a) & (S["due"] <= b)]
            if e.empty:
                return None
            sql = (f"SELECT ROUND(SUM({c2('tiền kỳ')})) FROM {T2} WHERE {win('CAST(' + c2('hạn trả') + ' AS DATE)', a, b)}" if layout == "sched" else
                   f"WITH s AS ({sched}) SELECT ROUND(SUM(amount)) FROM s WHERE {win('due_date', a, b)}")
            return out("plan", f"{cap(ph)} có bao nhiêu tiền {pw} {plan} đến hạn?", "number", int(e["amount"].sum()), sql,
                       [f"Rule: a period falls in the report period by its due date, paid or not.", how, per_note])
        r = c.rules["pl_late"]
        e = S[(S["due"] >= a) & (S["due"] <= b)]
        due_c, paid_c = f"CAST({c2('hạn trả')} AS DATE)", c2("ngày trả kỳ")
        if r["variant"] == "after":
            e = e[e["paid"].notna() & (e["paid"] <= D)]
            if e.empty:
                return None
            ans = pct((e["paid"] > e["due"]).sum() / len(e) * 100)
            sql = (f"SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE {paid_d} > {due_c}) / COUNT(*), 2) FROM {T2} "
                   f"WHERE {win(due_c, a, b)} AND {paid_c} IS NOT NULL AND {paid_d} <= DATE '{D}'")
        else:
            if e.empty:
                return None
            ans = pct((e["paid"].isna() | (e["paid"] > e["due"])).sum() / len(e) * 100)
            sql = f"SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE {paid_c} IS NULL OR {paid_d} > {due_c}) / COUNT(*), 2) FROM {T2} WHERE {win(due_c, a, b)}"
        return out("plan", c.pick([f"{r['term']} {ph} là bao nhiêu %?", f"{cap(ph)} {lc(r['term'])} là bao nhiêu phần trăm?"]), "percent", ans, sql,
                   [rule_note(r), how, per_note + f" Today D = {D}."])
    if Pd.empty:
        return None
    if form == "balance":
        return out("plan", c.pick([f"{today}, dư nợ {plan} còn bao nhiêu?", f"{today} khách {plan} còn nợ tổng cộng bao nhiêu tiền?"]), "number", int(bal.sum()),
                   f"SELECT ROUND((SELECT SUM({owed}) FROM {T} WHERE {PD} <= DATE '{D}') - COALESCE((SELECT SUM({paid_amt}) FROM {T2} WHERE {is_paid}{paid_d} <= DATE '{D}'), 0))",
                   [f"Rule: {plan} balance at D = order value - down payment - payments made up to D, over the {plan} orders bought up to D.", how, asof])
    if form in ("top_debtor", "done"):
        j = f"WITH paid AS ({paid_to}) SELECT {{}} FROM {T} p LEFT JOIN paid x ON x.pid = p.{pid} WHERE CAST(p.{col('ngày mua')} AS DATE) <= DATE '{D}'"
        rest = f"p.{col('giá trị đơn')} - p.{col('trả trước')} - COALESCE(x.paid, 0)"
        if form == "done":
            return out("plan", f"{today} có bao nhiêu đơn {plan} đã trả xong?", "number", int((bal <= 0).sum()), j.format("COUNT(*)") + f" AND {rest} <= 0",
                       [f"Rule: a {plan} order is paid off at D when order value - down payment - payments made up to D is 0.", how, asof])
        by = bal.groupby(Pd.set_index("pid")["cname"]).sum()
        key = top1(by)
        if key is None:
            return None
        return out("plan", c.pick([f"{today}, khách nào còn dư nợ {plan} lớn nhất?", f"{today} ai đang nợ {plan} nhiều nhất?"]), "string", key,
                   j.format(f"p.{col('khách trả dần')}") + f" GROUP BY 1 ORDER BY SUM({rest}) DESC, 1 LIMIT 1",
                   [f"Rule: {plan} balance at D = order value - down payment - payments made up to D; one customer can have several {plan} orders.", how, asof])
    r = c.rules["pl_overdue"]
    cut = {"strict": D, "incl": D + timedelta(days=1), "grace": D - timedelta(days=r["n"])}[r["variant"]]
    due_now = Sd[Sd["due"] < cut].groupby("pid")["amount"].sum()
    over = due_now.sub(paid_now, fill_value=0).reindex(due_now.index).clip(lower=0)
    if layout == "sched":
        cond = f"CAST({c2('hạn trả')} AS DATE) < DATE '{cut}' AND ({c2('ngày trả kỳ')} IS NULL OR {paid_d} > DATE '{D}')"
        live = f" AND {pid2} IN (SELECT {pid} FROM {T} WHERE {PD} <= DATE '{D}')"
        sql_amt = f"SELECT ROUND(COALESCE(SUM({c2('tiền kỳ')}), 0)) FROM {T2} WHERE {cond}{live}"
        sql_n = f"SELECT COUNT(DISTINCT {pid2}) FROM {T2} WHERE {cond}{live}"
    else:
        base = (f"WITH s AS ({sched}), due AS (SELECT pid, SUM(amount) AS due FROM s WHERE due_date < DATE '{cut}' GROUP BY 1), paid AS ({paid_to}), "
                f"o AS (SELECT d.pid, greatest(d.due - COALESCE(x.paid, 0), 0) AS over FROM due d LEFT JOIN paid x ON x.pid = d.pid) ")
        sql_amt, sql_n = base + "SELECT ROUND(COALESCE(SUM(over), 0)) FROM o", base + "SELECT COUNT(*) FROM o WHERE over > 0"
    notes = [rule_note(r), how, f"Today D = {D}; a period is unpaid at D when it has no payment up to D."]
    if form == "overdue_n":
        return out("plan", f"{today} có bao nhiêu đơn {plan} đang có {lc(r['term'])}?", "number", int((over > 0).sum()), sql_n, notes)
    if not over.sum():
        return None
    return out("plan", c.pick([f"{today}, {lc(r['term'])} là bao nhiêu?", f"{r['term']} {lc(today)} là bao nhiêu tiền?"]), "number", int(over.sum()), sql_amt, notes)


def k_link(c: Ctx):
    if not c.flags.get("links"):
        return None
    lk = c.shop.extra["links"]
    E, w, book, cat_name = lk["df"], lk["w"], lk["book"], lk["cat_name"]
    T, col = c.tab("links"), lambda role: c.col("links", role)                    # noqa: E731
    rp, rv = c.rules["lk_period"], c.rules["lk_valid"]
    p = c.P(["month", "quarter", "this_month", "last_week", "recent", "month"])
    if not p:
        return None
    a, b, ph = p
    valid_ids = set(c.orders["order_id"])
    e = E[E["oid"].isin(valid_ids)] if rv["variant"] == "valid" else E
    e = per(e, a, b) if rp["variant"] == "book" else e[(e["odate"] >= a) & (e["odate"] <= b)]
    on = f"o.order_id = CAST(t.{col('mã đơn nối')} AS VARCHAR)"
    book_d = f"CAST(t.{col('ngày ghi')} AS DATE)"
    if rp["variant"] == "book":
        src = f"{T} t" + (f" JOIN v_orders o ON {on}" if rv["variant"] == "valid" else "") + f" WHERE {win(book_d, a, b)}"
    else:
        src = f"{T} t JOIN {'v_orders' if rv['variant'] == 'valid' else 'v_orders_all'} o ON {on} WHERE {win('o.order_date', a, b)}"
    amt, cat = f"t.{col('số tiền')}", f"t.{col('phân loại sổ phụ')}"
    notes = [rule_note(rp), rule_note(rv), f"{T} ({book}): one row per amount, tied to the sales order named in {col('mã đơn nối')}.",
             f"Period {ph} = [{a}, {b + timedelta(days=1)})."]
    per_note = notes[-1]
    form = c.pick(["total", "total", "count", "orders", "avg", "top_cat", "cat_total", "rate", "rate", "net", "net"])
    if form in ("rate", "net"):
        o = per(c.orders, a, b)
        if o.empty or not o["revenue"].sum():
            return None
        ids = f"SELECT CAST({col('mã đơn nối')} AS VARCHAR) FROM {T}"
        if form == "rate":
            r = c.rules["lk_rate"]
            hit = o[o["order_id"].isin(E["oid"])]
            if r["variant"] == "orders":
                ans, sql = pct(len(hit) / len(o) * 100), f"SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE order_id IN ({ids})) / COUNT(*), 2) FROM v_orders WHERE {win('order_date', a, b)}"
            else:
                ans = pct(hit["revenue"].sum() / o["revenue"].sum() * 100)
                sql = f"SELECT ROUND(100.0 * COALESCE(SUM(revenue) FILTER (WHERE order_id IN ({ids})), 0) / SUM(revenue), 2) FROM v_orders WHERE {win('order_date', a, b)}"
            return out("link", c.pick([f"{r['term']} {ph} là bao nhiêu %?", f"{cap(ph)} {lc(r['term'])} là bao nhiêu phần trăm?"]), "percent", ans, sql,
                       [rule_note(r), notes[2], "Valid orders of the period = v_orders by order date.", per_note])
        r = c.rules["lk_net"]
        x = E[E["oid"].isin(o["order_id"])]["amount"].sum()
        return out("link", c.pick([f"{r['term']} {ph} là bao nhiêu?", f"{cap(ph)} {lc(r['term'])} là bao nhiêu tiền?"]), "number", int(o["revenue"].sum() - x),
                   f"SELECT ROUND((SELECT SUM(revenue) FROM v_orders WHERE {win('order_date', a, b)}) - COALESCE((SELECT SUM({amt}) FROM {T} t JOIN v_orders o ON {on} "
                   f"WHERE {win('o.order_date', a, b)}), 0))", [rule_note(r), notes[2], per_note])
    if e.empty:
        return None
    if form == "total":
        return out("link", c.pick([f"{cap(ph)} tổng tiền {w} là bao nhiêu?", f"Tổng {w} {ph} là bao nhiêu tiền?"]), "number", int(e["amount"].sum()),
                   f"SELECT ROUND(SUM({amt})) FROM {src}", notes)
    if form == "count":
        return out("link", f"{cap(ph)} có bao nhiêu khoản {w}?", "number", len(e), f"SELECT COUNT(*) FROM {src}", notes + [f"One row of {T} = one amount."])
    if form == "orders":
        return out("link", f"{cap(ph)} có bao nhiêu đơn có {w}?", "number", int(e["oid"].nunique()), f"SELECT COUNT(DISTINCT t.{col('mã đơn nối')}) FROM {src}",
                   notes + ["An order with two rows counts once."])
    if form == "avg":
        return out("link", f"Trung bình một khoản {w} {ph} là bao nhiêu tiền?", "number", money(e["amount"].sum() / len(e)), f"SELECT ROUND(AVG({amt})) FROM {src}", notes)
    by = e.groupby("cat")["amount"].sum()
    if form == "top_cat":
        key = top1(by)
        if key is None:
            return None
        return out("link", f"{cap(ph)} {cat_name} nào có tổng tiền {w} lớn nhất?", "string", key, f"SELECT {cat} FROM {src} GROUP BY 1 ORDER BY SUM({amt}) DESC, 1 LIMIT 1", notes)
    key = str(c.pick(sorted(by.index)))
    return out("link", f"{cap(ph)} tổng tiền {w} của {cat_name} {key} là bao nhiêu?", "number", int(by[key]),
               f"SELECT ROUND(SUM({amt})) FROM {src} AND {low(cat)} = {lit(key.lower())}", notes)


# ---------- catalog attributes ----------

def k_attr(c: Ctx):
    if not c.flags.get("attr"):
        return None
    p = c.P(["month", "quarter", "last_week", "this_month"])
    if not p:
        return None
    a, b, ph = p
    A = c.shop.extra["attr_name"]
    prod = c.shop.products.dropna(subset=["attr"])
    l = per(c.lines, a, b).merge(prod[["code", "attr"]], on="code")
    if l.empty:
        return None
    C, cc = c.tab("catalog"), lambda role: c.col("catalog", role)                  # noqa: E731
    join = f"v_sales s JOIN {C} k ON {low('s.product_name')} = {low('k.' + cc('tên hàng'))}"
    w = win("s.order_date", a, b)
    notes = [f"Rule: {A} of an item comes from the catalog column {cc('thuộc tính')}; an item with that cell blank belongs to no value.",
             f"Join v_sales to {C} on the item name.", f"Period {ph} = [{a}, {b + timedelta(days=1)})."]
    form = c.pick(["revenue", "revenue", "top", "items"])
    if form == "top":
        key = top1(l.groupby("attr")["amount"].sum())
        if key is None:
            return None
        return out("attr", f"{cap(ph)} {A.lower()} nào có doanh thu cao nhất?", "string", key,
                   f"SELECT k.{cc('thuộc tính')} FROM {join} WHERE {w} AND k.{cc('thuộc tính')} IS NOT NULL GROUP BY 1 ORDER BY SUM(s.amount) DESC, 1 LIMIT 1", notes)
    v = str(c.pick(sorted(prod["attr"].unique())))
    hit = l[l["attr"] == v]
    cond = f"{w} AND k.{cc('thuộc tính')} = {lit(v)}"
    if form == "revenue":
        return out("attr", c.pick([f"Doanh thu các mặt hàng có {A.lower()} {v} {ph} là bao nhiêu?", f"{cap(ph)} hàng {A.lower()} {v} bán được bao nhiêu tiền?"]), "number",
                   money(hit["amount"].sum()), f"SELECT COALESCE(ROUND(SUM(s.amount)), 0) FROM {join} WHERE {cond}", notes)
    return out("attr", f"{cap(ph)} bán được bao nhiêu mặt hàng khác nhau có {A.lower()} {v}?", "number", int(hit["code"].nunique()),
               f"SELECT COUNT(DISTINCT s.product_name) FROM {join} WHERE {cond}", notes)


def k_variant(c: Ctx):
    """An attribute the rulebook reads from the item NAME (size, loại, số... by a suffix or prefix): totals, shares and counts over
    every item carrying one value."""
    R = c.rules.get("variant")
    if not R or not c.flags.get("variant"):
        return None
    p = c.P(["month", "quarter", "last_week", "this_month", "recent"])
    if not p:
        return None
    a, b, ph = p
    word, like = R["word"], lambda v: lit(R["like"].format(word=R["word"], v=v))      # noqa: E731
    prod = c.shop.products
    l = per(c.lines, a, b).merge(prod[["code", "variant", "base_unit"]], on="code")
    tagged = l.dropna(subset=["variant"])
    if tagged.empty:
        return None
    w = win("order_date", a, b)
    form = str(c.pick(["revenue", "revenue", "share", "orders", "items", "qty", "qty", "top", "best"] + (["in_group", "group_share"] * 2 if c.flags.get("group") else [])))
    if form == "top":
        key = top1(tagged.groupby("variant")["amount"].sum())
        if key is None:
            return None
        case = "CASE " + " ".join(f"WHEN product_name ILIKE {like(v)} THEN {lit(v)}" for v in R["values"]) + " END"
        notes = [rule_note(R), f"Read the {word} from product_name with one ILIKE per value; names matching none are left out.", f"Period {ph} = [{a}, {b + timedelta(days=1)})."]
        return out("variant", c.pick([f"{cap(ph)} {word} nào bán được nhiều tiền nhất?", f"{cap(word)} nào có doanh thu cao nhất {ph}?"]), "string", key,
                   f"SELECT v FROM (SELECT {case} AS v, SUM(amount) AS m FROM v_sales WHERE {w} GROUP BY 1) WHERE v IS NOT NULL ORDER BY m DESC, v LIMIT 1",
                   notes, [f"{word} {key}"])
    v = str(c.pick(sorted(tagged["variant"].unique())))
    hit, lab = tagged[tagged["variant"] == v], f"{word} {v}"
    cond = f"{w} AND product_name ILIKE {like(v)}"
    notes = [rule_note(R), f"{cap(lab)}: product_name ILIKE {like(v)} on v_sales (lines of valid orders), every item with that mark.",
             f"Period {ph} = [{a}, {b + timedelta(days=1)})."]
    if form in ("in_group", "group_share"):  # the mark inside one product group: both filters, the share over that group only
        g = str(c.pick(sorted(hit["group"].unique())))
        hg, lg, gw = hit[hit["group"] == g], l[l["group"] == g], f"lower(product_group) = {lit(g.lower())}"
        notes = notes + [f"Only the group {g}: product_group on v_sales."]
        if form == "in_group":
            return out("variant", c.pick([f"{cap(ph)} trong nhóm {g.lower()}, hàng {lab} bán được bao nhiêu tiền?", f"Doanh thu {lab} của nhóm {g.lower()} {ph} là bao nhiêu?"]),
                       "number", money(hg["amount"].sum()), f"SELECT ROUND(SUM(amount)) FROM v_sales WHERE {cond} AND {gw}", notes)
        return out("variant", c.pick([f"{cap(ph)} hàng {lab} chiếm bao nhiêu % doanh thu của nhóm {g.lower()}?",
                                      f"Trong nhóm {g.lower()}, {lab} chiếm bao nhiêu phần trăm doanh thu nhóm {ph}?"]), "percent",
                   pct(100 * hg["amount"].sum() / lg["amount"].sum()),
                   f"SELECT ROUND(100.0 * SUM(amount) FILTER (WHERE product_name ILIKE {like(v)}) / SUM(amount), 2) FROM v_sales WHERE {w} AND {gw}", notes)
    if form == "revenue":
        return out("variant", c.pick([f"{cap(ph)} hàng {lab} bán được bao nhiêu tiền, cộng mọi mặt hàng?", f"Doanh thu các mặt hàng {lab} {ph} là bao nhiêu?",
                                      f"Tổng doanh thu {lab} {ph}, tính tất cả các mặt hàng?"]), "number",
                   money(hit["amount"].sum()), f"SELECT ROUND(SUM(amount)) FROM v_sales WHERE {cond}", notes)
    if form == "share":
        return out("variant", c.pick([f"{cap(ph)} hàng {lab} chiếm bao nhiêu % doanh thu?", f"Hàng {lab} đóng góp bao nhiêu phần trăm tổng doanh thu {ph}?"]), "percent",
                   pct(100 * hit["amount"].sum() / l["amount"].sum()),
                   f"SELECT ROUND(100.0 * SUM(amount) FILTER (WHERE product_name ILIKE {like(v)}) / SUM(amount), 2) FROM v_sales WHERE {w}", notes)
    if form == "orders":
        return out("variant", c.pick([f"{cap(ph)} có bao nhiêu đơn có mua hàng {lab}?", f"Số đơn có ít nhất một mặt hàng {lab} {ph} là bao nhiêu?"]), "number",
                   int(hit["order_id"].nunique()), f"SELECT COUNT(DISTINCT order_id) FROM v_sales WHERE {cond}", notes)
    if form == "items":
        return out("variant", f"{cap(ph)} bán được bao nhiêu mặt hàng {lab} khác nhau?", "number", int(hit["code"].nunique()),
                   f"SELECT COUNT(DISTINCT product_name) FROM v_sales WHERE {cond}", notes)
    if form == "qty":
        units = hit["base_unit"].unique()
        if len(units) != 1:
            return None
        return out("variant", c.pick([f"{cap(ph)} bán được tổng cộng bao nhiêu {units[0]} {lab}, cộng tất cả các mặt hàng?",
                                      f"Tổng số {units[0]} hàng {lab} bán ra {ph} là bao nhiêu?"]), "number", int(hit["qty_base"].sum()),
                   f"SELECT ROUND(SUM(qty)) FROM v_sales WHERE {cond}", notes + ["v_sales.qty is already in the item's base unit."])
    key = top1(hit.groupby("code")["amount"].sum())
    if key is None:
        return None
    return out("variant", f"{cap(ph)} mặt hàng {lab} nào có doanh thu cao nhất?", "string", hit.loc[hit["code"] == key, "name"].iloc[0],
               f"SELECT product_name FROM v_sales WHERE {cond} GROUP BY product_name ORDER BY SUM(amount) DESC, product_name LIMIT 1", notes, [key])


# ---------- question shapes the train split lacks, on the canonical views ----------

def k_shape(c: Ctx):
    p = c.P(["month", "quarter", "last_week", "this_month", "recent", "range"])
    if not p:
        return None
    a, b, ph = p
    o, l = per(c.orders, a, b), per(c.lines, a, b)
    if o.empty:
        return None
    w = win("order_date", a, b)
    pn = f"Period {ph} = [{a}, {b + timedelta(days=1)})."
    form = c.pick(["distinct_items", "days_over", "top_share", "cust_revenue", "cust_orders", "active_days", "avg_lines", "max_order"])
    if form == "distinct_items":
        return out("shape", f"{cap(ph)} bán được bao nhiêu mặt hàng khác nhau?", "number", int(l["code"].nunique()),
                   f"SELECT COUNT(DISTINCT product_name) FROM v_sales WHERE {w}", ["Count items at their own grain: distinct product_name on v_sales.", pn])
    if form == "days_over":
        daily = l.groupby("date")["amount"].sum()
        x = int(np.quantile(daily, float(c.rng.uniform(0.3, 0.9))) // 100000 * 100000) or 100000
        word = f"{x // 1000000} triệu" if x % 1000000 == 0 else f"{vnd(x)}đ"
        return out("shape", f"{cap(ph)} có mấy ngày doanh thu từ {word} trở lên?", "number", int((daily >= x).sum()),
                   f"SELECT COUNT(*) FROM (SELECT order_date FROM v_sales WHERE {w} GROUP BY order_date HAVING SUM(amount) >= {x})",
                   ["Grain = days: total the revenue per day first, then count the days.", "Rulebook §2: 'từ N trở lên' includes N.", pn])
    if form == "top_share":
        key = top1(l.groupby("code")["qty_base"].sum())
        if key is None:
            return None
        share = l.loc[l["code"] == key, "amount"].sum() / l["amount"].sum() * 100
        sql = (f"SELECT ROUND(100.0 * (SELECT SUM(amount) FROM v_sales WHERE {w} AND product_name = (SELECT product_name FROM v_sales WHERE {w} "
               f"GROUP BY product_name ORDER BY SUM(qty) DESC, product_name LIMIT 1)) / (SELECT SUM(amount) FROM v_sales WHERE {w}), 2)")
        return out("shape", f"Mặt hàng bán chạy nhất {ph} chiếm bao nhiêu % doanh thu?", "percent", pct(share), sql,
                   ["Rulebook §3: best seller = highest base-unit quantity; then its share of the revenue of the same period.", pn])
    if form in ("cust_revenue", "cust_orders"):
        named = o.dropna(subset=["ckey"])
        counts = c.shop.customers["name"].value_counts()
        named = named[named["cname"].map(lambda n: counts.get(n, 0) == 1)]
        if named.empty:
            return None
        row = named.iloc[int(c.rng.integers(0, len(named)))]
        mine = named[named["ckey"] == row["ckey"]]
        cond = f"{w} AND customer_name = {lit(row['cname'])}"
        if form == "cust_revenue":
            return out("shape", c.pick([f"{cap(ph)} khách {row['cname']} mua bao nhiêu tiền?", f"Doanh thu từ khách {row['cname']} {ph} là bao nhiêu?"]), "number",
                       money(mine["revenue"].sum()), f"SELECT ROUND(SUM(revenue)) FROM v_orders WHERE {cond}", ["v_orders: valid orders with customer_name.", pn])
        return out("shape", f"{cap(ph)} khách {row['cname']} có bao nhiêu đơn?", "number", len(mine), f"SELECT COUNT(*) FROM v_orders WHERE {cond}",
                   ["v_orders: valid orders with customer_name.", pn])
    if form == "active_days":
        return out("shape", f"{cap(ph)} có bao nhiêu ngày có đơn hàng?", "number", int(o["date"].nunique()), f"SELECT COUNT(DISTINCT order_date) FROM v_orders WHERE {w}",
                   ["Grain = days with at least one valid order.", pn])
    if form == "avg_lines":
        return out("shape", f"Trung bình mỗi đơn {ph} có bao nhiêu dòng hàng?", "number", pct(o["n_lines"].mean()), f"SELECT ROUND(AVG(n_lines), 2) FROM v_orders WHERE {w}",
                   ["v_orders.n_lines = item lines of the order; an average that is not money rounds to 2 decimals.", pn])
    return out("shape", f"Đơn có giá trị cao nhất {ph} là bao nhiêu tiền?", "number", money(o["revenue"].max()), f"SELECT ROUND(MAX(revenue)) FROM v_orders WHERE {w}",
               ["v_orders.revenue = line amounts of the order.", pn])


KINDS = [(k_cust_rank, 8), (k_item_rank, 6), (k_day_rank, 5), (k_hour_rank, 4), (k_staff_rank, 4), (k_big_order, 8), (k_loyal, 7), (k_weekday_avg, 6),
         (k_bad_rate, 5), (k_new_cust, 5), (k_profit, 9), (k_purchase, 9), (k_stock, 8), (k_lots, 8), (k_price_effect, 5), (k_event, 14), (k_attr, 8), (k_shape, 12),
         (k_variant, 16), (k_plan, 18), (k_link, 14)]


def reply(r: dict) -> str:
    return "Notes:\n" + "\n".join("- " + n for n in r["notes"]) + f"\n\nTYPE: {r['type']}\n```sql\n{r['sql']}\n```"


def make2(shop, meta: dict, rules: dict, rng: np.random.Generator, n: int) -> list[tuple[dict, dict]]:
    funcs, w = zip(*KINDS)
    w = np.array(w, float) / sum(w)
    rows, seen, tries, ctx = [], set(), 0, {}
    per_kind: dict = {}
    while len(rows) < n and tries < n * 25:
        tries += 1
        today = shop.end if rng.random() < 0.75 else shop.start + timedelta(days=int(rng.integers(60, max(61, (shop.end - shop.start).days))))
        if today not in ctx:
            if len(ctx) > 6:
                ctx.clear()
            ctx[today] = Ctx(shop, meta, rules, rng, today)
        r = funcs[int(rng.choice(len(funcs), p=w))](ctx[today])
        if not r or r["text"] in seen or per_kind.get(r["kind"], 0) >= max(3, n // 4):
            continue
        if r["type"] == "unanswerable" and sum(1 for _, a in rows if a["type"] == "unanswerable") >= max(1, n // 20):
            continue
        seen.add(r["text"])
        per_kind[r["kind"]] = per_kind.get(r["kind"], 0) + 1
        text = noisy(rng, r["text"])
        qid = "s2_" + hashlib.sha1(f"{shop.sid}|{text}|{today}".encode()).hexdigest()[:12]
        rows.append(({"id": qid, "business_id": shop.sid, "question": text, "as_of_date": today.isoformat()},
                     {"id": qid, "type": r["type"], "answer": r["answer"], "alias": r["alias"], "clean_question": r["text"], "kind": r["kind"],
                      "sql": r["sql"], "reply": reply(r)}))
    return rows
