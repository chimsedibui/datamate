"""Questions about a simulated shop, with answers computed from its clean tables under the train rulebook.

Each generator draws a period and an entity, computes the answer, and returns None when the rulebook would exclude the
question (a tie at a ranking boundary, an empty denominator). Unanswerable questions come from the same rulebook section 7:
a period wholly outside the data, or information the shop does not record. Phrasing noise (no diacritics, teencode, fillers,
typos) is applied last and never touches numbers or dates.
"""
import hashlib
import math
import re
import unicodedata
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

from synth.world import Shop

VALID_OUT = {"huỷ", "trả hàng"}


def money(x: float) -> int:
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def pct(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# ---------- periods: (start, end inclusive, phrase) ----------

def month_bounds(y: int, m: int) -> tuple[date, date]:
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return date(y, m, 1), nxt - timedelta(days=1)


def period(rng: np.random.Generator, shop: Shop, today: date, kind: str | None = None):
    kind = kind or str(rng.choice(["month", "month", "month", "quarter", "last_week", "this_month", "yesterday", "recent", "range", "ytd"]))
    if kind == "month":
        first = date(shop.start.year, shop.start.month, 1)
        months = (today.year - first.year) * 12 + today.month - first.month
        k = int(rng.integers(0, months + 1))
        y, m = first.year + (first.month - 1 + k) // 12, (first.month - 1 + k) % 12 + 1
        a, b = month_bounds(y, m)
        no_year = (y == today.year and m <= today.month) or (y == today.year - 1 and m > today.month)
        forms = [f"tháng {m}/{y}", f"th{m}/{y}", f"tháng {m} năm {y}", f"{m}/{y}", f"tháng {m:02d}/{y}"] + ([f"tháng {m}"] if no_year else [])
        return a, b, str(rng.choice(forms)), ("month", y, m)        # a named period keeps its whole window (rows after today count)
    if kind == "quarter":
        q = int(rng.integers(1, 5))
        y = int(rng.choice([today.year, today.year - 1]))
        a, b = date(y, 3 * q - 2, 1), month_bounds(y, 3 * q)[1]
        if a > today:
            return None
        return a, b, str(rng.choice([f"quý {q}/{y}", f"quý {q} năm {y}", f"Q{q}/{y}"])), ("quarter", y, q)
    if kind == "last_week":
        monday = today - timedelta(days=today.weekday()) - timedelta(days=7)
        return monday, monday + timedelta(days=6), str(rng.choice(["tuần rồi", "tuần trước"])), ("last_week",)
    if kind == "this_month":
        return today.replace(day=1), today, "tháng này", ("this_month",)
    if kind == "yesterday":
        d = today - timedelta(days=1)
        return d, d, str(rng.choice(["hôm qua", "bữa qua", "ngày hôm qua"])), ("day",)
    if kind == "recent":
        n = int(rng.choice([7, 7, 14, 30]))
        return today - timedelta(days=n - 1), today, f"{n} ngày gần nhất", ("recent", n)
    if kind == "range":
        a = shop.start + timedelta(days=int(rng.integers(0, max(1, (today - shop.start).days - 10))))
        b = min(a + timedelta(days=int(rng.integers(2, 20))), today)
        return a, b, f"từ {a:%d/%m/%Y} đến {b:%d/%m/%Y}", ("range",)
    if kind == "ytd":
        return date(today.year, 1, 1), today, str(rng.choice(["từ đầu năm", "từ đầu năm tới nay", "từ đầu năm đến giờ"])), ("ytd",)
    return None


def previous_period(tag: tuple, a: date, b: date, today: date):
    """The comparison period of rulebook 1 (percent change) and 2 (this month vs same days of last month)."""
    if tag[0] == "month":
        y, m = (tag[1], tag[2] - 1) if tag[2] > 1 else (tag[1] - 1, 12)
        pa, pb = month_bounds(y, m)
        return pa, pb, "tháng trước"
    if tag[0] == "this_month":
        pa, last = month_bounds(*(((today.year, today.month - 1)) if today.month > 1 else (today.year - 1, 12)))
        pb = min(pa.replace(day=min(today.day, last.day)), last)
        return pa, pb, "tháng trước"
    if tag[0] == "last_week":
        return a - timedelta(days=7), b - timedelta(days=7), "tuần trước nữa"
    return None


# ---------- measures on the clean tables ----------

class Facts:
    def __init__(self, shop: Shop, today: date):
        self.shop, self.today = shop, today
        # Rows after today are "not happened yet" only for questions relative to today (tháng này, tuần rồi, N ngày gần nhất,
        # từ đầu năm...), and those windows end at today by construction. A named period (tháng M/Y, quý Q/Y, a named year)
        # keeps its whole window, as the train answers do (datamate/hints.py: "a named period keeps its own window").
        o = shop.orders
        self.orders_all = o
        self.orders = o[~o["status"].isin(VALID_OUT)]
        self.lines = shop.lines.merge(self.orders[["order_id", "date", "time", "ckey", "cname", "staff", "channel", "payment"]], on="order_id")

    def in_period(self, df, a, b):
        return df[(df["date"] >= a) & (df["date"] <= b)]

    def covered(self, a, b) -> bool:      # rulebook 7: a period wholly outside the data is unanswerable
        return not (b < self.shop.start or a > self.shop.end)

    def revenue(self, a, b, mask=None) -> int:
        lines = self.in_period(self.lines, a, b)
        if mask is not None:
            lines = lines[mask(lines)]
        return money(lines["amount"].sum())

    def order_count(self, a, b) -> int:
        return int(self.in_period(self.orders, a, b)["order_id"].nunique())

    def ranked(self, df, by, label, ascending_key="code"):
        """Totals by label, descending, ties broken by code (rulebook 9)."""
        t = df.groupby([ascending_key, label], as_index=False)[by].sum().sort_values([by, ascending_key], ascending=[False, True])
        return t.reset_index(drop=True)


def ok_top(t: pd.DataFrame, by: str, k: int) -> bool:
    """Rulebook: a tie at the k boundary (or at rank 1) excludes the question."""
    if len(t) < k or t[by].iloc[0] <= 0:
        return len(t) > 0 and len(t) < k
    return len(t) == k or t[by].iloc[k - 1] != t[by].iloc[k]


# ---------- question kinds: each returns (text, type, answer, alias) or None ----------

def q_revenue(rng, f: Facts):
    p = period(rng, f.shop, f.today)
    if not p:
        return None
    a, b, ph, _ = p
    if not f.covered(a, b):
        return _unans(rng, ph)
    t = str(rng.choice(["Doanh thu {p} là bao nhiêu?", "{P} cửa hàng bán được bao nhiêu tiền?", "Cho mình biết doanh thu {p}.",
                         "{P} thu về bao nhiêu tiền vậy?", "Tổng doanh thu {p} được bao nhiêu?"]))
    return _fmt(t, ph), "number", f.revenue(a, b), []


def q_orders(rng, f: Facts):
    p = period(rng, f.shop, f.today)
    if not p:
        return None
    a, b, ph, _ = p
    if not f.covered(a, b):
        return _unans(rng, ph, "số đơn")
    t = str(rng.choice(["{P} có bao nhiêu đơn hàng?", "Số đơn {p} là bao nhiêu?", "Đếm giúp mình số đơn {p}.", "{P} bán được mấy đơn?"]))
    return _fmt(t, ph), "number", f.order_count(a, b), []


def q_avg(rng, f: Facts):
    p = period(rng, f.shop, f.today)
    if not p:
        return None
    a, b, ph, _ = p
    n = f.order_count(a, b)
    if not f.covered(a, b) or n == 0:
        return None
    t = str(rng.choice(["Giá trị trung bình mỗi đơn {p} là bao nhiêu?", "{P} trung bình mỗi đơn bán được bao nhiêu tiền?",
                         "Bình quân một đơn {p} là bao nhiêu?"]))
    return _fmt(t, ph), "number", money(f.revenue(a, b) / n), []


def q_change(rng, f: Facts):
    p = period(rng, f.shop, f.today, str(rng.choice(["month", "this_month", "last_week"])))
    if not p:
        return None
    a, b, ph, tag = p
    prev = previous_period(tag, a, b, f.today)
    if not prev or not f.covered(prev[0], prev[1]) or prev[0] < f.shop.start:
        return None
    cur, old = f.revenue(a, b), f.revenue(prev[0], prev[1])
    if old == 0:
        return None
    t = str(rng.choice(["Doanh thu {p} tăng hay giảm bao nhiêu % so với {q}?", "{P} doanh thu thay đổi bao nhiêu phần trăm so với {q}?",
                         "So với {q}, doanh thu {p} tăng trưởng bao nhiêu %?"]))
    return _fmt(t, ph, q=prev[2]), "percent", pct((cur - old) / old * 100), []


def q_top_item(rng, f: Facts):
    p = period(rng, f.shop, f.today)
    if not p:
        return None
    a, b, ph, _ = p
    if not f.covered(a, b):
        return None
    t = f.ranked(f.in_period(f.lines, a, b), "qty_base", "name")
    if len(t) == 0 or not ok_top(t, "qty_base", 1):
        return None
    word = "món" if f.shop.kind == "fnb" else str(rng.choice(["mặt hàng", "sản phẩm", "hàng"]))
    q = str(rng.choice([f"{{W}} bán chạy nhất {{p}} là gì?", f"{{P}} {word} nào bán được nhiều nhất?", f"{{W}} nào hot nhất {{p}}?"]))
    return _fmt(q.replace("{W}", word.capitalize()).replace("{w}", word), ph), "string", t["name"].iloc[0], [t["code"].iloc[0]]


def q_top3(rng, f: Facts):
    p = period(rng, f.shop, f.today)
    if not p:
        return None
    a, b, ph, _ = p
    if not f.covered(a, b):
        return None
    t = f.ranked(f.in_period(f.lines, a, b), "qty_base", "name")
    if len(t) < 3 or not ok_top(t, "qty_base", 3):
        return None
    q = str(rng.choice(["Top 3 mặt hàng bán chạy nhất {p}?", "Liệt kê 3 sản phẩm bán nhiều nhất {p}.", "{P} 3 món nào bán chạy nhất?"]))
    names = list(t["name"].iloc[:3])
    return _fmt(q, ph), "list", names, [[c] for c in t["code"].iloc[:3]]


def q_item_qty(rng, f: Facts):
    p = period(rng, f.shop, f.today, str(rng.choice(["month", "quarter", "range"])))
    if not p:
        return None
    a, b, ph, _ = p
    if not f.covered(a, b):
        return None
    prod = f.shop.products.iloc[int(rng.integers(0, len(f.shop.products)))]
    lines = f.in_period(f.lines, a, b)
    qty = int(lines.loc[lines["code"] == prod["code"], "qty_base"].sum())
    unit = prod["base_unit"]
    q = str(rng.choice([f"{{P}} bán được bao nhiêu {unit} {prod['name']}?", f"Số lượng {prod['name']} bán ra {{p}} là bao nhiêu {unit}?",
                        f"{{p}} bán đc mấy {unit} {prod['name']}"]))
    return _fmt(q, ph), "number", qty, []


def q_group(rng, f: Facts):
    p = period(rng, f.shop, f.today, str(rng.choice(["month", "quarter"])))
    if not p:
        return None
    a, b, ph, _ = p
    group = str(rng.choice(sorted(f.shop.products["group"].unique())))
    share = rng.random() < 0.5
    if not f.shop.flags["group"]:
        q = f"Nhóm {group.lower()} chiếm bao nhiêu % doanh thu {{p}}?" if share else f"Doanh thu nhóm {group.lower()} {{p}} là bao nhiêu?"
        return _fmt(q, ph), "unanswerable", None, []
    if not f.covered(a, b):
        return None
    g = f.revenue(a, b, lambda d: d["group"] == group)
    if share:
        total = f.revenue(a, b)
        if total == 0:
            return None
        return _fmt(f"Nhóm {group.lower()} chiếm bao nhiêu % doanh thu {{p}}?", ph), "percent", pct(g / total * 100), []
    return _fmt(str(rng.choice([f"Doanh thu nhóm {group.lower()} {{p}} là bao nhiêu?", f"{{P}} nhóm {group.lower()} bán được bao nhiêu tiền?"])), ph), "number", g, []


def q_cancel(rng, f: Facts):
    p = period(rng, f.shop, f.today, str(rng.choice(["month", "quarter", "last_week"])))
    if not p:
        return None
    a, b, ph, _ = p
    q = str(rng.choice(["Tỉ lệ đơn huỷ {p} là bao nhiêu %?", "{P} có bao nhiêu phần trăm đơn bị huỷ?"]))
    if not f.shop.flags["status"]:
        return _fmt(q, ph), "unanswerable", None, []
    allo = f.in_period(f.orders_all, a, b)
    if len(allo) == 0:
        return None
    return _fmt(q, ph), "percent", pct((allo["status"] == "huỷ").sum() / len(allo) * 100), []


def q_customers(rng, f: Facts):
    p = period(rng, f.shop, f.today)
    if not p:
        return None
    a, b, ph, _ = p
    if not f.covered(a, b):
        return None
    o = f.in_period(f.orders, a, b)
    q = str(rng.choice(["{P} có bao nhiêu khách đã mua hàng?", "Số khách mua hàng {p} là bao nhiêu?", "{P} có mấy khách mua?"]))
    return _fmt(q, ph), "number", int(o["ckey"].dropna().nunique()), []


def q_top_customer(rng, f: Facts):
    p = period(rng, f.shop, f.today, str(rng.choice(["month", "quarter", "ytd", "last_week"])))
    if not p:
        return None
    a, b, ph, _ = p
    if not f.covered(a, b):
        return None
    lines = f.in_period(f.lines, a, b).dropna(subset=["ckey"])
    t = lines.groupby("ckey", as_index=False).agg(amount=("amount", "sum"), cname=("cname", "first")).sort_values(["amount", "ckey"], ascending=[False, True])
    if len(t) == 0 or (len(t) > 1 and t["amount"].iloc[0] == t["amount"].iloc[1]):
        return None
    q = str(rng.choice(["Khách nào mua nhiều tiền nhất {p}?", "{P} ai là khách mua nhiều tiền nhất?", "Khách hàng chi nhiều nhất {p} là ai?"]))
    alias = [t["ckey"].iloc[0]] if f.shop.flags["customer_code"] else []
    return _fmt(q, ph), "string", t["cname"].iloc[0], alias


def q_best_day(rng, f: Facts):
    p = period(rng, f.shop, f.today, "month")
    if not p:
        return None
    a, b, ph, _ = p
    if not f.covered(a, b):
        return None
    d = f.in_period(f.lines, a, b).groupby("date")["amount"].sum().sort_values(ascending=False)
    if len(d) == 0 or (len(d) > 1 and d.iloc[0] == d.iloc[1]):
        return None
    q = str(rng.choice(["{P} ngày nào cửa hàng bán được nhiều tiền nhất?", "Ngày nào {p} có doanh thu cao nhất?"]))
    return _fmt(q, ph), "date", d.index[0].isoformat(), []


def q_best_month(rng, f: Facts):
    years = sorted({d.year for d in f.orders["date"]})
    y = int(rng.choice(years))
    m = f.orders[[d.year == y for d in f.orders["date"]]]
    lines = f.lines[[d.year == y for d in f.lines["date"]]]
    if len(m) == 0:
        return None
    s = lines.groupby(lines["date"].map(lambda d: f"{d.year}-{d.month:02d}"))["amount"].sum().sort_values(ascending=False)
    if len(s) < 2 or s.iloc[0] == s.iloc[1]:
        return None
    q = str(rng.choice([f"Năm {y} tháng nào có doanh thu cao nhất?", f"Tháng nào trong năm {y} bán được nhiều tiền nhất?"]))
    return q, "date", s.index[0], []


def q_staff(rng, f: Facts):
    p = period(rng, f.shop, f.today, str(rng.choice(["month", "quarter", "last_week"])))
    if not p:
        return None
    a, b, ph, _ = p
    q = str(rng.choice(["Nhân viên nào bán giỏi nhất {p}?", "{P} bạn nhân viên nào bán được nhiều tiền nhất?"]))
    if not f.shop.flags["staff"]:
        return _fmt(q, ph), "unanswerable", None, []
    if not f.covered(a, b):
        return None
    s = f.in_period(f.lines, a, b).groupby("staff")["amount"].sum().sort_values(ascending=False)
    if len(s) == 0 or (len(s) > 1 and s.iloc[0] == s.iloc[1]):
        return None
    return _fmt(q, ph), "string", s.index[0], []


def q_channel(rng, f: Facts):
    p = period(rng, f.shop, f.today, str(rng.choice(["month", "quarter"])))
    if not p:
        return None
    a, b, ph, _ = p
    ch = str(rng.choice(["Facebook", "Zalo", "Shopee", "Tại quầy"]))
    q = f"Doanh thu bán qua kênh {ch} {{p}} là bao nhiêu?"
    if not f.shop.flags["channel"]:
        return _fmt(q, ph), "unanswerable", None, []
    if not f.covered(a, b):
        return None
    return _fmt(q, ph), "number", f.revenue(a, b, lambda d: d["channel"] == ch), []


def q_payment(rng, f: Facts):
    p = period(rng, f.shop, f.today, str(rng.choice(["month", "last_week", "this_month"])))
    if not p:
        return None
    a, b, ph, _ = p
    pay = str(rng.choice(["Chuyển khoản", "Tiền mặt"]))
    q = f"{{P}} doanh thu thanh toán bằng {pay.lower()} là bao nhiêu?"
    if not f.shop.flags["payment"]:
        return _fmt(q, ph), "unanswerable", None, []
    if not f.covered(a, b):
        return None
    return _fmt(q, ph), "number", f.revenue(a, b, lambda d: d["payment"] == pay), []


def q_shift(rng, f: Facts):
    p = period(rng, f.shop, f.today, str(rng.choice(["month", "last_week"])))
    if not p:
        return None
    a, b, ph, _ = p
    # 'bán được nhiều hơn' alone could mean quantity: always name the measure (revenue)
    q = str(rng.choice(["Doanh thu ca chiều {p} hơn hay kém ca sáng bao nhiêu %?", "{P} doanh thu ca chiều cao hơn ca sáng bao nhiêu phần trăm?"]))
    if not f.shop.flags["time"]:
        return _fmt(q, ph), "unanswerable", None, []
    if not f.covered(a, b):
        return None
    morning = f.revenue(a, b, lambda d: d["time"].map(lambda t: 6 <= t.hour <= 13))
    afternoon = f.revenue(a, b, lambda d: d["time"].map(lambda t: 14 <= t.hour <= 21))
    if morning == 0:
        return None
    return _fmt(q, ph), "percent", pct((afternoon - morning) / morning * 100), []


def q_diff(rng, f: Facts):
    p1 = period(rng, f.shop, f.today, "month")
    p2 = period(rng, f.shop, f.today, "month")
    if not p1 or not p2 or p1[3] == p2[3] or not f.covered(p1[0], p1[1]) or not f.covered(p2[0], p2[1]):
        return None
    q = str(rng.choice(["Doanh thu {a} hơn hay kém {b} bao nhiêu tiền?", "Doanh thu {a} chênh lệch bao nhiêu so với {b}?"]))
    return q.format(a=p1[2], b=p2[2]), "number", f.revenue(p1[0], p1[1]) - f.revenue(p2[0], p2[1]), []


def q_big_orders(rng, f: Facts):
    p = period(rng, f.shop, f.today, str(rng.choice(["month", "last_week", "quarter"])))
    if not p:
        return None
    a, b, ph, _ = p
    if not f.covered(a, b):
        return None
    per = f.in_period(f.lines, a, b).groupby("order_id")["amount"].sum()
    if len(per) == 0:
        return None
    x = int(np.quantile(per, float(rng.uniform(0.6, 0.95))) // 10000 * 10000) or 10000
    word = f"{x // 1000}k" if x < 1_000_000 else (f"{x // 1_000_000} triệu" if x % 1_000_000 == 0 else f"{x:,}".replace(",", "."))
    q = str(rng.choice([f"{{P}} có bao nhiêu đơn trên {word}?", f"Số đơn có giá trị hơn {word} {{p}} là bao nhiêu?"]))
    return _fmt(q, ph), "number", int((per > x).sum()), []


def q_not_sold(rng, f: Facts):
    n = int(rng.choice([7, 14, 30]))
    a, b = f.today - timedelta(days=n - 1), f.today
    before = set(f.lines.loc[f.lines["date"] < a, "code"])
    during = set(f.in_period(f.lines, a, b)["code"])
    q = str(rng.choice([f"Có bao nhiêu mặt hàng đã từng bán mà trong {n} ngày gần nhất không bán được cái nào?",
                        f"{n} ngày gần nhất có mấy mặt hàng bị ế (không bán ra)?"]))
    return q, "number", len(before - during), []


def q_unrecorded(rng, f: Facts):
    q = str(rng.choice(["Khách nữ mua bao nhiêu tiền {p}?", "Địa chỉ của khách mua nhiều nhất {p} là gì?",
                        "{P} khách đánh giá cửa hàng mấy sao?", "Lý do khách huỷ đơn nhiều nhất {p} là gì?"]))
    p = period(rng, f.shop, f.today, "month")
    return (_fmt(q, p[2]), "unanswerable", None, []) if p else None


KINDS = [(q_revenue, 14), (q_orders, 8), (q_avg, 6), (q_change, 8), (q_top_item, 7), (q_top3, 4), (q_item_qty, 7), (q_group, 6),
         (q_cancel, 4), (q_customers, 4), (q_top_customer, 4), (q_best_day, 3), (q_best_month, 2), (q_staff, 2), (q_channel, 2),
         (q_payment, 2), (q_shift, 2), (q_diff, 3), (q_big_orders, 3), (q_not_sold, 2), (q_unrecorded, 2)]


def _unans(rng, ph, what="doanh thu"):
    return _fmt(str(rng.choice([f"{what.capitalize()} {{p}} là bao nhiêu?", f"{{P}} {what} được bao nhiêu?"])), ph), "unanswerable", None, []


def _fmt(t: str, ph: str, **kw) -> str:
    return t.replace("{P}", ph[:1].upper() + ph[1:]).replace("{p}", ph).format(**kw) if kw else t.replace("{P}", ph[:1].upper() + ph[1:]).replace("{p}", ph)


# ---------- phrasing noise ----------

TEEN = [("doanh thu", ["dthu", "doanh thu", "DT"]), ("bao nhiêu", ["bn", "bnhiu", "bao nhiu", "nhiêu"]), ("được", ["đc", "dc", "được"]),
        ("không", ["ko", "k", "không"]), ("vậy", ["z", "v", "vậy"]), ("cửa hàng", ["cửa hàng", "shop", "tiệm"]), ("khách hàng", ["khách", "kh"])]
PREFIX = ["", "", "", "Cho mình hỏi ", "Xem giúp ", "ê ", "Shop ơi ", "Cho hỏi ", "Giúp mình xem "]
COMMON = {"doanh", "tháng", "bao", "nhiêu", "được", "hàng", "khách", "bán", "nhất", "trong", "những", "của", "tiền", "đơn", "nhiều",
          "ngày", "tuần", "trước", "chạy", "hỏi", "giúp", "biết", "mình", "phần", "trăm", "trung", "bình", "nhóm"}
SUFFIX = ["", "", "", " vậy", " ta", " z", " nhé", " 🤔", " ạ"]


def strip_marks(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")


def noisy(rng, text: str) -> str:
    t = text.rstrip(".?")
    if rng.random() < 0.4:
        for word, subs in TEEN:
            if word in t and rng.random() < 0.7:
                t = t.replace(word, str(rng.choice(subs)))
    prefix = "" if re.match(r"(cho|đếm|liệt kê|xem|giúp)\b", t, re.I) else str(rng.choice(PREFIX))
    t = prefix + (t[:1].lower() + t[1:] if rng.random() < 0.5 else t) + str(rng.choice(SUFFIX))
    if rng.random() < 0.1:                       # one typo in a common word: never in a name, a number or a date
        words = t.split(" ")
        cand = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha() and w.islower() and w in COMMON]
        if cand:
            i = int(rng.choice(cand))
            w = words[i]
            j = int(rng.integers(1, len(w) - 1))
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            t = " ".join(words)
    if rng.random() < 0.26:
        t = strip_marks(t)
    return t + ("?" if rng.random() < 0.5 else "")


def make_questions(shop: Shop, rng: np.random.Generator, n: int) -> list[tuple[dict, dict]]:
    out, seen, tries = [], set(), 0
    funcs, w = zip(*KINDS)
    w = np.array(w, float) / sum(w)
    while len(out) < n and tries < n * 20:
        tries += 1
        today = shop.end if rng.random() < 0.75 else shop.start + timedelta(days=int(rng.integers(60, max(61, (shop.end - shop.start).days))))
        res = funcs[int(rng.choice(len(funcs), p=w))](rng, _facts(shop, today))
        if not res:
            continue
        text, typ, ans, alias = res
        if text in seen:
            continue
        seen.add(text)
        qtext = noisy(rng, text)
        qid = "s_" + hashlib.sha1(f"{shop.sid}|{qtext}|{today}".encode()).hexdigest()[:12]
        out.append(({"id": qid, "business_id": shop.sid, "question": qtext, "as_of_date": today.isoformat()},
                    {"id": qid, "type": typ, "answer": ans, "alias": alias, "clean_question": text}))
    return out


_CACHE: dict = {}


def _facts(shop: Shop, today: date) -> Facts:
    key = (shop.sid, today)
    if key not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = Facts(shop, today)
    return _CACHE[key]
