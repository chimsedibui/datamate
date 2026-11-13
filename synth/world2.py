"""World v2: what a shop records beyond sales, simulated consistently with its sales.

Adds to a simulated shop (synth/world.py): cost of goods on every sale line, purchase receipts from suppliers (with lots and
expiry dates where the industry has them), ending stock per item, a price-change list, one catalog attribute column, and an
event table of a kind the train split does not have (repair / warranty tickets or service appointments). Everything is
derived from the same orders, so answers on any of these tables can be computed exactly.
"""
import math
import re
import zlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from synth.world import Shop, simulate

OUT = {"huỷ", "trả hàng"}
EXPIRY = {"nha_thuoc": (180, 900), "tap_hoa": (60, 365), "me_va_be": (180, 720), "my_pham": (365, 1000), "thu_cung": (120, 540),
          "nong_san": (365, 900)}                                   # shelf life in days, industries that track lots
SUPPLIERS = ["Công ty TNHH Minh Phát", "NPP Hoàng Gia", "Đại lý Thành Công", "Công ty CP Phương Nam", "Tổng kho Bình An",
             "NPP Đức Thịnh", "Công ty TNHH An Khang", "Đại lý Kim Liên"]
ATTRS = {"Xuất xứ": ["Việt Nam", "Thái Lan", "Nhật Bản", "Hàn Quốc"], "Kệ trưng bày": ["A1", "A2", "B1", "B2"],
         "Dòng hàng": ["Phổ thông", "Cao cấp", "Khuyến mãi"], "Mùa bán": ["Quanh năm", "Mùa hè", "Mùa lễ Tết"]}
EVENTS = {   # flavor -> vocabulary; "free" marks rows that bring no money (warranty work, a no-show)
    "ticket": {"industries": {"dien_thoai", "dien_may", "sua_xe", "giay_dep", "thoi_trang", "van_phong_pham"},
               "what": ["Thay linh kiện", "Vệ sinh, bảo dưỡng", "Sửa lỗi", "Kiểm tra", "Đổi mới"], "free": "Bảo hành", "paid": "Sửa tính phí",
               "fee": (50, 600)},
    "visit": {"industries": {"thu_cung", "my_pham", "hoa_tuoi", "me_va_be", "tiem_banh", "sua_xe"},
              "what": ["Tư vấn", "Chăm sóc cơ bản", "Gói đầy đủ", "Làm theo yêu cầu", "Kiểm tra định kỳ"], "free": "Không đến", "paid": "Đã đến",
              "fee": (80, 900)},
}


def simulate2(sid: str, industry: str, rng: np.random.Generator, name_rule: dict | None = None, books: dict | None = None) -> Shop:
    shop = simulate(sid, industry, rng)
    if name_rule:  # before every other table, so purchases, stock and price lists carry the same names
        _variants(shop, np.random.default_rng([zlib.crc32(sid.encode()), 77]), name_rule)
    x = np.random.default_rng(int(rng.integers(1 << 31)))
    _cost(shop, x)
    _purchases(shop, x)
    _prices(shop, x)
    _attribute(shop, x)
    _events(shop, x)
    if books:  # their own generators, so the tables above are the same with and without the side books
        _plans(shop, np.random.default_rng([zlib.crc32(sid.encode()), 78]), books)
        _links(shop, np.random.default_rng([zlib.crc32(sid.encode()), 79]), books)
    return shop


def _variants(shop: Shop, x, rule: dict) -> None:
    """Write the split's attribute into about 4 of 5 item names, the way its rulebook row reads it back; the others keep a
    plain name and belong to no value. Names the catalogue already wrote with a size lose that mark first."""
    p = shop.products
    used, names, tags = set(), [], []
    for old in p["name"]:
        base = re.sub(r"\s*\((?:M|L)\)$|\s+size \w+$", "", old)
        free = [v for v in rule["values"] if rule["name"].format(base=base, word=rule["word"], v=v) not in used]
        v = str(x.choice(free)) if free and (x.random() < 0.8 or base in used) else None
        name = rule["name"].format(base=base, word=rule["word"], v=v) if v else base
        if name in used:  # no value left for a second item with this base name
            name, v = f"{base} {len(names) + 1}", None
        used.add(name); names.append(name); tags.append(v)
    p["name"], p["variant"] = names, tags
    shop.lines["name"] = shop.lines["code"].map(dict(zip(p["code"], names)))
    shop.flags["variant"] = True


def _cost(shop: Shop, x) -> None:
    p = shop.products
    step = [100 if v < 20000 else 1000 for v in p["price"]]
    p["cost"] = [max(s, int(round(v * x.uniform(0.55, 0.85) / s)) * s) for v, s in zip(p["price"], step)]
    cost = dict(zip(p["code"], p["cost"]))
    shop.lines["unit_cost"] = [cost[c] * f for c, f in zip(shop.lines["code"], shop.lines["factor"])]   # per sale unit (rulebook 5)
    shop.lines["cost"] = shop.lines["unit_cost"] * shop.lines["qty"]
    shop.flags["cost"] = bool(x.random() < (0.3 if shop.kind == "fnb" else 0.55))


def _purchases(shop: Shop, x) -> None:
    p = shop.products
    goods = p[p["group"] != "Dịch vụ"] if shop.kind != "fnb" else p.iloc[0:0]
    shop.flags.update(purchases=False, stock=False, lots=False)
    if len(goods) == 0 or x.random() >= 0.6:
        return
    sold = shop.lines.merge(shop.orders[~shop.orders["status"].isin(OUT)][["order_id", "date"]], on="order_id")
    days = (shop.end - shop.start).days + 1
    sup = [str(s) for s in x.choice(SUPPLIERS, int(x.integers(3, 6)), replace=False)]
    with_lots = shop.industry in EXPIRY and x.random() < 0.7
    rows = []
    for _, g in goods.iterrows():
        daily = np.zeros(days)
        s_ = sold[sold["code"] == g["code"]]
        np.add.at(daily, [(d - shop.start).days for d in s_["date"]], s_["qty_base"].to_numpy(dtype=float))
        unit, factor = g["units"][-1] if len(g["units"]) > 1 else g["units"][0]
        cycle, main = int(x.integers(7, 29)), sup[int(x.integers(0, len(sup)))]
        for k in range(0, days, cycle):                     # a receipt at the start of each cycle covers that cycle's sales
            need = daily[k:k + cycle].sum()
            if need == 0 and k > 0:
                continue
            qty = max(1, math.ceil(need * x.uniform(1.05, 1.4) / factor))
            unit_cost = max(100, int(round(g["cost"] * factor * x.uniform(0.96, 1.04), -2)))
            rows.append({"date": shop.start + timedelta(days=k), "supplier": main if x.random() < 0.8 else sup[int(x.integers(0, len(sup)))],
                         "code": g["code"], "name": g["name"], "unit": unit, "factor": int(factor), "qty": qty, "qty_base": qty * int(factor),
                         "unit_cost": unit_cost, "amount": qty * unit_cost, "status": "hoàn thành"})
    for r in [dict(rows[int(i)]) for i in x.choice(len(rows), max(1, len(rows) // 30), replace=False)]:   # cancelled receipts: never arrived
        r.update(status="huỷ", date=r["date"] + timedelta(days=int(x.integers(1, 5))))
        if r["date"] <= shop.end:
            rows.append(r)
    df = pd.DataFrame(rows).sort_values(["date", "supplier", "status", "code"]).reset_index(drop=True)
    done = sold.groupby("code")["qty_base"].sum()
    for code in goods["code"]:                              # some goods sell down to (almost) nothing by the last day
        if x.random() < 0.2:
            idx = df.index[(df["code"] == code) & (df["status"] != "huỷ")]
            left = int(df.loc[idx, "qty_base"].sum() - done.get(code, 0))
            i, f = idx[-1], int(df.loc[idx[-1], "factor"])
            cut = min(left // f, int(df.loc[i, "qty"]) - 1)
            if cut > 0:
                df.loc[i, "qty"] -= cut
                df.loc[i, "qty_base"] = df.loc[i, "qty"] * f
                df.loc[i, "amount"] = df.loc[i, "qty"] * df.loc[i, "unit_cost"]
    key = df["date"].astype(str) + "|" + df["supplier"] + "|" + df["status"]
    df["receipt_id"] = ["PN%05d" % (i + 1) for i in pd.factorize(key)[0]]
    df["line"] = df.groupby("receipt_id").cumcount() + 1
    lo, hi = EXPIRY.get(shop.industry, (0, 0))
    df["lot"] = [f"L{d:%y%m}{i + 1:04d}" if with_lots and st != "huỷ" else None for i, (d, st) in enumerate(zip(df["date"], df["status"]))]
    df["expiry"] = [d + timedelta(days=int(x.integers(lo, hi))) if lot else None for d, lot in zip(df["date"], df["lot"])]
    shop.extra["purchases"], shop.extra["suppliers"] = df, sup
    shop.flags["purchases"] = True

    ok = df[df["status"] != "huỷ"]
    bought, out = ok.groupby("code")["qty_base"].sum(), sold.groupby("code")["qty_base"].sum()
    stock = goods[["code", "name", "base_unit", "cost", "units"]].copy()
    stock["ending"] = [int(bought.get(c, 0) - out.get(c, 0)) for c in stock["code"]]
    assert (stock["ending"] >= 0).all(), "stock went negative"
    shop.extra["stock"] = stock.reset_index(drop=True)
    shop.flags["stock"] = bool(x.random() < 0.75)
    if with_lots:                                           # sales use the oldest receipts first, so what is left sits in the newest lots
        lots = ok[["lot", "code", "name", "date", "expiry", "qty_base", "unit_cost", "factor"]].copy()
        lots["unit_cost_base"] = lots["unit_cost"] / lots["factor"]
        remaining = []
        for code, g in lots.groupby("code", sort=False):
            used = float(out.get(code, 0))
            for q in g.sort_values("date")["qty_base"]:
                take = min(q, used)
                used -= take
                remaining.append(int(q - take))
        lots = lots.sort_values(["code", "date"]).assign(remaining=remaining).sort_values(["date", "code"]).reset_index(drop=True)
        shop.extra["lots"] = lots
        shop.flags["lots"] = True


def _prices(shop: Shop, x) -> None:
    rows = []
    price = dict(zip(shop.products["code"], shop.products["price"]))
    name = dict(zip(shop.products["code"], shop.products["name"]))
    for code, (day, f) in shop.extra["increase"].items():
        old = price[code]
        new = int(round(old * f, -3)) if old >= 10000 else int(round(old * f, -2))
        if new > old and shop.start < day <= shop.end:
            rows.append({"code": code, "name": name[code], "date": day, "old": int(old), "new": new})
    shop.extra["prices"] = pd.DataFrame(rows, columns=["code", "name", "date", "old", "new"])
    shop.flags["price_history"] = bool(len(rows) and x.random() < 0.6)


def _attribute(shop: Shop, x) -> None:
    attr = str(x.choice(list(ATTRS)))
    vals = ATTRS[attr]
    shop.products["attr"] = [None if x.random() < 0.15 else str(vals[int(x.integers(0, len(vals)))]) for _ in range(len(shop.products))]
    shop.extra["attr_name"] = attr
    shop.flags["attr"] = bool(x.random() < 0.45)


def _events(shop: Shop, x) -> None:
    flavors = [k for k, v in EVENTS.items() if shop.industry in v["industries"]]
    shop.flags["events"] = False
    if not flavors or x.random() >= 0.7:
        return
    flavor = str(x.choice(flavors))
    spec = EVENTS[flavor]
    staff = sorted(set(shop.orders["staff"]))
    days = (shop.end - shop.start).days + 1
    rate = float(x.uniform(1.5, 9))
    keys = ["09" + "".join(str(d) for d in x.integers(0, 10, 8)) for _ in range(max(20, int(days * rate / 3)))]
    w = 1.0 / np.arange(1, len(keys) + 1) ** 0.7
    w /= w.sum()
    rows, n = [], 0
    for k in range(days):
        day = shop.start + timedelta(days=k)
        for _ in range(int(x.poisson(rate * (1.3 if day.weekday() >= 5 else 1.0)))):
            n += 1
            free = bool(x.random() < 0.22)
            lo, hi = spec["fee"]
            done = day + timedelta(days=int(x.integers(0, 6))) if flavor == "ticket" else None
            rows.append({"eid": f"{'PH' if flavor == 'ticket' else 'LH'}{n:05d}", "date": day,
                         "time": datetime(day.year, day.month, day.day, int(x.integers(8, 20)), int(x.integers(0, 60))),
                         "date2": done if done and done <= shop.end else None, "entity": keys[int(x.choice(len(keys), p=w))],
                         "what": str(x.choice(spec["what"])), "free": free, "staff": str(x.choice(staff)),
                         "amount": 0 if free else int(x.integers(lo, hi)) * 1000})
    shop.extra["events"] = {"flavor": flavor, "df": pd.DataFrame(rows)}
    shop.flags["events"] = True


def _month_day(d, k: int, day: int):
    """Day `day` of the k-th month after the month of d."""
    m = d.year * 12 + d.month - 1 + k
    return d.replace(year=m // 12, month=m % 12 + 1, day=day)


def _plans(shop: Shop, x, words: dict) -> None:
    """Orders paid in periods: a down payment at purchase, then n equal periods, period k due on a fixed day of the k-th month
    after the purchase month. Payments come in period order, one per period, early, late or never."""
    shop.flags["plans"] = False
    if x.random() >= 0.8:
        return
    rev = shop.lines.groupby("order_id")["amount"].sum()
    o = shop.orders[shop.orders["cname"].notna() & ~shop.orders["status"].isin(OUT)].copy()
    o["rev"] = o["order_id"].map(rev)
    o = o[o["rev"] >= max(300000, float(o["rev"].quantile(0.6)))]
    take = min(len(o), int(x.integers(40, 160)))
    if take < 25:
        return
    o = o.iloc[np.sort(x.choice(len(o), take, replace=False))]
    plans, sched = [], []
    for i, r in enumerate(o.itertuples(), 1):
        n = int(x.choice([3, 4, 6, 9, 12]))
        total = int(r.rev)
        per = max(1000, int(total * float(x.uniform(0.5, 0.8)) / n) // 1000 * 1000)
        down = total - n * per
        day = int(x.integers(1, 29))
        pid = f"TG{i:04d}"
        plans.append({"pid": pid, "oid": r.order_id, "date": r.date, "cname": r.cname, "total": total, "down": down, "n": n, "per": per, "day": day})
        stop = n + 1 if x.random() < 0.8 else int(x.integers(1, n + 1))
        prev = r.date
        for k in range(1, n + 1):
            due = _month_day(r.date, k, day)
            paid = None
            if k < stop and prev is not None:
                u = x.random()
                delay = int(x.integers(-5, 1)) if u < 0.65 else int(x.integers(1, 16)) if u < 0.9 else int(x.integers(16, 41))
                paid = max(due + timedelta(days=delay), prev + timedelta(days=1))   # one payment a day per order: no two identical rows
                if paid > shop.end:
                    paid = None
            prev = paid
            sched.append({"pid": pid, "k": k, "due": due, "amount": per, "paid": paid})
    shop.extra["plans"] = {"layout": str(x.choice(["two", "sched"])), "plans": pd.DataFrame(plans), "sched": pd.DataFrame(sched),
                           "plan": words["plan"], "p": words["p"]}
    shop.flags["plans"] = True


def _links(shop: Shop, x, words: dict) -> None:
    """A book of amounts tied to orders by the order id: a deposit taken before the order, a credit for goods handed in, a delivery
    fee the shop pays, a gift. Orders of every status can have a row; a few orders have two."""
    from synth.rules import LINK_BOOKS
    shop.flags["links"] = False
    flavor = words["flavor"]
    if x.random() >= 0.8:
        return
    rev = shop.lines.groupby("order_id")["amount"].sum()
    o = shop.orders
    take = int(len(o) * float(x.uniform(0.05, 0.18)))
    if take < 30:
        return
    o = o.iloc[np.sort(x.choice(len(o), take, replace=False))]
    cats = LINK_BOOKS[flavor]["cat"]
    w = x.dirichlet(np.ones(len(cats)) * 2)
    rows, n = [], 0
    for r in o.itertuples():
        total = int(rev.get(r.order_id, 0))
        for _ in range(2 if x.random() < 0.06 else 1):
            n += 1
            if flavor == "deposit":
                d, amount = max(shop.start, r.date - timedelta(days=int(x.integers(1, 11)))), max(10000, int(total * float(x.uniform(0.1, 0.4))) // 10000 * 10000)
            elif flavor == "credit":
                d, amount = r.date, max(5000, int(total * float(x.uniform(0.05, 0.35))) // 5000 * 5000)
            elif flavor == "ship":
                d, amount = min(shop.end, r.date + timedelta(days=int(x.integers(0, 3)))), int(x.integers(3, 16)) * 5000
            else:
                d, amount = r.date, int(x.integers(2, 21)) * 5000
            rows.append({"lid": f"SP{n:05d}", "date": d, "oid": r.order_id, "odate": r.date, "cat": str(cats[int(x.choice(len(cats), p=w))]),
                         "amount": amount, "valid": r.status not in OUT})
    shop.extra["links"] = {"flavor": flavor, "df": pd.DataFrame(rows), "book": words["book"], "w": words["w"], "cat_name": LINK_BOOKS[flavor]["cat_name"]}
    shop.flags["links"] = True
