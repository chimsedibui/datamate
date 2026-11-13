"""Simulate one shop as clean tables: products, customers, orders and order lines. Every answer is computed from these
tables, so the files rendered from them can be as messy as the real ones without losing the ground truth."""
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from synth import catalog as C

TET = {2025: date(2025, 1, 29), 2026: date(2026, 2, 17)}     # Lunar New Year days, for a sales bump around Tết


@dataclass
class Shop:
    sid: str
    industry: str
    label: str
    kind: str
    start: date
    end: date
    products: pd.DataFrame          # code, name, group, base_unit, price, units [(unit, factor)]
    customers: pd.DataFrame         # ckey, code, name, group, phone
    orders: pd.DataFrame            # order_id, date, time, ckey, cname, status, staff, channel, payment
    lines: pd.DataFrame             # order_id, line, code, name, group, unit, factor, qty, qty_base, unit_price, discount, amount
    flags: dict = field(default_factory=dict)   # what the shop records: customer_code, staff, channel, payment, status, time, group
    extra: dict = field(default_factory=dict)   # tables beyond sales (synth/world2.py): purchases, stock, lots, prices, events


def _name(rng: np.random.Generator) -> str:
    title = rng.choice(["", "", "", "Chị ", "Anh ", "Cô ", "Chú "])
    if title:
        return title + str(rng.choice(C.FIRST))
    return f"{rng.choice(C.LAST)} {rng.choice(C.MIDDLE)} {rng.choice(C.FIRST)}"


def simulate(sid: str, industry: str, rng: np.random.Generator) -> Shop:
    spec = C.INDUSTRIES[industry]
    start = date(2025, 1, 1) + timedelta(days=int(rng.integers(0, 6)) * 30)
    start = start.replace(day=1)
    end = min(start + timedelta(days=int(rng.integers(240, 560))), date(2026, 9, 30))
    flags = {"customer_code": bool(rng.random() < 0.5), "staff": bool(rng.random() < 0.3), "channel": bool(rng.random() < 0.3),
             "payment": bool(rng.random() < 0.4), "status": bool(rng.random() < 0.85), "time": bool(rng.random() < 0.6),
             "group": bool(rng.random() < 0.7), "customer_group": bool(rng.random() < 0.3)}

    # products, with Zipf-like popularity so rankings have clear leaders
    rows = []
    for group, items in spec["groups"].items():
        for name, unit, price, big in items:
            rows.append({"name": name, "group": group, "base_unit": unit, "price": int(price), "units": [(unit, 1)] + list(big)})
    order = rng.permutation(len(rows))
    products = pd.DataFrame([rows[i] for i in order]).reset_index(drop=True)
    prefix = str(rng.choice(["SP", "MH", "HH", "P", "SKU"]))
    products["code"] = [f"{prefix}{i + 1:03d}" for i in range(len(products))]
    weights = 1.0 / np.arange(1, len(products) + 1) ** float(rng.uniform(0.6, 1.1))
    weights /= weights.sum()

    # customers
    n_cust = int(rng.integers(40, 400))
    names = [_name(rng) for _ in range(n_cust)]
    customers = pd.DataFrame({"name": names})
    customers["code"] = [f"KH{i + 1:04d}" for i in range(n_cust)]
    customers["group"] = rng.choice(C.GROUPS_CUSTOMER, n_cust) if flags["customer_group"] else None
    customers["phone"] = ["09" + "".join(str(d) for d in rng.integers(0, 10, 8)) for _ in range(n_cust)]
    if not flags["customer_code"]:
        customers = customers.drop_duplicates("name").reset_index(drop=True)   # without codes a name is the customer
    customers["ckey"] = customers["code"] if flags["customer_code"] else customers["name"]
    cust_w = 1.0 / np.arange(1, len(customers) + 1) ** 0.8
    cust_w /= cust_w.sum()
    walk_in = float(rng.uniform(0.25, 0.6))

    staff = list(rng.choice(C.STAFF, int(rng.integers(2, 5)), replace=False))
    lo, hi = spec["orders"]
    base_rate = float(rng.uniform(lo, hi)) * 0.35                      # keep files near the real sizes
    increase = {}                                                        # code -> (date, factor): one price rise
    for code in products["code"]:
        if rng.random() < 0.3:
            increase[code] = (start + timedelta(days=int(rng.integers(30, max(31, (end - start).days - 30)))), float(rng.choice([1.05, 1.1])))

    o_rows, l_rows = [], []
    n = 0
    day = start
    while day <= end:
        rate = base_rate * (1.25 if day.weekday() >= 5 else 1.0)
        for tet in TET.values():
            if -20 <= (day - tet).days <= 3:
                rate *= 1.6
        for _ in range(int(rng.poisson(rate))):
            n += 1
            oid = f"DH{n:06d}"
            hour = int(rng.choice(np.arange(6, 22), p=_hour_weights(spec["kind"])))
            t = datetime(day.year, day.month, day.day, hour, int(rng.integers(0, 60)), int(rng.integers(0, 60)))
            if rng.random() < walk_in:
                ckey, cname = None, None
            else:
                c = customers.iloc[int(rng.choice(len(customers), p=cust_w))]
                ckey, cname = c["ckey"], c["name"]
            near_end = (end - day).days <= 4
            u = rng.random()
            status = ("đang giao" if near_end and u < 0.15 else "huỷ" if u < 0.045 else "trả hàng" if u < 0.075 else "hoàn thành") \
                if flags["status"] else "hoàn thành"
            pay = str(rng.choice(C.PAYMENTS, p=[0.55, 0.35, 0.05, 0.05]))
            o_rows.append({"order_id": oid, "date": day, "time": t, "ckey": ckey, "cname": cname, "status": status,
                           "staff": str(rng.choice(staff)), "channel": str(rng.choice(C.CHANNELS, p=[0.6, 0.15, 0.1, 0.1, 0.05])),
                           "payment": pay})
            k = int(rng.integers(1, {"fnb": 5, "retail": 4, "service": 3}[spec["kind"]]))
            for li, pi in enumerate(rng.choice(len(products), size=k, replace=False, p=weights)):
                p = products.iloc[int(pi)]
                unit, factor = p["units"][0]
                if len(p["units"]) > 1 and rng.random() < 0.2:
                    unit, factor = p["units"][int(rng.integers(1, len(p["units"])))]
                qty = int(rng.choice([1, 1, 1, 2, 2, 3, 4])) if factor == 1 else int(rng.choice([1, 1, 2]))
                price = p["price"]
                if p["code"] in increase and day >= increase[p["code"]][0]:
                    price = int(round(price * increase[p["code"]][1], -3)) if price >= 10000 else int(round(price * increase[p["code"]][1], -2))
                unit_price = price * factor
                disc = int(round(unit_price * qty * float(rng.choice([0.05, 0.1])), -3)) if rng.random() < 0.05 else 0
                l_rows.append({"order_id": oid, "line": li + 1, "code": p["code"], "name": p["name"], "group": p["group"], "unit": unit,
                               "factor": int(factor), "qty": qty, "qty_base": qty * int(factor), "unit_price": int(unit_price),
                               "discount": disc, "amount": int(unit_price * qty - disc)})
        day += timedelta(days=1)
    shop = Shop(sid, industry, spec["label"], spec["kind"], start, end, products, customers, pd.DataFrame(o_rows), pd.DataFrame(l_rows), flags)
    shop.extra["increase"] = increase          # code -> (day the price rose, factor)
    return shop


def _hour_weights(kind: str) -> np.ndarray:
    hours = np.arange(6, 22)
    if kind == "fnb":
        w = np.exp(-((hours - 9) ** 2) / 6) + np.exp(-((hours - 19) ** 2) / 8)
    else:
        w = np.exp(-((hours - 10) ** 2) / 10) + 0.8 * np.exp(-((hours - 17) ** 2) / 10)
    return w / w.sum()
