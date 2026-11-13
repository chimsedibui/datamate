"""Two sets of saved shop layers side by side, without labels: do the canonical views and the debt layer of set B describe the
same sales as set A? The private-test run drafts its layers fresh, so a fresh set should agree with a set that scored.

    python analysis/compare_layers.py configs/public_v9 runs/public_layers_v10/shard0 --data data/public_test

A and B are dirs holding logs/_views_<shop>.json and logs/_debt_<shop>.json. Per shop: whether each set has usable views,
the numbers every sales question rests on (lines, orders, revenue, base quantity, cancelled and returned orders, customers,
months whose revenue differs), how many lines carry each optional column, the cost column, and the debt fingerprint.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate import debts, views  # noqa: E402
from datamate.ingest import load_business  # noqa: E402

OPTIONAL = ["order_ts", "product_code", "product_group", "unit", "customer_key", "staff", "channel", "payment"]
num = lambda x: "-" if x is None else f"{x:,.0f}"


def facts(db, sql, prefix):
    views.build(db, sql, prefix)
    one = lambda q: db.execute(q).fetchone()
    lines, orders, rev, qty, items, cust = one(f"SELECT count(*), count(DISTINCT order_id), round(sum(amount)), round(sum(qty), 2), count(DISTINCT product_name), "
                                                f"count(DISTINCT customer_key) FROM {prefix}sales")
    n_all, ncan, nret = one(f"SELECT count(*), count(*) FILTER (WHERE is_cancelled), count(*) FILTER (WHERE is_returned) FROM {prefix}orders_all")
    f = {"lines": lines, "orders": orders, "revenue": rev, "qty": qty, "items": items, "customers": cust, "orders_all": n_all, "cancelled": ncan, "returned": nret,
         "months": dict(db.execute(f"SELECT strftime(order_date, '%Y-%m'), round(sum(amount)) FROM {prefix}sales GROUP BY 1").fetchall()),
         "filled": {c: round(one(f"SELECT avg(CAST({c} IS NOT NULL AS INT)) FROM {prefix}sales")[0] or 0, 2) for c in OPTIONAL}}
    f["time of day"] = round(one(f"SELECT avg(CAST(order_ts IS NOT NULL AND CAST(order_ts AS TIME) <> TIME '00:00:00' AS INT)) FROM {prefix}sales")[0] or 0, 2)
    if views.has_cost(sql):
        filled, margin = one(f"SELECT avg(CAST(cost IS NOT NULL AS INT)), 1 - sum(cost) / nullif(sum(amount) FILTER (WHERE cost IS NOT NULL), 0) FROM {prefix}sales")
        f["cost"] = "no cost on the sale lines" if not filled else f"on {filled:.0%} of lines, gross margin {margin:.0%}"
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--shops", nargs="*")
    args = ap.parse_args()
    read = lambda d, kind, b: json.loads(f.read_text(encoding="utf-8")) if (f := d / "logs" / f"_{kind}_{b}.json").exists() else None
    shops = sorted({f.stem[len("_views_"):] for d in (args.a, args.b) for f in (d / "logs").glob("_views_*.json")})
    same = diff = 0
    for b in args.shops or shops:
        db = load_business(args.data, b, canon=True)[0]
        va, vb = read(args.a, "views", b), read(args.b, "views", b)
        out, fs = [], []
        for name, v, prefix in (("A", va, "cmp_a_"), ("B", vb, "cmp_b_")):
            if not v or not v.get("sql"):
                out.append(f"   {name}: no views file" if not v else f"   {name}: no view picked ({v['agree']}/{v['k']} agree)")
                fs.append(None)
                continue
            try:
                f = facts(db, v["sql"], prefix)
            except Exception as e:  # noqa: BLE001
                out.append(f"   {name}: build failed: {str(e)[:160]}")
                fs.append(None)
                continue
            fs.append(f if v["used"] else None)
            out.append(f"   {name}: {v['agree']}/{v['k']} agree, used={v['used']}, {len(v.get('errors') or [])} failed drafts | lines {f['lines']} orders {f['orders']} revenue {num(f['revenue'])} "
                       f"qty {num(f['qty'])} items {f['items']} customers {f['customers']} | all orders {f['orders_all']} cancelled {f['cancelled']} returned {f['returned']}"
                       + (f" | cost: {f['cost']}" if "cost" in f else ""))
        notes = []
        if fs[0] and fs[1]:
            fa, fb = fs
            for k in ("lines", "orders", "revenue", "qty", "items", "customers", "orders_all", "cancelled", "returned"):
                if fa[k] != fb[k]:
                    rel = abs((fb[k] or 0) - (fa[k] or 0)) / max(1, abs(fa[k] or 0))
                    notes.append(f"{k} {num(fa[k])} -> {num(fb[k])} ({rel:.1%})")
            months = sorted(set(fa["months"]) | set(fb["months"]))
            off = [m for m in months if fa["months"].get(m) != fb["months"].get(m)]
            if off:
                notes.append(f"revenue differs in {len(off)}/{len(months)} months")
            for c in OPTIONAL:
                if abs(fa["filled"][c] - fb["filled"][c]) > 0.02:
                    notes.append(f"{c} filled {fa['filled'][c]:.0%} -> {fb['filled'][c]:.0%}")
            if abs(fa["time of day"] - fb["time of day"]) > 0.02:
                notes.append(f"lines with a time of day {fa['time of day']:.0%} -> {fb['time of day']:.0%}")
        elif bool(fs[0]) != bool(fs[1]):
            notes.append("views used in only one set")
        da, dbt = read(args.a, "debt", b), read(args.b, "debt", b)
        dline = []
        for name, d, prefix in (("A", da, "cmp_da_"), ("B", dbt, "cmp_db_")):
            if not d or not d.get("sqls"):
                dline.append(f"{name}: none")
                continue
            try:
                fp = debts.build(db, d["sqls"], prefix)
                dline.append(f"{name}: {d['agree']}/{d.get('k', '?')} used={d['used']} orders {fp[0]} payments {fp[2]} owing {fp[4]} owed {fp[5]:,}")
            except Exception as e:  # noqa: BLE001
                dline.append(f"{name}: {d['agree']}/{d.get('k', '?')} used={d['used']} build failed: {str(e)[:100]}")
        if da and dbt and (bool(da.get("used")) != bool(dbt.get("used")) or (da.get("used") and dline[0].split("orders")[-1] != dline[1].split("orders")[-1])):
            notes.append("debt layer differs")
        same += not notes
        diff += bool(notes)
        print(f"{b}: " + ("SAME sales numbers" if not notes else "DIFF: " + "; ".join(notes)))
        print("\n".join(out))
        print("   debt " + " | ".join(dline))
        db.close()
    print(f"\n{same} shops agree, {diff} differ")


if __name__ == "__main__":
    main()
