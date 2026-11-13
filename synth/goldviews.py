"""Correct canonical views for synthetic shops, written from what the generator knows (OUT/_meta/<shop>.json), not by a model.

Each view is the one SELECT datamate/views.py asks the model for, built here from the shop's real table and column names
(read back through datamate.ingest, so they match what the pipeline sees). views.build() materializes it with the pipeline's
own code, and the shop is kept only when the view's monthly revenue equals the generator's ground truth. The views are saved
as <out>/logs/_views_<shop>.json, the format `--views-from` reuses, so prompts built on them are the prompts inference shows.

    python -m synth.goldviews data/synth_train --out runs/goldviews_synth_train
"""
import argparse
import json
from pathlib import Path

import numpy as np

from datamate import views
from datamate.ingest import load_business, norm
from synth.questions import Facts, month_bounds
from synth.world import simulate


def q(c: str) -> str:
    return '"' + c.replace('"', '""') + '"'


def lit(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def low(expr: str) -> str:
    return f"lower(trim(CAST({expr} AS VARCHAR)))"


def in_list(expr: str, values: list[str]) -> str:
    return f"{low(expr)} IN ({', '.join(lit(v.lower().strip()) for v in values)})" if values else "false"


def factor_case(meta: dict, name: str, unit: str) -> str:
    whens = [f"WHEN {low(name)} = {lit(p['name'].lower())} AND {low(unit)} = {lit(u.lower())} THEN {k}"
             for p in meta["products"] for u, k in p["units"][1:]]
    return f"(CASE {' '.join(whens)} ELSE 1 END)" if whens else "1"


def base_unit_case(meta: dict, name: str) -> str:
    whens = [f"WHEN {lit(p['name'].lower())} THEN {lit(p['base_unit'])}" for p in meta["products"]]
    return f"(CASE {low(name)} {' '.join(whens)} END)"


def table_for(db, file_name: str, union: bool = False) -> str:
    names = [r[0] for r in db.execute("SHOW TABLES").fetchall()]
    stem = norm(file_name.rsplit(".", 1)[0])
    if union:
        cand = [n for n in names if n.endswith("_all") and stem.split("_")[0] in n]
    else:
        cand = [n for n in names if n == stem] or [n for n in names if n.startswith(stem)]
    if not cand:
        raise LookupError(f"no table for {file_name} among {names}")
    return cand[0]


def common(meta: dict, H, cust_name: str, status: str | None) -> dict:
    st, walk = meta["status"], meta["walk_in"]
    name = f"CASE WHEN {in_list(cust_name, walk)} OR trim(CAST({cust_name} AS VARCHAR)) = '' THEN NULL ELSE trim(CAST({cust_name} AS VARCHAR)) END" \
        if cust_name else "NULL"
    return {"customer_name": name,
            "is_cancelled": in_list(status, st["huỷ"]) if status else "false",
            "is_returned": in_list(status, st["trả hàng"]) if status else "false"}


def flat_sql(db, meta: dict) -> str:
    f = meta["files"][0]
    t, roles, h = q(table_for(db, f["name"])), f["roles"], meta["head"]
    H = lambda r: q(h[r]) if r in roles else None                          # noqa: E731
    ts = "NULL"
    if meta["flags"]["time"] and "giờ đặt đơn" not in roles:   # the date column carries the time of day
        ts = f"CAST({H('ngày bán')} AS TIMESTAMP)"
    if "giờ đặt đơn" in roles:
        ts = f"CAST(CAST({H('ngày bán')} AS DATE) + CAST({H('giờ đặt đơn')} AS TIME) AS TIMESTAMP)"
    qty = H("số lượng quy đổi") or f"{H('số lượng')} * {factor_case(meta, H('tên hàng'), H('đơn vị tính'))}"
    c = common(meta, H, H("tên khách"), H("trạng thái đơn"))
    return f"""SELECT CAST({H('mã đơn')} AS VARCHAR) AS order_id, CAST({H('ngày bán')} AS DATE) AS order_date, {ts} AS order_ts,
  NULL AS product_code, {H('nhóm hàng') or 'NULL'} AS product_group, {H('tên hàng')} AS product_name, {qty} AS qty,
  {base_unit_case(meta, H('tên hàng'))} AS unit, {H('thành tiền')} AS amount,
  {f"{H('số lượng')} * {H('giá vốn')}" if H('giá vốn') else 'NULL'} AS cost,
  {f"nullif(trim(CAST({H('mã khách')} AS VARCHAR)), '')" if H('mã khách') else 'NULL'} AS customer_id, {c['customer_name']} AS customer_name,
  {H('nhân viên') or 'NULL'} AS staff, {H('kênh bán') or 'NULL'} AS channel, {H('thanh toán') or 'NULL'} AS payment,
  {c['is_cancelled']} AS is_cancelled, {c['is_returned']} AS is_returned, 'item' AS line_kind
FROM {t}"""


def pos_sql(db, meta: dict) -> str:
    files = {f["name"]: f["roles"] for f in meta["files"]}
    h = meta["head"]
    O, L, P = (q(table_for(db, n)) for n in ("DanhSachDonHang.xlsx", "ChiTietDonHang.xlsx", "DanhMucHangHoa.xlsx"))
    oroles = files["DanhSachDonHang.xlsx"]
    oc = lambda r: f"o.{q(h[r])}" if r in oroles else None              # noqa: E731
    has_k = "DanhSachKhachHang.xlsx" in files
    K = q(table_for(db, "DanhSachKhachHang.xlsx")) if has_k else None
    cust = f"k.{q(h['tên khách'])}" if has_k else oc("tên khách")
    c = common(meta, None, cust, oc("trạng thái đơn"))
    ts = f"CAST({oc('ngày bán')} AS TIMESTAMP)" if meta["flags"]["time"] else "NULL"
    group = f"p.{q(h['nhóm hàng'])}" if "nhóm hàng" in files["DanhMucHangHoa.xlsx"] else "NULL"
    pname = f"coalesce(p.{q(h['tên hàng'])}, CAST(l.{q(h['mã hàng'])} AS VARCHAR))"
    unit_col = "l." + q(h["đơn vị tính"])
    qty = f"l.{q(h['số lượng'])} * {factor_case(meta, pname, unit_col)}"
    join_k = f"LEFT JOIN {K} k ON CAST(o.{q(h['mã khách'])} AS VARCHAR) = CAST(k.{q(h['mã khách'])} AS VARCHAR)" if has_k else ""
    return f"""SELECT CAST(l.{q(h['mã đơn'])} AS VARCHAR) AS order_id, CAST({oc('ngày bán')} AS DATE) AS order_date, {ts} AS order_ts,
  CAST(l.{q(h['mã hàng'])} AS VARCHAR) AS product_code, {group} AS product_group, {pname} AS product_name, {qty} AS qty,
  p.{q(h['đơn vị cơ bản'])} AS unit, l.{q(h['thành tiền'])} AS amount,
  {f"l.{q(h['số lượng'])} * l.{q(h['giá vốn'])}" if 'giá vốn' in files['ChiTietDonHang.xlsx'] else 'NULL'} AS cost,
  {f"nullif(trim(CAST({oc('mã khách')} AS VARCHAR)), '')" if oc('mã khách') else 'NULL'} AS customer_id, {c['customer_name']} AS customer_name,
  {oc('nhân viên') or 'NULL'} AS staff, {oc('kênh bán') or 'NULL'} AS channel, {oc('thanh toán') or 'NULL'} AS payment,
  {c['is_cancelled']} AS is_cancelled, {c['is_returned']} AS is_returned, 'item' AS line_kind
FROM {L} l JOIN {O} o ON CAST(l.{q(h['mã đơn'])} AS VARCHAR) = CAST(o.{q(h['mã đơn'])} AS VARCHAR)
LEFT JOIN {P} p ON CAST(l.{q(h['mã hàng'])} AS VARCHAR) = CAST(p.{q(h['mã hàng'])} AS VARCHAR) {join_k}"""


def ledger_sql(db, meta: dict) -> str:
    f = meta["files"][0]
    roles, h = f["roles"], meta["head"]
    # every month sheet of every book, by column name: ingest's *_all union leaves out a sheet whose columns differ
    # (a last month with no cancelled or returned order gets no status flags), so union the sheets here
    stem = norm(f["name"].rsplit(".", 1)[0]).split("_")[0]
    sheets = [n for (n,) in db.execute("SHOW TABLES").fetchall() if n.startswith(stem) and not n.endswith("_all")]
    keep = ", ".join(q(h[r]) for r in roles) + ', "_sheet"'
    t = "(" + " UNION ALL BY NAME ".join(f"SELECT {keep} FROM {q(n)}" for n in sheets) + ")"
    raw = keep
    oid = q(h["mã đơn"])
    fill = lambda r: f"max({q(h[r])}) OVER w" if r in roles else "NULL"   # order-level cells sit on the first line only
    c = common(meta, None, "c0", "s0" if "trạng thái đơn" in roles else None)
    ts = "CAST(t0 AS TIMESTAMP)" if "giờ đặt đơn" in roles else "NULL"
    return f"""WITH d AS (SELECT DISTINCT {raw} FROM {t}),
f AS (SELECT *, {fill('ngày bán')} AS d0, {fill('trạng thái đơn')} AS s0, {fill('tên khách')} AS c0, {fill('giờ đặt đơn')} AS t0
      FROM d WINDOW w AS (PARTITION BY {oid}))
SELECT CAST({oid} AS VARCHAR) AS order_id, CAST(d0 AS DATE) AS order_date, {ts} AS order_ts,
  NULL AS product_code, NULL AS product_group, {q(h['tên hàng'])} AS product_name,
  {q(h['số lượng'])} * {factor_case(meta, q(h['tên hàng']), q(h['đơn vị tính']))} AS qty,
  {base_unit_case(meta, q(h['tên hàng']))} AS unit, {q(h['thành tiền'])} AS amount, NULL AS cost,
  NULL AS customer_id, {c['customer_name']} AS customer_name, NULL AS staff, NULL AS channel, NULL AS payment,
  {c['is_cancelled']} AS is_cancelled, {c['is_returned']} AS is_returned, 'item' AS line_kind
FROM f"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shops", nargs="*", help="only these shops")
    a = ap.parse_args()
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    seed = json.loads((a.split / "meta.json").read_text())["seed"]
    ok, bad = 0, []
    for mf in sorted((a.split / "_meta").glob("*.json")):
        sid = mf.stem
        if a.shops and sid not in a.shops:
            continue
        meta = json.loads(mf.read_text(encoding="utf-8"))
        try:
            db = load_business(a.split, sid)[0]
            sql = {"flat": flat_sql, "pos": pos_sql, "ledger": ledger_sql}[meta["archetype"]](db, meta)
            views.build(db, sql)
            shop = simulate(sid, meta["industry"], np.random.default_rng([seed, int(sid[-3:]) - 1]))
            facts = Facts(shop, shop.end)
            for y, m in {(shop.start.year, shop.start.month), (shop.end.year, shop.end.month)}:
                a_, b_ = month_bounds(y, m)
                gold = facts.revenue(a_, b_)
                got = db.execute("SELECT round(sum(amount)) FROM v_sales WHERE order_date BETWEEN ? AND ?", [a_, b_]).fetchone()[0] or 0
                if int(got) != gold:
                    raise ValueError(f"{y}-{m:02d} revenue {int(got)} != gold {gold}")
        except Exception as e:  # noqa: BLE001
            bad.append((sid, meta["archetype"], str(e)[:200]))
            continue
        (a.out / "logs" / f"_views_{sid}.json").write_text(json.dumps({"agree": 8, "k": 8, "used": True, "sql": sql, "errors": [], "drafts": []},
                                                                      ensure_ascii=False), encoding="utf-8")
        ok += 1
    print(f"gold views: {ok} ok, {len(bad)} failed")
    for b in bad[:20]:
        print("  ", *b)


if __name__ == "__main__":
    main()
