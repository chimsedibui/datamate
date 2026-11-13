"""Deterministic fallback semantic layer (no LLM): orders + order_lines from the main sales sheets.

Used when every LLM-written loader failed. A Qwen3-14B build fails on about half of the shops, and a shop without a
layer is abstained entirely; this mapper recovers the core sales facts (orders, dates, status, customers, products,
quantities, amounts) from column names, which cover the common revenue / order-count / product questions.

    tables, meta = auto_load(store)      # same contract as an LLM loader's load(raw)

Columns get a role from their accent-folded name (plus the README description when it names the column). Sheets with
the same columns are concatenated (one sheet per month, one file per year). The largest sheet group with product and
quantity/amount columns becomes order_lines; a header sheet (order code + date + totals, no product) is joined when
its codes match. Purchases, stock, payments, catalogues and side books are skipped by file/sheet name.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..core.parsers import fold_text, map_status, parse_datetime, parse_money, year_from_text

_SKIP_SHEET = re.compile(
    r"\b(nhap|ton|kho|giao hang|thanh toan|phieu thu|so quy|khach hang|san pham|hang hoa|danh muc|danhmuc|quy doi|"
    r"quydoi|bang gia|banggia|lich su gia|chi phi|bao cao|so no|sono|no|gop|tra gop|bao hanh|thu may|ghi tat|lo|thuoc|"
    r"han no|cong no|don vi tinh|donvitinh|chiphi|tonkho|nhaphang|khachhang|sanpham|hanghoa|thanhtoan|giaohang|"
    r"phieuthu|soquy|baocao|danhsachsanpham|danhsachkhachhang|danh sach san pham|danh sach khach hang)\b")
_NEVER = re.compile(r"barcode|ma vach|imei|hinh anh|\bsdt\b|phone|dien thoai|email|dia chi|\bmst\b|ma so thue|cccd|"
                    r"cmnd|bien so|lai xe|khu vuc|^ban$|gioi tinh|so khach$|trong luong|bao hanh|mo ta|thuoc tinh|"
                    r"id cua hang|ma chi nhanh|ma ca|hoa don dien tu|hddt")

# README column descriptions are written from a fixed vocabulary; checked before the column-name patterns.
_DESC_ROLES = [(r, re.compile(p)) for r, p in (
    ("skip", r"^cot trong|^so khach cua don|^ban$|^khu vuc|^so dien thoai|^dia chi|^so lo|^han su dung"),
    ("qty_base", r"^so luong da doi ra don vi co ban"),
    ("line_no", r"^so thu tu dong trong don"),
    ("order_id", r"^ma don\b(?! nhap)|^so hoa don|^ma hoa don|^so thu tu don|^so don\b"),
    ("product_name", r"^ten hang(?! \(file)|^ten san pham|^ten mon|^hang ban|^mat hang"),
    ("product_id", r"^ma hang\b"),
    ("qty", r"^so luong(?! (nhap|khach tra|ban lay|giao|da doi))"),
    ("datetime", r"^ngay ban|^thoi diem dat don|^ngay dat|^ngay gio ban"),
    ("time", r"^gio dat don|^gio ban"),
    ("unit", r"^don vi tinh|^don vi cua dong"),
    ("unit_price", r"^don gia$|^don gia\b(?! nhap)|^gia ban mot|^gia mot don vi ban"),
    ("line_amount", r"^thanh tien"),
    ("status", r"^trang thai don"),
    ("customer_name", r"^ten khach"),
    ("customer_id", r"^ma khach"),
    ("order_discount", r"^giam gia cua ca don|^tien giam cua ca don|^giam gia cua don"),
    ("line_discount", r"^giam gia$|^tien giam cua ca dong|^giam gia cua dong"),
    ("cost_per_unit", r"^gia von"),
    ("channel", r"^kenh ban"),
    ("debt_remaining", r"^con no|^so khach con no"),
    ("paid_amount", r"^da thanh toan|^khach da tra"),
    ("total_payable", r"^tong tien don|^so khach phai tra cho ca don|^khach can tra cho ca don"),
    ("goods_total", r"^tong tien hang|^tien hang cua don"),
    ("shipping_fee", r"^phi giao hang"),
    ("payment_method_raw", r"^hinh thuc thanh toan"),
    ("employee", r"^ma nhan vien ban|^nhan vien ban"),
    ("customer_group", r"^nhom khach"),
    ("note", r"^ghi chu"),
)]

# (role, pattern on the folded column name, exclusion pattern); first match wins.
_ROLES = [
    ("qty_base", r"quy doi|quy ra|so luong le|sl le|\ble\b.*so luong", None),
    ("cost_per_unit", r"gia von", None),
    ("line_discount", r"^giam|giam gia|tien giam|chiet khau|^ck$|^bot$|^giam \(", None),
    ("shipping_fee", r"phi (ship|van chuyen|giao)|^ship", None),
    ("goods_total", r"tong tien hang|tien hang", None),
    ("total_payable", r"tong (bill|phieu|thanh toan|sau giam|don|hoa don)|phai tra|tongtien|can tra|tong cong", None),
    ("paid_amount", r"khach (da )?tra|^da tra|da thu|^tien thu|da thanh toan", None),
    ("debt_remaining", r"con no|chua tra|cong no|^no$|con lai", None),
    ("unit_price", r"don gia|gia ban|gia le|gia 1 dv|gia don vi|gia san pham|^gia$|^gia \(|gia ban le|^price",
     r"gia (von|nhap|cu|moi|thu)"),
    ("line_amount", r"thanh tien|tri gia|t\.? ?tien|^tong tien$|^tong$|^cong$|amount|thanh_tien|^tong tien \(|"
                    r"^tien$|^tien ban$|^tt$", None),
    ("qty", r"^(so luong|sl|s\.l|sluong|soluong|so_luong|so luong san pham|sl ban|so luong ban|qty)\b",
     r"tra|giao|nhap|ton|quy doi|\ble\b"),
    ("unit", r"^(dvt|don vi|dv|unit)\b", r"quy doi|goc"),
    ("product_id", r"^(ma (hang|sp|san pham|h\.?\s?hoa|hang hoa|mon|hh)|id sp|sku|ma sku|ma_sp)", None),
    ("product_name", r"^(ten (hang|san pham|mon|hang hoa|sp|goi)|mat hang|hang hoa|hang ban|hang|san pham|mon|"
                     r"noi dung|item|product|ten_hang|lam gi|cong viec|dich vu)\b", r"nhom|loai|ap dung"),
    ("customer_id", r"^(ma (kh|khach)|so kh|ma khach mua|ma_kh)", None),
    ("order_id", r"^(ma (don|don hang|hoa don|hd|bill|chung tu|gd|giao dich|phieu)|so (ct|chung tu|hd|hoa don|don|phieu)|"
                 r"don so|ma_don|order|so bill|ma bill|hd)\b", r"nhap|thu\b|tra\b"),
    ("line_no", r"^(stt|dong|thu tu|no\.?|line)$", None),
    ("time", r"^(gio|gio ban|gio tao)", None),
    ("datetime", r"^(ngay|thoi gian|date|time|ngaythang|ngay_ban)",
     r"\b(sinh|nhap|tra|gop|thu|tt|ap dung|han|nhan|tao|phai|vao lam|khai truong|thoi viec|nhan viec|dang ky)\b|"
     r"giao(?! dich)"),
    ("status", r"trang thai|tinh trang|status|xu ly|ket qua|tien do", r"hddt|hoa don dien tu"),
    ("customer_name", r"khach|ben mua|doi tac|ho ten|ten kh|^kh$", r"\bma\b|nhom|so khach|loai|tra\b|doi tuong|kieu|"
                                                                    r"ten be"),
    ("customer_group", r"nhom khach|loai khach|doi tuong", None),
    ("channel", r"kenh|nguon|ban qua", None),
    ("payment_method_raw", r"phuong thuc|pttt|hinh thuc thanh toan|loai tt", None),
    ("employee", r"nhan vien|nguoi ban|ma nv", None),
    ("note", r"ghi chu|note|luu y", None),
]
_ROLES = [(r, re.compile(p), re.compile(x) if x else None) for r, p, x in _ROLES]


def _readme_desc(readme: str) -> dict[str, str]:
    """'- **Col**: description' lines of the README -> {folded col: folded description}."""
    out = {}
    for m in re.finditer(r"^\s*-\s*\*\*(.+?)\*\*\s*:\s*(.+)$", readme or "", re.M):
        out[fold_text(m.group(1))] = fold_text(m.group(2))
    return out


def _fold_col(c) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(c))  # CamelCase (sqlite exports): NgayLap -> Ngay Lap
    return re.sub(r"[_]+", " ", fold_text(s) or "").strip()


def column_roles(columns, desc: dict[str, str] | None = None) -> dict[str, str]:
    """{role: column} for one sheet; each role and each column used once. README descriptions win over names."""
    desc = desc or {}
    roles: dict[str, str] = {}
    by_name = []
    for c in columns:
        f = _fold_col(c)
        if not f or f.startswith("col"):
            continue
        d = desc.get(fold_text(str(c)), "")
        role = next((r for r, rx in _DESC_ROLES if d and rx.search(d)), None)
        if role == "skip":
            continue
        if role and role not in roles:
            roles[role] = c
            continue
        if not _NEVER.search(f):
            by_name.append((c, f))
    for c, f in by_name:
        for role, rx, ex in _ROLES:
            if role in roles or not rx.search(f) or (ex and ex.search(f)):
                continue
            roles[role] = c
            break
    return roles


def _sheets(store):
    for f in store.list_files():
        for sh in store.sheet_names(f):
            label = re.sub(r"[_\-.]+", " ", fold_text(f"{f} {sh or ''}") or "")
            label = re.sub(r"\.(xlsx|xls|csv|sqlite|db)\b", " ", label)
            yield f, sh, label


def _groups(store, desc):
    """Sheets with identical columns concatenated: [(roles, df, n_sheets, labels)]."""
    groups: dict[tuple, list] = {}
    for f, sh, label in _sheets(store):
        name_part = re.sub(r"\b(xlsx|xls|csv)\b", " ", label)
        if _SKIP_SHEET.search(name_part) and not re.search(r"ban hang|banhang|hoa don|hoadon|don hang|donhang|"
                                                           r"chi tiet|chitiet|so ban|soban", name_part):
            continue
        try:
            df = store.read_table(f, sh)
        except Exception:  # noqa: BLE001 - unreadable sheet: skip
            continue
        if not len(df):
            continue
        df = df.copy()
        df["__sheet"] = f"{f}::{sh}" if sh else f
        df["__year"] = year_from_text(f"{f} {sh or ''}")
        groups.setdefault(tuple(sorted(map(str, df.columns))), []).append(df)
    out = []
    for cols, dfs in groups.items():
        df = pd.concat(dfs, ignore_index=True)
        roles = column_roles([c for c in df.columns if not str(c).startswith("__")], desc)
        out.append((roles, df, len(dfs)))
    return out


def _parse_dt(df, roles):
    if "datetime" not in roles:
        return pd.Series(pd.NaT, index=df.index)
    years = df["__year"].dropna()
    dy = int(years.mode().iat[0]) if len(years) else None
    dt = parse_datetime(df[roles["datetime"]], default_year=dy)
    if "time" in roles:
        t = pd.to_timedelta(df[roles["time"]].astype(str).str.extract(r"(\d{1,2}:\d{2}(?::\d{2})?)")[0]
                            .map(lambda s: s if pd.isna(s) or s.count(":") == 2 else f"{s}:00"), errors="coerce")
        dt = dt.where(t.isna() | (dt.dt.hour.fillna(0) > 0) | (dt.dt.minute.fillna(0) > 0), dt.dt.normalize() + t)
    return dt


def _order_ids(df, roles, multi_sheet: bool):
    if "order_id" in roles:
        ids = df[roles["order_id"]].astype(object).where(df[roles["order_id"]].notna())
        ids = ids.map(lambda v: str(int(v)) if isinstance(v, float) and v.is_integer() else (None if pd.isna(v) else str(v).strip()))
        # Order header repeated only on the first line of the order: carry it down within the sheet.
        if ids.isna().mean() > 0.05:
            ids = ids.groupby(df["__sheet"]).ffill()
        if multi_sheet and _restarts(ids, df["__sheet"]):
            ids = df["__sheet"].astype(str) + "#" + ids.astype(str)
        return ids, True
    return pd.Series([f"R{i + 1}" for i in range(len(df))], index=df.index), False


def _restarts(ids: pd.Series, sheet: pd.Series) -> bool:
    """True when the same code appears in several sheets (per-sheet numbering such as STT)."""
    per = pd.DataFrame({"id": ids, "s": sheet}).dropna().drop_duplicates()
    return per["id"].duplicated().mean() > 0.2 if len(per) else False


def _status(df, roles):
    raw = df[roles["status"]] if "status" in roles else pd.Series(None, index=df.index, dtype=object)
    st = map_status(raw, default="completed")
    if "note" in roles:
        note = map_status(df[roles["note"]], default=None)
        st = note.where(note.isin(["returned", "cancelled"]), st)
    return st, raw


def auto_load(store):
    readme = store.readme() if hasattr(store, "readme") else ""
    desc = _readme_desc(readme)
    groups = _groups(store, desc)
    line_groups = [g for g in groups if ("product_name" in g[0] or "product_id" in g[0])
                   and ({"qty", "line_amount"} & set(g[0]))]
    head_groups = [g for g in groups if "order_id" in g[0] and "product_name" not in g[0] and "product_id" not in g[0]
                   and ({"datetime", "total_payable", "goods_total", "line_amount", "paid_amount"} & set(g[0]))]
    meta = {"industry": "", "auto_mapped": True, "notes": ["auto-mapped from column names (no LLM loader)"],
            "extra_desc": {}}
    if not line_groups and not head_groups:
        raise ValueError("auto_load: no sales sheet recognised")

    tables = {}
    head = None
    if head_groups:
        if line_groups:
            lr, ldf, _ = max(line_groups, key=lambda g: len(g[1]))
            lids = set(ldf[lr["order_id"]].dropna().astype(str)) if "order_id" in lr else set()
            def overlap(g):
                return len(lids & set(g[1][g[0]["order_id"]].dropna().astype(str))), len(g[1])
            roles, df, n = max(head_groups, key=overlap)
        else:
            roles, df, n = max(head_groups, key=lambda g: ("datetime" in g[0], len(g[1])))
        ids, _ = _order_ids(df, roles, n > 1)
        h = pd.DataFrame({"order_id": ids, "order_datetime": _parse_dt(df, roles)})
        h["status"], h["status_raw"] = _status(df, roles)
        for r in ("customer_id", "customer_name", "customer_group", "channel", "payment_method_raw", "employee", "note"):
            if r in roles:
                h[r] = df[roles[r]].astype(object)
        for r in ("goods_total", "order_discount", "line_discount", "shipping_fee", "total_payable", "paid_amount",
                  "debt_remaining"):
            if r in roles and not (r == "line_discount" and "order_discount" in roles):
                h["order_discount" if r == "line_discount" else r] = parse_money(df[roles[r]])
        if "total_payable" not in h and "line_amount" in roles:
            h["total_payable"] = parse_money(df[roles["line_amount"]])
        head = h.dropna(subset=["order_id"]).drop_duplicates("order_id")

    if line_groups:
        roles, df, n = max(line_groups, key=lambda g: len(g[1]))
        ids, from_file = _order_ids(df, roles, n > 1)
        meta["order_codes_from_file"] = from_file
        ln = pd.DataFrame({"order_id": ids})
        pid = df[roles["product_id"]].astype(object) if "product_id" in roles else None
        name = df[roles["product_name"]].astype(object) if "product_name" in roles else pid
        ln["product_name"] = name
        if pid is not None:
            ln["product_id"] = pid
        if "unit" in roles:
            ln["unit"] = df[roles["unit"]].astype(object)
        qty = parse_money(df[roles["qty"]]) if "qty" in roles else pd.Series(np.nan, index=df.index)
        price = parse_money(df[roles["unit_price"]]) if "unit_price" in roles else pd.Series(np.nan, index=df.index)
        amount = parse_money(df[roles["line_amount"]]) if "line_amount" in roles else qty * price
        if "line_discount" in roles and "line_amount" not in roles:
            amount = amount - parse_money(df[roles["line_discount"]]).fillna(0)
        qty = qty.where(qty.notna(), (amount / price).where(price > 0))
        qty = qty.fillna(1.0)
        ln["qty"] = qty
        ln["qty_base"] = parse_money(df[roles["qty_base"]]).fillna(qty) if "qty_base" in roles else qty
        implied = (amount / qty).where(qty > 0)
        # A price in another unit (per base unit, per pack) disagrees with the line amount; the amount is the revenue.
        off = (price / implied).where(implied > 0)
        price = price.where(off.isna() | ((off > 0.1) & (off < 10)), implied)
        price = price.where(price.notna(), implied)
        # Still inconsistent (free items, amount 0): leave the price unknown rather than contradict the amount.
        bad = (qty * price - amount).abs() > 9 * amount.abs().clip(lower=1)
        ln["unit_price"] = price.mask(bad & amount.notna())
        ln["line_amount"] = amount.where(amount.notna(), qty * ln["unit_price"])
        if "line_discount" in roles:
            ln["line_discount"] = parse_money(df[roles["line_discount"]])
        if "cost_per_unit" in roles:
            ln["cost_per_unit"] = parse_money(df[roles["cost_per_unit"]])
        disc = ln["line_discount"].fillna(0) if "line_discount" in ln else 0
        exp = ln["qty"] * ln["unit_price"] - disc
        ratio = (ln["line_amount"] / exp.where(exp > 0)).abs()
        ln["unit_price"] = ln["unit_price"].mask((ratio > 10) | (ratio < 0.1))
        ln = ln[ln["order_id"].notna() & ln["product_name"].notna()]

        o = pd.DataFrame({"order_id": ids, "order_datetime": _parse_dt(df, roles)})
        o["status"], o["status_raw"] = _status(df, roles)
        for r in ("customer_id", "customer_name", "customer_group", "channel", "payment_method_raw", "employee", "note"):
            if r in roles:
                o[r] = df[roles[r]].astype(object)
        for r in ("goods_total", "order_discount", "shipping_fee", "total_payable", "paid_amount", "debt_remaining"):
            if r in roles:
                o[r] = parse_money(df[roles[r]])
        o = o.dropna(subset=["order_id"])
        agg = {c: "first" for c in o.columns if c != "order_id"}
        o = o.groupby("order_id", sort=False).agg(agg).reset_index()
        if head is not None and len(set(head["order_id"].astype(str)) & set(o["order_id"].astype(str))) > 0.5 * len(o):
            extra = [c for c in head.columns if c not in o.columns or o[c].isna().all()]
            o = o.drop(columns=[c for c in extra if c in o.columns and c != "order_id"]).merge(
                head[["order_id"] + [c for c in extra if c != "order_id"]], on="order_id", how="left")
            if o["order_datetime"].isna().all() and "order_datetime" in head:
                o = o.drop(columns="order_datetime").merge(head[["order_id", "order_datetime"]], on="order_id",
                                                           how="left")
        o = o[o["order_datetime"].notna()]
        ln = ln[ln["order_id"].isin(set(o["order_id"]))]
        tables["orders"] = o
        tables["order_lines"] = ln.reset_index(drop=True)
    else:
        tables["orders"] = head[head["order_datetime"].notna()].reset_index(drop=True)
        meta["notes"].append("orders only: the data has no product lines")
    tables["orders"]["order_date"] = tables["orders"]["order_datetime"].dt.normalize()
    if "payment_method_raw" in tables["orders"]:
        tables["orders"]["payment_method"] = tables["orders"]["payment_method_raw"].map(_pay_method)
    return tables, meta


def _pay_method(v):
    f = fold_text(v) or ""
    for k, rx in (("cash", r"tien mat|cash"), ("transfer", r"chuyen khoan|ck|bank|ngan hang"),
                  ("ewallet", r"momo|zalo ?pay|vi|vnpay|shopee ?pay"), ("card", r"the|card|pos"),
                  ("cod", r"cod"), ("installment", r"tra gop|gop"), ("debt", r"no|cong no|ghi no")):
        if re.search(rf"\b({rx})\b", f):
            return k
    return "other" if f else None
