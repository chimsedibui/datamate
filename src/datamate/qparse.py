"""Deterministic pre-analysis of a question, injected into the QA prompt.

Small models misread periods ("quý 4" -> April), guess entity names and pick the wrong answer type. These are
resolved here with fixed rules (rulebook §2 time conventions) so the model only has to write the computation:

    analyze(question, as_of, tables, cal) -> str   (markdown block; empty when nothing was found)
"""
from __future__ import annotations

import re

import pandas as pd

from .parsers import fold_text, norm_text

_STOP = {"ban", "duoc", "bao", "nhieu", "trong", "thang", "nam", "ngay", "cua", "hang", "cho", "the", "nao", "la",
         "va", "co", "khong", "mot", "cac", "nhung", "tong", "so", "luong", "tien", "doanh", "thu", "don", "voi",
         "tu", "den", "het", "hom", "nay", "qua", "tuan", "truoc", "roi", "nhat", "khach", "mat", "san", "pham",
         "mon", "gia", "size", "loai", "nhom", "bn", "z", "j", "a", "e", "oi", "nhe", "vay", "nhi", "giup", "gium",
         "xem", "coi", "check", "minh", "toi", "shop", "cua", "quan", "tiem", "o", "duoc", "dc", "may", "bnhiu"}


def _d(x) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def _month_end(y: int, m: int) -> pd.Timestamp:
    return pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)


def _year_for(m: int, as_of: pd.Timestamp, day: int = 1) -> int:
    """Month/day without a year: this year unless that is after today, else last year (rulebook §2)."""
    try:
        return as_of.year if pd.Timestamp(as_of.year, m, day) <= as_of else as_of.year - 1
    except ValueError:
        return as_of.year - 1


def _full_year(y: str) -> int:
    y = int(y)
    return y + 2000 if y < 100 else y


def resolve_periods(q: str, as_of: pd.Timestamp, cal: pd.DataFrame | None = None) -> list[tuple[str, str, str]]:
    """[(phrase, start, end)] for every period expression found in the question."""
    f = fold_text(q)
    a = pd.Timestamp(as_of).normalize()
    out: list[tuple[str, str, str]] = []
    used: list[tuple[int, int]] = []

    def add(m, label, s, e):
        if any(m.start() < u1 and u0 < m.end() for u0, u1 in used):
            return
        used.append((m.start(), m.end()))
        out.append((label, _d(s), _d(e)))

    # explicit ranges first: "từ 1/3/2026 đến 15/3/2026"
    for m in re.finditer(r"tu (?:ngay )?(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))? (?:den|toi|-) (?:het )?(?:ngay )?"
                         r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?", f):
        d1, m1, y1, d2, m2, y2 = m.groups()
        y2 = _full_year(y2) if y2 else _year_for(int(m2), a, int(d2))
        y1 = _full_year(y1) if y1 else y2
        try:
            add(m, m.group(0), pd.Timestamp(y1, int(m1), int(d1)), pd.Timestamp(y2, int(m2), int(d2)))
        except ValueError:
            pass
    for m in re.finditer(r"(\d+) ngay (?:gan (?:day|nhat)|qua|vua qua|cuoi)", f):
        n = int(m.group(1))
        add(m, m.group(0), a - pd.Timedelta(days=n - 1), a)
    for m in re.finditer(r"quy (\d)(?:[/ ](?:nam )?(\d{4}))?", f):
        qn = int(m.group(1))
        if 1 <= qn <= 4:
            y = int(m.group(2)) if m.group(2) else (a.year if pd.Timestamp(a.year, 3 * qn - 2, 1) <= a else a.year - 1)
            add(m, m.group(0), pd.Timestamp(y, 3 * qn - 2, 1), _month_end(y, 3 * qn))
    for m in re.finditer(r"thangs? (\d{1,2})(?:\s*[/-]\s*|\s+nams?\s+|\s+)(\d{4})", f):
        mo, y = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            add(m, m.group(0), pd.Timestamp(y, mo, 1), _month_end(y, mo))
    for m in re.finditer(r"(?:(den het|tinh den|cuoi|het) )?(?:ngay )?(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", f):
        try:
            d = pd.Timestamp(_full_year(m.group(4)), int(m.group(3)), int(m.group(2)))
            if m.group(1):
                used.append((m.start(), m.end()))
                out.append((m.group(0) + " (tình trạng lúc hết ngày: cộng mọi phát sinh từ đầu dữ liệu)",
                            "start of data", _d(d)))
            else:
                add(m, m.group(0), d, d)
        except ValueError:
            pass
    for m in re.finditer(r"thangs? (\d{1,2})(?![\d/])", f):
        mo = int(m.group(1))
        if 1 <= mo <= 12:
            y = _year_for(mo, a)
            add(m, m.group(0) + " (không ghi năm)", pd.Timestamp(y, mo, 1), _month_end(y, mo))
    for m in re.finditer(r"ngay (\d{1,2})[/-](\d{1,2})(?![\d/])", f):
        try:
            y = _year_for(int(m.group(2)), a, int(m.group(1)))
            d = pd.Timestamp(y, int(m.group(2)), int(m.group(1)))
            add(m, m.group(0) + " (không ghi năm)", d, d)
        except ValueError:
            pass
    for m in re.finditer(r"nam (\d{4})", f):
        y = int(m.group(1))
        add(m, m.group(0), pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31))

    rel = [
        (r"(?:tinh den|den het|cuoi ngay|het) (?:ngay )?hom nay", None),
        (r"hom nay|ngay hom nay|hnay", lambda: (a, a)),
        (r"hom qua|bua qua|hqua", lambda: (a - pd.Timedelta(days=1),) * 2),
        (r"hom kia", lambda: (a - pd.Timedelta(days=2),) * 2),
        (r"tuan nay", lambda: (a - pd.Timedelta(days=a.weekday()), a)),
        (r"tuan (?:truoc|roi|vua roi|qua)", lambda: (a - pd.Timedelta(days=a.weekday() + 7),
                                                     a - pd.Timedelta(days=a.weekday() + 1))),
        (r"thang nay", lambda: (a.replace(day=1), a)),
        (r"thang (?:truoc|roi|vua roi|qua)", lambda: ((a.replace(day=1) - pd.offsets.MonthBegin(1)),
                                                      a.replace(day=1) - pd.Timedelta(days=1))),
        (r"tu dau nam|nam nay", lambda: (pd.Timestamp(a.year, 1, 1), a)),
        (r"nam (?:ngoai|truoc|roi)", lambda: (pd.Timestamp(a.year - 1, 1, 1), pd.Timestamp(a.year - 1, 12, 31))),
        (r"quy nay", lambda: (pd.Timestamp(a.year, 3 * ((a.month - 1) // 3) + 1, 1), a)),
    ]
    for rx, fn in rel:
        for m in re.finditer(rf"\b(?:{rx})\b", f):
            if fn is None:
                if not any(m.start() < u1 and u0 < m.end() for u0, u1 in used):
                    used.append((m.start(), m.end()))
                    out.append((m.group(0) + " (tình trạng lúc hết hôm nay: cộng mọi phát sinh từ đầu dữ liệu)",
                                "start of data", _d(a)))
                continue
            s, e = fn()
            add(m, m.group(0), s, e)
    if re.search(r"\bthang nay\b", f) and re.search(r"thang (?:truoc|roi)", f):
        prev_start = a.replace(day=1) - pd.offsets.MonthBegin(1)
        prev_end = min(prev_start + pd.Timedelta(days=a.day - 1), a.replace(day=1) - pd.Timedelta(days=1))
        out.append(("tháng trước, cùng số ngày với tháng này (dùng khi so sánh tháng này với tháng trước)",
                    _d(prev_start), _d(prev_end)))

    if cal is not None and re.search(r"\btet\b", f):
        from .helpers import tet_window
        m = re.search(r"tet (?:nam )?(\d{4})", f)
        if m:
            ly = int(m.group(1))
        else:
            c = cal[pd.to_datetime(cal["date"]) == a]
            ly = int(c["lunar_year"].iloc[0]) if len(c) else a.year
            try:
                if tet_window(ly, cal)[0] > a:
                    ly -= 1
            except IndexError:
                pass
        try:
            s, e = tet_window(ly, cal)
            out.append((f"Tết {ly} (23 tháng Chạp -> mùng 10 tháng Giêng)", _d(s), _d(e)))
            n = (e - s).days + 1
            out.append((f"trước Tết {ly}, cùng {n} ngày (dùng khi so sánh với trước Tết)",
                        _d(s - pd.Timedelta(days=n)), _d(s - pd.Timedelta(days=1))))
        except IndexError:
            pass
    return out


def _tokens(s: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", fold_text(s)) if w]


ENTITY_COLUMNS = [
    ("products", "name"), ("order_lines", "product_name"), ("products", "category"), ("products", "active_ingredient"),
    ("products", "brand"), ("customers", "name"), ("customers", "group"), ("orders", "customer_name"),
    ("orders", "customer_group"), ("orders", "employee"), ("orders", "branch"), ("orders", "channel"),
    ("purchases", "supplier"), ("purchases", "product_name"), ("expenses", "category"),
]


def entity_candidates(q: str, tables: dict, max_per_col: int = 3, max_total: int = 6) -> list[tuple[str, str, float]]:
    """[(table.column, value, score)] — data values whose distinctive words appear in the question."""
    qt = set(_tokens(q))
    qf = " " + " ".join(_tokens(q)) + " "
    accented = fold_text(q) != norm_text(q)
    qn = " " + " ".join(re.split(r"[^\w]+", norm_text(q))) + " "
    found: dict[str, tuple[str, str, float]] = {}
    cols = [(t, c) for t, c in ENTITY_COLUMNS if t in tables and c in tables[t].columns]
    cols += [(n, c) for n, df in tables.items() if n.startswith("extra_") for c in df.columns
             if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    for tname, col in cols:
        vals = pd.Series(tables[tname][col].dropna().astype(str).unique())
        if len(vals) == 0 or len(vals) > 20000:
            continue
        scored = []
        for v in vals:
            vt = _tokens(v)
            key = [w for w in vt if w not in _STOP and not w.isdigit() and len(w) > 1]
            if not key:
                continue
            hits = sum(w in qt for w in key)
            hit = hits / len(key)
            phrase = (" " + " ".join(vt) + " ") in qf
            if phrase and len(key) == 1 and accented:  # short names: require the accents to match too
                phrase = (" " + " ".join(re.split(r"[^\w]+", norm_text(v))) + " ") in qn
            if phrase or (len(key) >= 2 and hit >= 0.75 and hits >= 2):
                scored.append((v, 1.0 + len(vt) / 100 if phrase else hit + len(key) / 1000))
        scored.sort(key=lambda x: -x[1])
        for v, s in scored[:max_per_col]:
            k = fold_text(v)
            if k not in found or found[k][2] < s:
                found[k] = (f"{tname}.{col}", v, s)
    res = sorted(found.values(), key=lambda x: -x[2])
    return res[:max_total]


def type_hint(q: str) -> str | None:
    f = fold_text(q)
    if re.search(r"%|phan tram|ti le|ty le|ti trong|ty trong|tang truong|tang bao nhieu phan|giam bao nhieu phan", f):
        return "percent"
    if re.search(r"\b(ngay|hom) nao\b", f):
        return "date (YYYY-MM-DD)"
    if re.search(r"\bthang nao\b", f):
        return "date (YYYY-MM)"
    if re.search(r"\btop \d|\bliet ke\b|\bnhung (mat hang|mon|san pham|khach|nhan vien)\b|\bcac (mat hang|mon|san pham|"
                 r"khach hang|nhan vien) nao\b", f):
        return "list (or table if each item needs several measures)"
    if re.search(r"\bmoi (nguoi|khach|mat hang|mon|san pham|nhan vien|nhom|ngay|thang)\b.*(bao nhieu|\bbn\b|bnhiu|"
                 r"nhieu)|(bao nhieu|\bbn\b|bnhiu).*\bmoi (nguoi|khach|mat hang|mon|san pham|nhan vien|nhom)\b", f):
        return "table (first column = name, then the measures in the order asked)"
    if re.search(r"\b(hai|ba|2|3) so\b|theo thu tu", f):
        return "list of numbers in the order asked"
    if re.search(r"\b\d+ (khach|mat hang|mon|san pham|nhan vien|nha cung cap)\w* nao\b", f):
        return "list"
    if re.search(r"\b(co|phai|dung|hay) khong\b|\bcó không\b", f) and not re.search(r"bao nhieu|may", f):
        return "boolean"
    if re.search(r"\b(mat hang|mon|san pham|khach hang|khach|nhan vien|nha cung cap|chi nhanh|nhom)\b.*\bnao\b"
                 r".*\bnhat\b|\bnao\b.*\bnhat\b", f) and not re.search(r"bao nhieu|\bbn\b|bnhiu", f):
        return "string (one name)"
    if re.search(r"bao nhieu|\bbn\b|bnhiu|\bmay\b|tong cong|\bla bao\b", f):
        return "number"
    return None


GLOSSARY = {
    "dthu": "doanh thu", "dth": "doanh thu", "bn": "bao nhiêu", "bnhiu": "bao nhiêu", "bnh": "bao nhiêu",
    "nhiu": "nhiêu", "bao nhiu": "bao nhiêu", "sl": "số lượng", "sp": "sản phẩm", "kh": "khách hàng",
    "ncc": "nhà cung cấp", "ck": "chuyển khoản", "tm": "tiền mặt", "hnay": "hôm nay", "hqua": "hôm qua",
    "bua qua": "hôm qua", "bữa qua": "hôm qua", "bữa nay": "hôm nay", "bua nay": "hôm nay", "dc": "được",
    "đc": "được", "ko": "không", "hđ": "hoá đơn", "hd": "hoá đơn", "bill": "đơn", "j": "gì", "z": "vậy",
    "mua chịu": "mua ghi nợ", "mua chiu": "mua ghi nợ",
}
_GLOSS_RX = re.compile(r"(?<!\w)(" + "|".join(sorted(map(re.escape, GLOSSARY), key=len, reverse=True)) + r")(?!\w)",
                       re.I)


def normalize_question(q: str) -> str:
    """Question with chat abbreviations expanded (dthu -> doanh thu, bn -> bao nhiêu, hqua -> hôm qua...)."""
    return _GLOSS_RX.sub(lambda m: GLOSSARY[m.group(1).lower()], q)


_UNITS = r"kg|m3|m2|lon|chai|thung|hop|goi|vien|tuyp|ly|coc|to|dia|phan|cai|chiec|cay|bao|tui|lo|vi|loc|ket|bo|cuon|met|lit"


def helper_hints(q: str, tables: dict) -> list[str]:
    """Which packaged helper computes what the question asks (helpers encode the rulebook; free-hand code on these
    topics was right 24/77 times vs 38/52 with the helper)."""
    f = fold_text(q)
    out = []
    debt = re.search(r"\bno\b|cong no|thieu tien|mua chiu|ghi no|chua tra", f)
    if debt and "orders" in tables:
        if re.search(r"tra (bot )?no|thu (hoi |duoc )?(.* )?no|no them|mua chiu them|ghi no them|tra duoc|tra bot",
                     f):
            out.append("debt added / repaid in a period -> debt_flow(start, end): added = nợ thêm / mua chịu thêm, "
                       "repaid = trả nợ / thu nợ; per customer filter on cust_name (use match()).")
        else:
            out.append("open debt at the end of a day -> d = debts('YYYY-MM-DD'): total d.remaining.sum(); number of "
                       "debtor customers d.cust_key.nunique(); 'quá/hơn N ngày' -> d[d.age_days > N]; ranking -> "
                       "d.groupby('cust_key') sum remaining, report cust_name; age buckets 0–7 / 8–15 / >15 -> list "
                       "of 3 numbers.")
    m = re.search(rf"\b(?:gia|nhap)\b.*\b(?:binh quan|trung binh)\b.*\bmot ({_UNITS})\b|"
                  rf"\bmot ({_UNITS})\b.*\b(?:binh quan|trung binh)\b", f)
    if m:
        u = m.group(1) or m.group(2)
        src = "purchases rows (nhập)" if "nhap" in f else "sales rows"
        out.append(f"average price per {u} -> avg_price(<{src} of the item and period>, '{u}').")
    elif re.search(rf"\b(?:gia|don gia)\b.*\bmot ({_UNITS})\b", f):
        u = re.search(rf"\b(?:gia|don gia)\b.*\bmot ({_UNITS})\b", f).group(1)
        out.append(f"price of one {u} -> unit_price of the item's sale lines on that day/period (lines in unit {u}; "
                   f"if several prices, avg_price(rows, '{u}')). Not the revenue.")
    elif (m := re.search(rf"\b(?:bao nhieu|bao nhiu|bn|bnhiu|may|tong cong|so|soos)\s+({_UNITS})\b", f)) \
            and not debt and not re.search(r"\bkho\b|\bton\b|con lai|con bao", f):
        u = m.group(1)
        out.append(f"quantity in {u} -> qty_in(sales(start, end), '{u}').sum() (only products whose base unit is "
                   f"{u}, lines sold in bigger units included); filter the item first if one is named.")
    if re.search(r"\bkhach\b", f) and re.search(r"nhat|top|xep hang|nhieu nhat|cao nhat|bao nhieu khach|so khach", f) \
            and not debt and "orders" in tables:
        out.append("customers -> with_customer(rows): group/count by cust_key (walk-in = NaN, excluded), report "
                   "cust_name.")
    return out


def analyze(q: str, as_of, tables: dict, cal: pd.DataFrame | None = None) -> str:
    a = pd.Timestamp(as_of)
    lines = []
    nq = normalize_question(q)
    if nq != q:
        lines.append(f"Question with abbreviations expanded: {nq}")
    hints = helper_hints(q, tables)
    if hints:
        lines.append("Use these helpers (they implement the rulebook for this kind of question):")
        lines += [f"- {h}" for h in hints]
    periods = resolve_periods(q, a, cal)
    if periods:
        lines.append("Periods (inclusive start..end, already resolved per the rulebook — use these exact dates):")
        lines += [f'- "{p}" -> {s} .. {e}' for p, s, e in periods]
    ents = entity_candidates(q, tables)
    if ents:
        lines.append("Names in the data that match words of the question (use the exact value; verify it is the "
                     "same item):")
        lines += [f"- {col} = {v!r}" for col, v, _ in ents]
    th = type_hint(q)
    if th:
        lines.append(f"Likely answer type: {th}")
    return "\n".join(lines)
