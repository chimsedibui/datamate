"""Robust parsers for messy Vietnamese small-business data.

These helpers are exposed to LLM-generated loader code, so every function is
vectorised over pandas Series and tolerant of mixed types in one column.

Public helpers:
    parse_money(s)      "55.000đ", "25k", "1,01tr", "1,500.0000", 22000 -> float
    parse_number(s)     generic numeric parse (same rules as money, no suffix scaling needed)
    parse_date(s, default_year=None)   Excel serials, dd/mm/yyyy, "1 thg 5 2025", "1/5", ISO
    parse_datetime(s, default_year=None)
    parse_qty_unit(s)   "5 chai" -> (5.0, "chai")
    norm_text(s)        NFC + lowercase + collapse spaces (keeps diacritics)
    fold_text(s)        norm_text + strip diacritics (for fuzzy matching)
    ffill_group(df, key, cols)   fill blanks in follow-up rows of a group from its first row
"""
from __future__ import annotations

import math
import re
import unicodedata
from decimal import ROUND_HALF_UP, Decimal
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

_EXCEL_EPOCH = datetime(1899, 12, 30)


_NULL_STR = {"<na>", "<nat>"}  # str() of a pandas null; never real text ("nan"/"None" may be)


def norm_text(x):
    if x is None or x is pd.NA or x is pd.NaT or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, str) and x.strip().lower() in _NULL_STR:
        return None
    s = unicodedata.normalize("NFC", str(x)).strip().lower()
    return re.sub(r"\s+", " ", s)


def fold_text(x):
    s = norm_text(x)
    if s is None:
        return None
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d")


def _is_blank(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip() in ("", "-", "nan", "NaN", "None", "null"):
        return True
    return x is pd.NaT


_MULT = [
    (re.compile(r"(tri[eệ]u|tr|m)$"), 1_000_000),
    (re.compile(r"(ngh[iì]n|ng[aà]n|k)$"), 1_000),
]
_INFIX_MULT = re.compile(r"^(\d+)(?:\s*(tri[eệ]u|tr|ngh[iì]n|ng[aà]n|k)\s*|(m))(\d+)$")  # '3 m3' is a volume, not 3.3 triệu


def _num_core(s: str, scaled: bool = False) -> float | None:
    """Parse a bare numeric string with ambiguous VN/EN separators.

    scaled=True (a 'tr'/'k' suffix followed): a single separator is a decimal mark, '1,156tr' = 1.156 triệu.
    """
    s = s.strip()
    if not s:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.startswith("-") or s.startswith("−"):
        neg, s = True, s[1:]
    s = s.replace(" ", "").replace(" ", "")
    if not re.fullmatch(r"[\d.,]+", s):
        return None
    has_dot, has_comma = "." in s, "," in s
    if has_dot and has_comma:
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        th = "," if dec == "." else "."
        s = s.replace(th, "").replace(dec, ".")
    elif has_dot or has_comma:
        sep = "." if has_dot else ","
        parts = s.split(sep)
        # "55.000", "1.250.000" -> thousands; "1.5", "1,01" -> decimal
        if len(parts) > 2 or (not scaled and len(parts) == 2 and len(parts[1]) == 3 and parts[0] not in ("", "0")):
            s = s.replace(sep, "")
        else:
            s = s.replace(sep, ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _money_one(x):
    if _is_blank(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        return float(x)
    s = norm_text(x)
    s = re.sub(r"(vnđ|vnd|đồng|đ|d)$", "", s).strip()
    s = s.replace("₫", "").strip()
    m = _INFIX_MULT.match(s)  # '1tr090' = 1.090 triệu, '1k5' = 1.5 nghìn
    if m:
        whole, unit, unit_m, frac = m.groups()
        unit = unit or unit_m
        mult = 1_000_000 if unit[0] in "tm" else 1_000
        return round((int(whole) + int(frac) / 10 ** len(frac)) * mult, 6)
    mult = 1
    for rx, m in _MULT:
        if rx.search(s):
            s = rx.sub("", s).strip()
            mult = m
            break
    v = _num_core(s, scaled=mult > 1)
    return np.nan if v is None else round(v * mult, 6)


def parse_money(s):
    """Parse money-like values. Returns float Series (NaN when unparseable)."""
    if not isinstance(s, pd.Series):
        return _money_one(s)
    return s.map(_money_one).astype(float)


_NUM_UNIT = re.compile(r"^([-−]?[\d.,]+)\s*([^\d\s.,].*)$")
_MONEY_SUFFIX = re.compile(r"^(tri[eệ]u|tr|ngh[iì]n|ng[aà]n|k|vnđ|vnd|đồng|đ|₫)\s*\d*$")


def _number_one(x):
    """Like money, but a trailing non-money unit is dropped: '3 m3' -> 3, '5 m' -> 5, '2 thùng' -> 2."""
    if not _is_blank(x) and isinstance(x, str):
        m = _NUM_UNIT.match(norm_text(x) or "")
        if m and not _MONEY_SUFFIX.match(m.group(2)):
            v = _num_core(m.group(1))
            return np.nan if v is None else v
    return _money_one(x)


def parse_number(s):
    """Quantities / factors. Money suffixes still scale ('1tr5'); unit suffixes are dropped."""
    if not isinstance(s, pd.Series):
        return _number_one(s)
    return s.map(_number_one).astype(float)


_QTY_RX = re.compile(r"^\s*([-−]?[\d.,]+)\s*(.*)$")


def _qty_one(x):
    if _is_blank(x):
        return (np.nan, None)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return (float(x), None)
    m = _QTY_RX.match(str(x))
    if not m:
        return (np.nan, None)
    v = _num_core(m.group(1))
    unit = m.group(2).strip() or None
    return (np.nan if v is None else v, unit)


def parse_qty_unit(s):
    """'5 chai' -> (5.0, 'chai'). Series input returns DataFrame[qty, unit]."""
    if not isinstance(s, pd.Series):
        return _qty_one(s)
    out = s.map(_qty_one)
    return pd.DataFrame(out.tolist(), index=s.index, columns=["qty", "unit"])


_VN_MONTH = re.compile(r"^(\d{1,2})\s*(?:thg|tháng|th)\s*(\d{1,2})(?:[\s,/]+(\d{2,4}))?$")
_DMY = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})(?:[/\-.](\d{2,4}))?$")
_YMD = re.compile(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$")


def _year(y, default_year):
    if y is None:
        return default_year
    y = int(y)
    return y + 2000 if y < 100 else y


def _datetime_one(x, default_year=None):
    if _is_blank(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, (datetime, date)):
        return pd.Timestamp(x)
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        v = float(x)
        if 20000 < v < 80000:
            return pd.Timestamp(_EXCEL_EPOCH + timedelta(days=v))
        return pd.NaT
    s = norm_text(x)
    s = re.sub(r"^(ngày|ngay)\s+", "", s)
    time_part = None
    m = re.search(r"[ t](\d{1,2}:\d{2}(?::\d{2})?)$", s)
    if m:
        time_part = m.group(1)
        s = s[: m.start()].strip()
    d = None
    try:
        if re.fullmatch(r"\d{5}(\.\d+)?", s):
            return _datetime_one(float(s))
        if (m := _YMD.match(s)):
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        elif (m := _DMY.match(s)) or (m := _VN_MONTH.match(s)):
            y = _year(m.group(3), default_year)
            if y is None:
                return pd.NaT
            d = datetime(y, int(m.group(2)), int(m.group(1)))
    except ValueError:
        return pd.NaT
    if d is None:
        try:
            return pd.Timestamp(s)
        except Exception:
            return pd.NaT
    if time_part:
        hh, mm, *ss = (int(p) for p in time_part.split(":"))
        d = d.replace(hour=hh, minute=mm, second=ss[0] if ss else 0)
    return pd.Timestamp(d)


def parse_datetime(s, default_year=None):
    """Parse mixed date/datetime values (day-first). `default_year` fills dates like '1/5'."""
    if not isinstance(s, pd.Series):
        return _datetime_one(s, default_year)
    if isinstance(default_year, pd.Series):
        vals = [_datetime_one(v, dy) for v, dy in zip(s, default_year)]
    else:
        vals = [_datetime_one(v, default_year) for v in s]
    return pd.Series(pd.to_datetime(vals), index=s.index)


def parse_date(s, default_year=None):
    out = parse_datetime(s, default_year)
    return out.dt.normalize() if isinstance(out, pd.Series) else (out.normalize() if out is not pd.NaT else out)


def ffill_group(df: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """Fill blank cells of `cols` within each `key` group from earlier rows of that group.

    Rulebook §8: blank cells in follow-up rows of an order equal the first row's value.
    If `key` itself is blank on follow-up rows, it is forward-filled first.
    """
    df = df.copy()
    df[key] = df[key].replace(r"^\s*$", np.nan, regex=True).ffill()
    for c in cols:
        df[c] = df[c].replace(r"^\s*$", np.nan, regex=True)
        df[c] = df.groupby(key, sort=False)[c].ffill()
    return df


def year_from_text(s: str) -> int | None:
    """Extract a 4-digit year from a sheet/file name such as 'Thang 02 2025' or 'T7-2025'."""
    m = re.search(r"(20\d{2})", str(s))
    return int(m.group(1)) if m else None


# Folded (accent-free) keywords that unambiguously signal an order status. Checked in order: a label such as
# 'không giao được' is cancelled before any delivering keyword is considered.
STATUS_HINTS = [
    ("cancelled", re.compile(r"\b(huy|cancel\w*|khong giao duoc|giao that bai|that bai|failed)\b")),
    ("returned", re.compile(r"\b(tra hang|tra lai|tra ve|hoan tra|hoan hang|doi tra|return\w*|refund\w*)\b")),
    ("delivering", re.compile(r"\b(dang giao|chua giao|dang van chuyen|dang gui|dang ship|cho giao|dang xu ly|cho xu ly|"
                              r"shipping|delivering|pending|processing)\b")),
    ("completed", re.compile(r"\b(thanh cong|hoan thanh|hoan tat|da giao|giao thanh cong|da lay|xong|"
                             r"success\w*|complete\w*|done|delivered)\b")),
]


def status_hint(label):
    """Status implied by keywords in a raw label/note, or None (e.g. 'Đã hủy đơn' -> 'cancelled')."""
    if _is_blank(label):
        return None
    f = fold_text(label)
    for status, rx in STATUS_HINTS:
        if rx.search(f):
            return status
    return None


def map_status(raw, mapping=None, default=None):
    """Map raw status labels: keyword hints first, then `mapping` (label -> status, matched on norm_text),
    else `default` (None keeps unknown labels null so validation reports them).
    Common 'completed' labels ('thành công', 'hoàn thành', 'đã lấy', 'xong'...) are keywords too; others ('ok')
    must be listed in `mapping`."""
    m = {norm_text(k): v for k, v in (mapping or {}).items()}

    def one(x):
        h = status_hint(x)
        if h:
            return h
        if _is_blank(x):
            return m.get("", default)
        return m.get(norm_text(x), default)
    if not isinstance(raw, pd.Series):
        return one(raw)
    return raw.map(one).astype(object)


def round_half_up(x: float, decimals: int = 0) -> float:
    """Round half away from zero on the decimal value as written (rulebook: "nửa đồng làm tròn lên").
    Python's round() rounds halves to even (round(1234.5) == 1234) and works on the binary float
    (round(2.675, 2) == 2.67); repr() gives the shortest decimal that reads back as the same float."""
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))
