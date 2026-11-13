"""Question normaliser: chat-style Vietnamese -> the plain wording the rest of the pipeline is written for.

Shop owners type "Dthu th7/26 bnhiu z?", "T4/2025 ...", "8-2025 ...", "tháng ni", "hôm ni". Every deterministic step
(period resolution, helper hints, type coercions, gates) matches plain words such as "tháng 7/2026", "doanh thu",
"bao nhiêu", so the question is rewritten once, up front:

    normalize(question) -> str     (unchanged when nothing applies; product / customer names are left alone)

Rules are token-level and conservative: an abbreviation is only expanded as a whole word, month shorthands only when
the numbers form a valid month (and year).
"""
from __future__ import annotations

import re

_I = re.IGNORECASE

_MONTH_WORD = r"(?:tháng|thang|thg|th|t)"


def _year(y: str) -> int | None:
    v = int(y)
    if len(y) == 2:
        v += 2000
    return v if 2015 <= v <= 2035 else None


_OWNED = re.compile(r"(?:tháng|thang|thg|th|quý|quy|qúy|q|ngày|ngay)s?\.?\s*$", _I)


def _owned(m: re.Match) -> bool:
    """A bare 'M/YYYY' that already belongs to a preceding word ('tháng 7/2026', 'quý 2/2026')."""
    return bool(_OWNED.search(m.string[:m.start("m")]))


def _month_sub(m: re.Match) -> str:
    mo, y = int(m.group("m")), _year(m.group("y"))
    if not (1 <= mo <= 12) or y is None or _owned(m):
        return m.group(0)
    return f"{m.group('pre') or ''}tháng {mo}/{y}"


def _month_word_sub(m: re.Match) -> str:
    mo, y = int(m.group("m")), _year(m.group("y"))
    if not (1 <= mo <= 12) or y is None:
        return m.group(0)
    return f"tháng {mo}/{y}"


# "th7/2026", "T10/2025", "Th 6/25", "thg 5-2026", "tháng 3/26"
_MONTH_WORD_YEAR = re.compile(rf"\b{_MONTH_WORD}\.?\s*(?P<m>\d{{1,2}})\s*[/-]\s*(?P<y>\d{{4}}|\d{{2}})\b(?![/-]?\d)", _I)
# bare "06/2026", "8-2025" (not the tail of a full date d/m/yyyy, not 'quý 2/2026')
_MONTH_BARE = re.compile(r"(?P<pre>^|[\s(,;:])(?P<m>\d{1,2})\s?[/-]\s?(?P<y>\d{4})\b(?![/-]?\d)")
_QUARTER = re.compile(r"\bq\.?\s?([1-4])\s*[/.-]\s*(\d{4}|\d{2})\b", _I)
_MONTH_NAMES = {"một": 1, "mot": 1, "giêng": 1, "hai": 2, "ba": 3, "tư": 4, "bốn": 4, "bon": 4, "sáu": 6,
                "bảy": 7, "bay": 7, "tám": 8, "tam": 8, "chín": 9, "chin": 9, "mười": 10, "muoi": 10,
                "mười một": 11, "muoi mot": 11, "mười hai": 12, "muoi hai": 12, "chạp": 12}
_MONTH_NAMED = re.compile(r"\b(?:tháng|thang)\s+(" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True))
                          + r")\b(?!\s*(?:tháng|thang|ngày|ngay|tuần|tuan|năm \d|nam \d|\d))", _I)
# bare "8/26", "11/25": a month/2-digit-year only when the second number cannot be a month
_MONTH_YY = re.compile(r"(?P<pre>^|[\s(,;:])(?P<m>\d{1,2})/(?P<y>\d{2})\b(?![/-]?\d)")
# "th6", "thg12" glued to the number, no year
_MONTH_ONLY = re.compile(r"\b(?:thg|th)\.?(\d{1,2})\b(?![/-]?\d)", _I)


def _month_yy_sub(m: re.Match) -> str:
    mo, yy = int(m.group("m")), int(m.group("y"))
    if 1 <= mo <= 12 and 13 <= yy <= 35 and not _owned(m):
        return f"{m.group('pre')}tháng {mo}/{2000 + yy}"
    return m.group(0)


def _month_only_sub(m: re.Match) -> str:
    return f"tháng {int(m.group(1))}" if 1 <= int(m.group(1)) <= 12 else m.group(0)


# (pattern, replacement) applied in order, whole words, case-insensitive
_PHRASES = [
    # relative time, regional / shorthand
    (r"\b(?:tháng|thang|thg|th)\s+(?:này|nay|ni|nì|ni`)\b", "tháng này"),
    (r"\b(?:tháng|thang|thg|th)\s+(?:trc|trước|truoc|rồi|roi|rùi|vừa rồi|vua roi)\b", "tháng trước"),
    (r"\b(?:tuần|tuan)\s+(?:này|nay|ni|nì)\b", "tuần này"),
    (r"\b(?:tuần|tuan)\s+(?:trc|trước|truoc|rồi|roi|rùi|vừa rồi|vua roi)\b", "tuần trước"),
    (r"\b(?:năm|nam)\s+(?:ni|nì)\b", "năm nay"),
    (r"\b(?:từ |tu )?(?:đầu|dau) (?:năm|nam) (?:tới|đến|toi|den) (?:chừ|giờ|nay|chu|gio)\b", "từ đầu năm đến hôm nay"),
    (r"\b(?:từ |tu )?(?:đầu|dau) (?:tháng|thang) (?:tới|đến|toi|den) (?:chừ|giờ|nay|chu|gio)\b",
     "từ đầu tháng này đến hôm nay"),
    (r"\b(?:quý|quy)\s+(?:ni|nì|nay|này)\b", "quý này"),
    (r"\b(?:hôm|hom|bữa|bua|bửa)\s+(?:ni|nì|nay|này)\b|\bhni\b|\bhnay\b|\bhn\b(?=\s|$|[,.?!])", "hôm nay"),
    (r"\b(?:hôm|hom|bữa|bua)\s+(?:wa|qa|qua)\b|\bhqua\b|\bhwa\b", "hôm qua"),
    (r"\b(?:tết|tet)\s+(?:âm lịch|am lich|nguyên đán|nguyen dan|ta)\b", "Tết"),
    # typos of fixed phrases
    (r"\b(?:tỉ|tỷ|ti|ty)\s+(?:nệ|ne)\b", "tỉ lệ"),
    (r"\bbao\s+(?:nhiu|nhju|nhiu`|nhiêu|nhieu)\b", "bao nhiêu"),
    # abbreviations
    (r"\bd\.?thu\b|\bdoanh\s+số\b", "doanh thu"),
    (r"\bbn(?:hi[eê]?u|hju|h)?\b|\bbnhiu\b", "bao nhiêu"),
    (r"\b(?:đh|dh)\b", "đơn hàng"),
    (r"\b(?:o đờ|ô đờ|o đơ|ọt đơ|order|orders)\b", "đơn hàng"),
    (r"\b(?:hđ|hd)\b", "hoá đơn"),
    (r"\bbq\b", "bình quân"),
    (r"\btb\b", "trung bình"),
    (r"\bsl\b", "số lượng"),
    (r"\bsp\b", "sản phẩm"),
    (r"\bncc\b", "nhà cung cấp"),
    (r"\bnv\b", "nhân viên"),
    (r"\bkh\b", "khách hàng"),
    (r"\bhsd\b", "hạn sử dụng"),
    (r"\bck\b", "chuyển khoản"),
    (r"\btm\b", "tiền mặt"),
    # teen-code spellings
    (r"\b(?:mik|mjk|mk|mìk)\b", "mình"),
    (r"(?<![\d.,])\b(?:ko|kô|hok|hông|hong|khum|hem)\b", "không"),
    (r"\b(?:dc|đc|dk|đk)\b", "được"),
    (r"\b(?:nhứt|nhut|nhứt)\b", "nhất"),
    (r"\b(?:kòn|kon)\b", "còn"),
    (r"\bkó\b", "có"),
    (r"\b(?:kủa|kua)\b", "của"),
    (r"\bcko\b", "cho"),
    (r"\btrog\b", "trong"),
    (r"\b(?:bít|bik|bjk)\b", "biết"),
    (r"\b(?:jùm|jum|dùm|dum|zùm|gium)\b", "giùm"),
    (r"\b(?:gồi|gùi|rùi)\b", "rồi"),
    (r"\b(?:ngừi|ng)\b(?=\s)", "người"),
    (r"\btiềng\b", "tiền"),
    (r"\btíh\b", "tính"),
    (r"\bnượt\b", "lượt"),
    (r"\b(?:wán|wan)\b", "quán"),
    (r"\bsốp\b", "shop"),
    (r"\bmí\b", "mấy"),
    # regional question words: "ngày mô" = ngày nào, "bán răng" left alone
    (r"(?<=\s)mô\b(?=\s|$|[,.?!])", "nào"),
]
_PHRASES = [(re.compile(p, _I), r) for p, r in _PHRASES]


def normalize(q: str) -> str:
    """Plain-wording version of a question (see module docstring). Idempotent."""
    if not isinstance(q, str) or not q.strip():
        return q
    s = q
    s = _QUARTER.sub(lambda m: f"quý {m.group(1)}/{_year(m.group(2)) or m.group(2)}", s)
    s = _MONTH_WORD_YEAR.sub(_month_word_sub, s)
    s = _MONTH_BARE.sub(_month_sub, s)
    s = _MONTH_NAMED.sub(lambda m: f"tháng {_MONTH_NAMES[m.group(1).lower()]}", s)
    s = _MONTH_YY.sub(_month_yy_sub, s)
    s = _MONTH_ONLY.sub(_month_only_sub, s)
    for rx, rep in _PHRASES:
        s = rx.sub(rep, s)
    return re.sub(r"[ \t]{2,}", " ", s).strip()
