"""Post-run checks on an answer produced by LLM-written `solve()` code, and the answerability gate.

Ideas borrowed from NVIDIA auto-ontology (sql_value_validation / entity_coverage), adapted to pandas code:

* value_checks(code, tables)  every string literal the code compares against a column (==, !=, in, isin, eq,
  str.contains / startswith) is checked against the real column values, using the same text transform the code
  applies (exact / lower / fold_text). A literal that matches nothing filters everything out, so the result is
  silently 0/empty. For each missing literal we collect candidate stored values, searched in a fixed order:
  trimmed exact -> case/diacritic-insensitive -> substring -> fuzzy (same column first, then other text columns).
* auto_fix(code, issues)  substitutes a literal only when the proof is unambiguous: the original matches no row
  and exactly one stored value differs from it only by case / diacritics / spacing. Everything else goes back to
  the LLM as feedback with the candidates as evidence.
* review(...)  the list of problems worth one feedback round (missing values, zero/empty result, impossible share,
  an "unanswerable" claim contradicted by the data range).
* gate(...)  final answer -> abstain when the evidence says the answer is probably wrong. Measured on 300 train
  answers: a 0 result was right 1/22 times; answers from raw sheets (no semantic layer) 14/68.

Configuration (env): DM_GATE=0 disables the gate, DM_ABSTAIN_RAW=0 keeps answering from raw sheets.
"""
from __future__ import annotations

import ast
import difflib
import os
import re
from dataclasses import dataclass, field

import pandas as pd

from ..core.parsers import fold_text
from ..semantic.schema import ORDER_STATUS, PAYMENT_METHOD

_SKIP_VALUES = set(ORDER_STATUS) | set(PAYMENT_METHOD)
_DATE_RX = re.compile(r"^\d{4}-\d{1,2}(-\d{1,2})?([ T].*)?$|^\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?$")
_REGEX_CHARS = re.compile(r"[|()\[\]^$*+?\\{}]")
_CHANGE_RX = re.compile(r"tang|giam|thay doi|\bso (?:voi|vs|v|thang|tuan|nam|quy|cung|ky)\b|hon hay kem|nhieu hay it|cao hay thap|"
                        r"len hay xuong|\blen\b|\bxuong\b|chenh|\bhon\b|\bkem\b|\bsut\b|tang truong")  # growth / diff % may be <0 or >100
_SHARE_RX = re.compile(r"chiem|ti trong|ty trong|ti le|ty le|dong gop|% (doanh thu|tong)")  # explicit share wording only
_UNANS_DATA_RX = re.compile(r"no (valid )?(data|orders|sales|records|rows)|not (found|available|present)|outside|"
                            r"does not (contain|include|cover)|khong co du lieu|no .* in ", re.I)


def _enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in ("0", "false", "no", "off")


@dataclass
class ValueIssue:
    literal: str
    column: str
    op: str                      # eq | contains
    mode: str                    # exact | lower | fold
    candidates: list[tuple[str, str]] = field(default_factory=list)   # (column, stored value)
    nodes: list[ast.Constant] = field(default_factory=list)

    def describe(self) -> str:
        how = {"exact": "", "lower": " (after lower())", "fold": " (after fold_text)"}[self.mode]
        rel = "equal to" if self.op == "eq" else "containing"
        s = f"- {self.literal!r}: no value of column {self.column!r} is {rel} it{how}."
        if self.candidates:
            s += " Stored values that may be meant: " + "; ".join(
                f"{c} = {v!r}" for c, v in self.candidates)
        else:
            s += " No similar stored value in any text column."
        return s


# ---------------------------------------------------------------- literal extraction

def _str_consts(node: ast.AST, names: dict[str, ast.Constant]) -> list[ast.Constant]:
    """String constants a node evaluates to: a literal, a list/tuple/set of literals, or a simple variable."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [c for e in node.elts for c in _str_consts(e, names)]
    if isinstance(node, ast.Name) and node.id in names:
        return [names[node.id]]
    return []


def _column_and_mode(node: ast.AST) -> tuple[str | None, str]:
    """The single column referenced by an expression (df['col'] / df.col) and the text transform applied to it."""
    cols, mode = set(), "exact"
    for n in ast.walk(node):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
            cols.add(n.slice.value)
        elif isinstance(n, ast.Name) and n.id in ("fold_text",):
            mode = "fold"
        elif isinstance(n, ast.Name) and n.id == "norm_text" and mode == "exact":
            mode = "lower"
        elif isinstance(n, ast.Attribute):
            if n.attr in ("lower", "casefold") and mode == "exact":
                mode = "lower"
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("map", "apply", "replace") \
                and not (n.args and isinstance(n.args[0], ast.Name) and n.args[0].id in ("fold_text", "norm_text", "str")):
            return None, mode  # col.map(lookup) compares the looked-up value, not the column itself
    if len(cols) != 1 and isinstance(node, ast.Attribute) and not isinstance(node.value, ast.Attribute):
        cols = {node.attr}
    return (next(iter(cols)) if len(cols) == 1 else None), mode


def _simple_assignments(tree: ast.AST) -> dict[str, ast.Constant]:
    seen: dict[str, list[ast.AST]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            seen.setdefault(n.targets[0].id, []).append(n.value)
    return {k: v[0] for k, v in seen.items()
            if len(v) == 1 and isinstance(v[0], ast.Constant) and isinstance(v[0].value, str)}


def extract_literals(code: str) -> list[tuple[ast.Constant, str, str, str]]:
    """[(constant node, column, op, mode)] for string literals compared against a column."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names = _simple_assignments(tree)
    # A literal that matches nothing only empties the result in a positive filter (==, in, isin), not in a negated one.
    negated = {id(n.operand) for n in ast.walk(tree) if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.Invert,
                                                                                                       ast.Not))}
    out = []
    for n in ast.walk(tree):
        if id(n) in negated:
            continue
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Eq, ast.In)):
            left, right = n.left, n.comparators[0]
            for side, other in ((right, left), (left, right)):
                consts = _str_consts(side, names)
                col, mode = _column_and_mode(other)
                if consts and col:
                    out += [(c, col, "eq", mode) for c in consts]
                    break
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.args:
            attr, base = n.func.attr, n.func.value
            if attr in ("isin", "eq"):
                op = "eq"
            elif attr in ("contains", "startswith", "endswith") and isinstance(base, ast.Attribute) \
                    and base.attr == "str":
                op = "contains"
                base = base.value
            else:
                continue
            col, mode = _column_and_mode(base)
            if op == "contains" and any(k.arg == "case" and isinstance(k.value, ast.Constant) and k.value.value is False
                                        for k in n.keywords) and mode == "exact":
                mode = "lower"
            if col:
                out += [(c, col, op, mode) for c in _str_consts(n.args[0], names)]
    return out


# ---------------------------------------------------------------- checking against the data

def _tf(mode: str):
    if mode == "fold":
        return lambda s: fold_text(s) or ""
    if mode == "lower":
        return lambda s: str(s).strip().lower()
    return lambda s: str(s)


def _text_values(tables: dict, col: str | None = None) -> dict[str, pd.Series]:
    """{'table.col': unique non-null string values} for one column name, or for every text column."""
    out = {}
    for tname, df in tables.items():
        for c in df.columns:
            if (col is None or c == col) and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])):
                vals = pd.Series(df[c].dropna().astype(str).unique())
                if 0 < len(vals) <= 50000:
                    out[f"{tname}.{c}"] = vals
    return out


def _exists(vals: pd.Series, literal: str, op: str, mode: str) -> bool:
    f = _tf(mode)
    lit = f(literal)
    tv = vals.map(f)
    return bool((tv == lit).any()) if op == "eq" else bool(tv.str.contains(lit, regex=False).any())


def _candidates(cols: dict[str, pd.Series], literal: str, k: int = 5) -> list[tuple[str, str]]:
    fl = fold_text(literal) or ""
    found: list[tuple[str, str, float]] = []
    for cname, vals in cols.items():
        folded = vals.map(lambda v: fold_text(v) or "")
        for v in vals[vals.str.strip() == literal.strip()]:
            found.append((cname, v, 4.0))
        for v in vals[folded == fl]:
            found.append((cname, v, 3.0))
        if len(fl) >= 3:
            for v in vals[folded.str.contains(fl, regex=False)].head(k):
                found.append((cname, v, 2.0 + len(fl) / max(len(fold_text(v) or "x"), 1)))
            inside = folded[(folded.str.len() >= 4) & folded.map(lambda x: x in fl)]
            for v in vals[inside.index].head(k):
                found.append((cname, v, 1.5 + len(fold_text(v) or "") / max(len(fl), 1) / 2))
        for m in difflib.get_close_matches(fl, folded.tolist(), n=k, cutoff=0.75):
            v = vals[folded == m].iloc[0]
            found.append((cname, v, 1.0 + difflib.SequenceMatcher(None, fl, m).ratio()))
    best: dict[tuple[str, str], float] = {}
    for c, v, s in found:
        best[(c, v)] = max(best.get((c, v), 0), s)
    return [cv for cv, _ in sorted(best.items(), key=lambda x: -x[1])[:k]]


def value_checks(code: str, tables: dict) -> list[ValueIssue]:
    """Literals that match no stored value of their column, with candidate stored values as evidence."""
    colnames = {c for df in tables.values() for c in df.columns}
    issues: dict[tuple[str, str, str, str], ValueIssue] = {}
    all_text = None
    for node, col, op, mode in extract_literals(code):
        lit = node.value
        if (len(lit.strip()) < 2 or lit in colnames or lit in _SKIP_VALUES or _DATE_RX.match(lit.strip())
                or (op == "contains" and _REGEX_CHARS.search(lit))):
            continue
        cols = _text_values(tables, col)
        if not cols:
            continue
        if any(_exists(v, lit, op, mode) for v in cols.values()):
            continue
        key = (lit, col, op, mode)
        if key not in issues:
            cands = _candidates(cols, lit)
            if len(cands) < 3:
                if all_text is None:
                    all_text = _text_values(tables)
                others = {k: v for k, v in all_text.items() if k not in cols}
                cands += [c for c in _candidates(others, lit, 3) if c not in cands]
            issues[key] = ValueIssue(lit, col, op, mode, cands[:5])
        issues[key].nodes.append(node)
    return list(issues.values())


def auto_fix(code: str, issues: list[ValueIssue]) -> tuple[str, list[dict]]:
    """Replace literals whose only candidate in the same column differs just by case/diacritics/spacing."""
    edits = []
    for it in issues:
        same = [v for c, v in it.candidates if c.split(".", 1)[1] == it.column
                and (fold_text(v) or "") == (fold_text(it.literal) or "")]
        if it.op != "eq" or len(set(same)) != 1:
            continue
        for node in it.nodes:
            edits.append((node, same[0], it.literal))
    if not edits:
        return code, []
    lines = code.splitlines(keepends=True)
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln.encode()))
    raw = code.encode()
    done, applied = set(), []
    for node, new, old in sorted(edits, key=lambda e: (e[0].lineno, e[0].col_offset), reverse=True):
        if node.end_lineno is None or (node.lineno, node.col_offset) in done:
            continue
        done.add((node.lineno, node.col_offset))
        a = offs[node.lineno - 1] + node.col_offset
        b = offs[node.end_lineno - 1] + node.end_col_offset
        raw = raw[:a] + repr(new).encode() + raw[b:]
        applied.append({"from": old, "to": new})
    return raw.decode(), applied


def skeleton(code: str) -> str | None:
    """AST of the code with every constant blanked: equal skeletons = only literals changed."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant):
            n.value = None
    return ast.dump(tree, annotate_fields=False)


# ---------------------------------------------------------------- review + gate

def _data_range(tables: dict) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    o = tables.get("orders")
    if o is None or "order_date" not in o:
        return None
    d = pd.to_datetime(o["order_date"], errors="coerce").dropna()
    return (d.min(), d.max()) if len(d) else None


def _is_zero(pred: dict) -> bool:
    a = pred.get("answer")
    if pred.get("type") in ("number", "percent"):
        return isinstance(a, (int, float)) and not isinstance(a, bool) and a == 0
    if pred.get("type") == "list":
        return isinstance(a, list) and not a
    if pred.get("type") == "table":
        return isinstance(a, dict) and not a.get("rows")
    return False


_NUM_WORDS = {"hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}
_NULLISH = {"<na>", "nan", "none", "null", "nat", ""}


def expected_count(question: str) -> int | None:
    """Number of items the question explicitly asks for ("ghi 3 số", "top 5", "3 khách ... nào")."""
    f = fold_text(question) or ""
    m = (re.search(r"\btop (\d+)\b", f) or re.search(r"\b(\d+|hai|ba|bon|nam|sau) so\b", f)
         or re.search(r"\b(\d+|hai|ba|bon|nam) (?:khach|mat hang|mon|san pham|nhan vien|nha cung cap)\w*\b.*\b(?:nao|la ai|ai)\b", f))
    if not m:
        return None
    n = m.group(1)
    n = int(n) if n.isdigit() else _NUM_WORDS.get(n)
    return n if n and 1 < n <= 20 else None


# A percentage is asked ('bao nhiêu %', 'phần trăm', 'tỉ trọng', 'chiếm'); a '%' right after a digit is part of a
# product name ('Efticol 0,9%', 'kem 5%') and does not count.
_ASKS_PCT = re.compile(r"phan tr|ti trong|ty trong|ti le|ty le|\bchiem\b|(?<![\d.,])%")


def asks_percent(question: str) -> bool:
    return bool(_ASKS_PCT.search(fold_text(question) or ""))


_COUNT_RX = re.compile(r"\b(so|bao nhieu|bnhiu|bn|may) (don|khach|lan|hoa don)\b")
_SUM_ALL_RX = re.compile(r"cong (moi|tat ca|het|ca cac|lai)|cong cac mon|tong cong cac")


def coerce(question: str, pred: dict) -> dict:
    """A table returned where the question asks a plain list of numbers -> list of its last numeric column.
    A plain number returned where the question asks a percentage -> percent (the value is already the percent)."""
    a = pred.get("answer")
    f = fold_text(question) or ""
    if pred.get("type") == "string" and isinstance(a, str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", a.strip()):
        return {"type": "date", "answer": a.strip()}  # a month / day answer is a date
    if pred.get("type") == "number" and isinstance(a, (int, float)) and not isinstance(a, bool) \
            and asks_percent(question):
        return {"type": "percent", "answer": round(float(a), 2)}
    if pred.get("type") == "table" and isinstance(a, dict) and a.get("rows") and _SUM_ALL_RX.search(f):
        last = [r[-1] for r in a["rows"] if r]
        if last and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in last):
            return {"type": "number", "answer": sum(last)}  # 'cộng mỗi món' = one total over all items
    if pred.get("type") == "list" and isinstance(a, list) and a and all(isinstance(x, list) and len(x) >= 2 for x in a) \
            and len({len(x) for x in a}) == 1:
        return {"type": "table", "answer": {"columns": [f"c{i}" for i in range(len(a[0]))], "rows": a}}
    if pred.get("type") == "table" and isinstance(a, dict) and a.get("rows") \
            and re.search(r"\b(\d+|hai|ba|bon|nam) so\b|theo thu tu|lan luot", f):
        last = [r[-1] for r in a["rows"] if r]
        if last and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in last):
            return {"type": "list", "answer": last}
    return pred


def _nullish_items(pred: dict) -> bool:
    a = pred.get("answer")
    items = a if pred.get("type") == "list" and isinstance(a, list) else \
        [r[0] for r in a.get("rows", []) if r] if pred.get("type") == "table" and isinstance(a, dict) else []
    return any(x is None or (isinstance(x, str) and (fold_text(x) or "") in _NULLISH | {"khach le", "kl"})
               for x in items)


def review(question: str, pred: dict, issues: list[ValueIssue], tables: dict,
           periods: list[tuple[str, str, str]]) -> list[str]:
    """Problems with a successful run that deserve one feedback round (empty list = accept)."""
    probs = []
    if issues:
        probs.append("These filter values match nothing in the data, so the filter keeps no rows:\n"
                     + "\n".join(i.describe() for i in issues)
                     + "\nUse the exact stored value that means the same thing (same item, brand, size), "
                       "or the right column. If none of them is the asked item, return unanswerable.")
    if pred.get("type") == "number" and isinstance(pred.get("answer"), float) \
            and pred["answer"] != int(pred["answer"]) and _COUNT_RX.search(fold_text(question) or "") \
            and not asks_percent(question):
        probs.append(f"The question asks HOW MANY (a count), but the answer {pred['answer']} is not a whole number. "
                     "Return the count itself (len / nunique), not a rate or average.")
    if _is_zero(pred) and not issues:
        probs.append("solve() returned 0 / empty. That is rarely the true answer: check that every filter value "
                     "really occurs in its column (print-free: compare against the values listed in the tables "
                     "summary or use match()), that the period and date column are right and that the status filter "
                     "does not drop everything. If after checking the answer really is 0, return the same code.")
    f = fold_text(question) or ""
    a = pred.get("answer")
    n = expected_count(question)
    got = len(a) if pred.get("type") == "list" and isinstance(a, list) else \
        len(a.get("rows", [])) if pred.get("type") == "table" and isinstance(a, dict) else None
    if n and got is not None and got != n:
        probs.append(f"The question asks for {n} items but the answer has {got}. Cover every asked period/item "
                     "(e.g. each month of a range, all top-k).")
    if _nullish_items(pred):
        probs.append("The answer lists a null / walk-in entry ('<NA>', None, 'khách lẻ'). Walk-in customers and "
                     "missing names are not ranked objects: drop null names before ranking (dropna) and take the "
                     "next ones.")
    if pred.get("type") == "percent" and isinstance(a, (int, float)) and _SHARE_RX.search(f) \
            and not _CHANGE_RX.search(f) and not 0 <= a <= 100:
        probs.append(f"The question asks a share of a total, but the result is {a}% (must be within 0..100). "
                     "The part must be a subset of the whole computed over the same period and filters.")
    if pred.get("type") == "unanswerable" and _UNANS_DATA_RX.search(str(pred.get("reason", ""))):
        rng = _data_range(tables)
        inside = [p for p in periods if rng and p[1][:1].isdigit()
                  and rng[0] <= pd.Timestamp(p[1]) and pd.Timestamp(p[2]) <= rng[1]]
        if inside:
            probs.append(f"You answered unanswerable ({pred.get('reason')!r}), but orders cover "
                         f"{rng[0].date()}..{rng[1].date()}, which includes the asked period. 'unanswerable' is only "
                         "for data the shop does not record (missing table/column). Re-check names with match() and "
                         "the filters; a period with no matching rows is 0.")
    return probs


def _has(tables: dict, table: str, col: str | None = None) -> bool:
    df = tables.get(table)
    if df is None or not len(df):
        return False
    return col is None or (col in df and df[col].notna().any())


# (folded question pattern, data the question needs, test on the tables). Without that data any number the model
# produces is a guess, and a wrong answer costs -1 against 0 for abstaining.
_NEEDS = [
    (r"nhap hang|nha cung cap|\bncc\b|gia nhap|phieu nhap|nhap bao nhieu|lan nhap|tien nhap",
     "purchases", lambda t: _has(t, "purchases")),
    (r"san pham|mat hang|\bmon\b|ten hang|ban chay|\bsku\b|nhom hang|loai hang|hang nao|mon nao|\bmat hang\b",
     "product lines", lambda t: _has(t, "order_lines", "product_name")),
    (r"cong no|con no|\bno\b|tra gop|ghi no|chua tra|chua thanh toan|(?<!dong )(?<!lai )\bgop (thang|ky)",
     "debt / payments", lambda t: _has(t, "payments") or _has(t, "orders", "debt_remaining")
     or _has(t, "orders", "paid_amount") or _has(t, "orders", "payment_method")),
    (r"nhan vien|thu ngan|nguoi ban|\bnv\b|\btho (nao|gioi|lanh nghe|chinh)",
     "employee", lambda t: _has(t, "orders", "employee")),
    (r"\bkenh\b|online|shopee|lazada|tiktok|zalo|facebook|tai quay|san tmdt|\bgrab\b|tren san|san thuong mai",
     "sales channel", lambda t: _has(t, "orders", "channel")),
    (r"chi nhanh|\bco so (\d|mot|hai|chinh)|cua hang so \d",
     "branch", lambda t: _has(t, "orders", "branch")),
    (r"tien mat|chuyen khoan|vi dien tu|momo|quet the|phuong thuc thanh toan|hinh thuc thanh toan|\bcod\b",
     "payment method", lambda t: _has(t, "orders", "payment_method") or _has(t, "payments", "method")),
    (r"loi nhuan|lai gop|lai rong|tien lai|gia von|\blo von\b|\bbi lo\b|\b(?:lai|loi) hay lo\b|"
     r"(?:^|\b(?:duoc|kiem|co|la|thi|quan|tiem|shop) )lai (?:bao nhieu|duoc bao)",
     "cost", lambda t: _has(t, "order_lines", "cost_per_unit") or _has(t, "products", "cost_price")
     or _has(t, "purchases")),
]
_NEEDS = [(re.compile(p), what, test) for p, what, test in _NEEDS]


def gate_tables(question: str, tables: dict) -> str | None:
    """Reason to abstain before any LLM call when the question needs data this layer does not have, or None."""
    if not _enabled("DM_GATE") or not _enabled("DM_GATE_TABLES"):
        return None
    f = fold_text(question) or ""
    for rx, what, test in _NEEDS:
        if rx.search(f) and not test(tables):
            return f"needs {what}, not in this business's tables"
    return None


def gate_source(source: str) -> str | None:
    """Reason to abstain before spending any LLM call on the question, or None."""
    if _enabled("DM_GATE") and source == "raw" and _enabled("DM_ABSTAIN_RAW"):
        return "raw sheets only (no semantic layer)"
    return None


def gate(pred: dict, source: str, unresolved: list[ValueIssue], question: str) -> str | None:
    """Reason to abstain instead of answering, or None to keep the answer."""
    if not _enabled("DM_GATE") or pred.get("type") == "abstain":
        return None
    if gate_source(source):
        return gate_source(source)
    if pred.get("type") == "unanswerable":
        return None
    if _is_zero(pred):
        return "result is 0 / empty"
    if unresolved:
        return "filter values not found in data: " + ", ".join(repr(i.literal) for i in unresolved[:3])
    a = pred.get("answer")
    f = fold_text(question) or ""
    if pred.get("type") == "percent" and isinstance(a, (int, float)) and _SHARE_RX.search(f) \
            and not _CHANGE_RX.search(f) and not 0 <= a <= 100:
        return f"share out of range ({a}%)"
    return None
