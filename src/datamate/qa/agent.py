"""Baseline question-answering agent over the semantic layer.

Per question:
    1. business context (tables, columns, value samples, meta notes) — built once per business
    2. LLM writes `solve()` (pandas) using the canonical tables + helpers + rulebook
    3. run it in a forked process with a timeout; on error / bad format feed back and retry
    4. check the run against the data (checks.py): filter literals that match nothing are auto-fixed when the
       proof is unambiguous, else fed back with candidate values; 0/empty results, impossible shares and
       contradicted "unanswerable" claims get one feedback round each (max_repairs in total)
    5. normalise the answer; the answerability gate turns low-evidence answers into abstain

If a business has no semantic layer (mapping failed), the raw sheets read with
RawStore.read_table are used instead, so the question can still be attempted.
"""
from __future__ import annotations

import json
import os
import multiprocessing as mp
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import checks, helpers, qnorm, qparse, rule_anchors
from ..core import parsers
from ..core.errors import short_traceback
from ..core.llm import LLM
from ..core.rawio import RawStore
from ..semantic.schema import TABLES

ANSWER_TYPES = ("number", "percent", "string", "date", "boolean", "list", "table", "unanswerable")

HELPERS_DOC = """Globals available in your code:
- t: dict table_name -> pandas DataFrame (the business's tables listed below)
- cal: DataFrame calendar.csv [date (Timestamp), weekday (1=Mon..7=Sun), lunar_day, lunar_month, lunar_year, lunar_leap, holiday]
- AS_OF: pd.Timestamp — "today" (hôm nay)
- pd, np, re
- sales(start, end, valid_only=True)    order_lines joined with their order header (order_date, status, customer...)
                                        for orders dated start..end INCLUSIVE (dates as "YYYY-MM-DD" strings or
                                        Timestamps), never after AS_OF. valid_only=True = not cancelled/returned.
                                        start=None means from the start of the data (cumulative "đến hết ngày X").
- orders(start, end, valid_only=True)   order headers for the same period/filters (count orders, cancellation
                                        rate with valid_only=False, customers who bought, ...)
  ALWAYS pass both start and end of the asked period, e.g. sales("2025-11-01", "2025-11-30").
- month_range(d) / week_range(d)        (start, end) Timestamps; weeks run Monday..Sunday
- tet_window(lunar_year, cal)           (start, end) of the Tết window of that lunar year
- match(values, text, k=5)              fuzzy-match a name from the question against data values -> list[str], best first (m[0] is the name)
- pct(part, whole)                      percentage rounded to 2 decimals
- qty_in(df, unit, convert=False)      quantity of each row of sales()/order_lines/purchases rows in `unit`. Default =
                                        rulebook "bán được bao nhiêu <đơn vị>": products whose BASE unit is `unit`
                                        (qty_base, lines in bigger units included), others NaN ->
                                        qty_in(sales(s, e), 'kg').sum(). convert=True also converts products that have
                                        `unit` as a larger unit (lon -> thùng). Never merge unit_conversions yourself.
- avg_price(df, unit)                   average price per `unit` = money / quantity in `unit` (sales: line_amount;
                                        purchases: qty*unit_cost) -> "giá nhập/bán bình quân một thùng"
- stock_at(at=AS_OF)                     stock in base unit of every product at the END of day `at` (rulebook: all
                                        movements from the start of data): DataFrame[product_id, product_name,
                                        base_unit, qty_base]. Use it for every "tồn kho / còn bao nhiêu trong kho /
                                        hết hàng" question; never rebuild stock from purchases/sales yourself.
- unit_factor(product_id, unit)         base units in one `unit` ('hộp' -> 30). "mấy hộp chẵn, mấy viên lẻ" ->
                                        list(divmod(qty_base, unit_factor(pid, 'hộp'))) as ints
- with_customer(df)                     adds cust_key (one customer per rulebook §3: by code, else by name ignoring
                                        spelling/phone; NaN = walk-in) and cust_name (display name). ALWAYS group,
                                        count and rank customers by cust_key and report cust_name.
- top_customers(rows, value='line_amount', n=None, ascending=False)
                                        rank customers of sales()/orders()/debts() rows: DataFrame[cust_key, cust_name,
                                        value] best first; value='orders' counts orders -> "top 3 khách mua nhiều tiền
                                        nhất" = top_customers(sales(s, e), 'line_amount', 3).cust_name.tolist()
- debts(at)                             open debt per order at the end of day `at` (rulebook §4: not cancelled,
                                        returned kept, COD/marketplace excluded, payments up to `at`, remaining > 0):
                                        order_id, order_date, payable, paid, remaining, age_days, cust_key, cust_name.
                                        Total debt = debts(X).remaining.sum(); debtors = cust_key.nunique();
                                        "quá N ngày" = age_days > N.
- debt_flow(start, end)                 per credit order: added (nợ thêm / mua chịu thêm: unpaid part of orders dated
                                        in the period after same-day payments) and repaid (trả nợ / thu nợ: payments
                                        after the order day, dated in the period); cust_key, cust_name.
- due_dates(purchase_date, due_day, n)  due dates (Timestamps) of n monthly instalments (trả góp): instalment i falls
                                        on day `due_day` of the i-th month after the purchase month. Instalments
                                        due up to a day X = sum(d <= X for d in ...); overdue = due BEFORE X and unpaid.
- fold_text(s) / norm_text(s)           scalar text normalisation (no diacritics / lowercase)
"""

def helpers_doc(withdrawn: set[str] = frozenset()) -> str:
    """HELPERS_DOC without the entries of withdrawn helpers (their rulebook definition changed)."""
    lines, keep = [], True
    for ln in HELPERS_DOC.splitlines():
        m = re.match(r"- (\w+)\(", ln)
        if m:
            keep = m.group(1) not in withdrawn
        elif ln.startswith("- ") or not ln.startswith(" "):
            keep = True
        if keep:
            lines.append(ln)
    return "\n".join(lines) + "\n"


SYSTEM = """You answer questions about one small Vietnamese business by writing pandas code over its tables.

{helpers}
## Rulebook (default meanings; the question overrides them when it says otherwise)
{rulebook}

## Your output
Think briefly, then return ONE ```python block defining `def solve():` that returns a dict:
  {{"type": <one of {types}>, "answer": <value>}}   or   {{"type": "unanswerable", "reason": "..."}}
Rules:
- Compute everything from the data; never guess numbers. Ignore records dated after AS_OF.
- Periods: use the start..end dates given in "Pre-analysis" when present (they follow the rulebook exactly);
  otherwise resolve from the question and AS_OF per the rulebook. Always filter on BOTH start and end.
- Names: use the exact data values listed in "Pre-analysis" when they fit; otherwise find them with match() against
  the real column values. Never type a product/customer name from memory; return names exactly as in the data.
  Check the match really is the same item (same brand/size); a different product (e.g. "TH true MILK" vs "Vinamilk")
  means the item is not in the data -> unanswerable.
- Revenue (doanh thu, bán được bao nhiêu tiền) = sum of order_lines.line_amount over valid orders (sales(start, end)),
  fee lines included, order-level discounts NOT subtracted — unless the rulebook/question says otherwise.
- "Giá ... một X", "bao nhiêu tiền một X" asks a PRICE per unit (unit_price, or avg_price over the period when
  several prices occur), never the revenue of the item.
- "Bao nhiêu ly/cái/hộp/kg…" asks a QUANTITY: sum qty (qty_base when the base unit is asked), never count rows.
  Count orders with nunique(order_id).
- Rankings: sort by the criterion descending, ties by product/customer code ascending (name if no code).
- Pick the type from what is asked:
  * one value -> number / percent / string / date / boolean (never wrap a single value in a list or table)
  * several values in a stated order or a top-k / "liệt kê" -> list
  * several objects each with one or more measures ("mỗi người bao nhiêu") -> table, first column = object name
  * "bao nhiêu mặt hàng/khách/đơn" asks a COUNT -> number; "ngày nào"/"tháng nào" -> date
  "Likely answer type" in Pre-analysis is a strong hint.
- number: plain JSON number; money rounded to whole VND (int). percent: number of percent rounded to 2 decimals
  (change % = (new - old) / old * 100). date: "YYYY-MM-DD" (or "YYYY-MM" for a month).
- "unanswerable" only when the tables do not record what is needed (the README/tables have no such column) or the
  period is entirely outside the data range. Stock/tồn kho needs a stock, stock_moves or purchases table, cost/lãi
  needs cost columns, debt needs payment/debt columns, groups need a filled category/group column: if that table or
  column is missing or entirely null, answer unanswerable — never rebuild it from sales lines or guess. A period inside the range with no matching rows is 0, not unanswerable.
- Code must be deterministic and fast; no file or network access; do not print large data.
"""


@dataclass
class QAResult:
    pred: dict
    log: dict = field(default_factory=dict)


def _py(v):
    """Convert numpy / pandas scalars and containers to plain JSON-able Python."""
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        return [_py(x) for x in list(v)]
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if v is pd.NaT or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def _num(x, decimals=2):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x):
        raise ValueError(f"expected a finite number, got {x!r}")
    x = round(float(x), decimals)
    return int(x) if x.is_integer() else x


def _money(x):
    """Values >= 1000 are VND amounts here (fractional quantities like kg / m3 stay far below): whole VND.
    The scorer compares exactly and the model often forgets the rounding (156304.55 vs 156305)."""
    return int(round(x)) if isinstance(x, float) and abs(x) >= 1000 else x


def normalize_answer(res) -> dict:
    """Validate and coerce solve()'s return value to a submission entry. Raises ValueError with a hint."""
    if not isinstance(res, dict) or res.get("type") not in ANSWER_TYPES:
        raise ValueError(f"solve() must return {{'type': one of {ANSWER_TYPES}, 'answer': ...}}, got {res!r}"[:500])
    t = res["type"]
    if t == "unanswerable":
        return {"type": t, "reason": str(res.get("reason", ""))}
    a = _py(res.get("answer"))
    if a is None:
        raise ValueError("answer is None/NaN — the computation found nothing; check filters, names and dates")
    if t == "number":
        a = _money(_num(a))
    elif t == "percent":
        a = _num(a)
    elif t == "string":
        if isinstance(a, list) and len(a) == 1:
            a = a[0]
        if not isinstance(a, str) or not a.strip():
            raise ValueError(f"string answer must be a non-empty str, got {a!r}")
    elif t == "date":
        if not (isinstance(a, str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", a)):
            raise ValueError(f"date answer must be 'YYYY-MM-DD' or 'YYYY-MM', got {a!r}")
    elif t == "boolean":
        if not isinstance(a, bool):
            raise ValueError(f"boolean answer must be true/false, got {a!r}")
    elif t == "list":
        if not isinstance(a, list):
            a = [a]
        a = [_money(_num(x)) if isinstance(x, (int, float)) and not isinstance(x, bool) else x for x in a]
    elif t == "table":
        if isinstance(a, list) and a and isinstance(a[0], dict):
            a = {"columns": list(a[0]), "rows": [list(r.values()) for r in a]}
        if not (isinstance(a, dict) and isinstance(a.get("columns"), list) and isinstance(a.get("rows"), list)):
            raise ValueError("table answer must be {'columns': [...], 'rows': [[...], ...]}")
        a = {"columns": [str(c) for c in a["columns"]],
             "rows": [[_money(_num(x)) if isinstance(x, (int, float)) and not isinstance(x, bool) else x for x in r]
                      for r in a["rows"]]}
    return {"type": t, "answer": a}


# ---------------------------------------------------------------- business context

def _col_summary(s: pd.Series, max_cat: int = 30, n_samples: int = 6) -> str:
    nn = s.notna().mean() if len(s) else 0
    null = f", null {1 - nn:.0%}" if nn < 1 else ""
    if pd.api.types.is_datetime64_any_dtype(s):
        d = s.dropna()
        return f"datetime{null}" + (f", {d.min().date()}..{d.max().date()}" if len(d) else "")
    if pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
        return f"bool{null}, {s.value_counts().to_dict()}"
    if pd.api.types.is_numeric_dtype(s):
        d = s.dropna()
        return f"number{null}" + (f", min {d.min():g}, median {d.median():g}, max {d.max():g}" if len(d) else "")
    vc = s.dropna().astype(str).value_counts()
    if len(vc) <= max_cat:
        return f"text{null}, ALL {len(vc)} values: " + ", ".join(f"{v!r}" for v in vc.index)
    return f"text{null}, {len(vc)} distinct, e.g. " + ", ".join(repr(v) for v in vc.index[:n_samples])


def describe_tables(tables: dict[str, pd.DataFrame], meta: dict) -> str:
    out = []
    if meta.get("industry"):
        out.append(f"Industry: {meta['industry']}")
    for n in meta.get("notes") or []:
        out.append(f"Note: {n}")
    units = (meta.get("domain") or {}).get("units") if isinstance(meta.get("domain"), dict) else None
    if units:
        out.append("Units: " + json.dumps(units, ensure_ascii=False)[:1500])
    for name, df in tables.items():
        spec = TABLES.get(name)
        desc = spec.desc if spec else (meta.get("extra_desc") or {}).get(name, "")
        out.append(f"\n### t['{name}'] — {len(df)} rows. {desc}")
        for c in df.columns:
            cdesc = spec.col(c).desc if spec and spec.col(c) else ""
            out.append(f"- {c}: {_col_summary(df[c])}" + (f"  // {cdesc}" if cdesc else ""))
    return "\n".join(out)


def readme_conventions(text: str, max_chars: int = 5000) -> str:
    """README without its file list and title (the intro, column notes and '## Quy ước' stay), capped."""
    if not text:
        return ""
    parts = re.split(r"\n(?=## )", text.strip())
    keep = [p for p in parts if not re.match(r"## File\b", p)]
    out = re.sub(r"^# .*\n+", "", "\n".join(keep)).strip()
    return out[:max_chars]


def raw_tables(biz_dir: str) -> dict[str, pd.DataFrame]:
    store = RawStore(biz_dir)
    out = {}
    for f in store.list_files():
        for sh in store.sheet_names(f):
            try:
                out[f if sh is None else f"{f}::{sh}"] = store.read_table(f, sh)
            except Exception:  # noqa: BLE001
                pass
    return out


# ---------------------------------------------------------------- sandboxed execution

def _child(conn, code, env):
    try:
        ns = dict(env)
        exec(compile(code, "<solve>", "exec"), ns)  # noqa: S102 - generated analysis code, offline sandbox
        res = ns["solve"]()
        conn.send(("ok", normalize_answer(res)))
    except Exception:  # noqa: BLE001
        conn.send(("error", short_traceback("<solve>")))
    finally:
        conn.close()


def run_code(code: str, env: dict, timeout: float = 60) -> tuple[str, object]:
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(child, code, env), daemon=True)
    p.start()
    child.close()
    if parent.poll(timeout):
        status, payload = parent.recv()
    else:
        status, payload = "error", f"TimeoutError: code ran longer than {timeout}s — use vectorised pandas"
    p.join(1)
    if p.is_alive():
        p.kill()
    return status, payload


def extract_code(text: str) -> str | None:
    m = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.S)
    return max(m, key=len) if m else None


# ---------------------------------------------------------------- agent

class Agent:
    def __init__(self, llm: LLM, rulebook: str, calendar_csv: str, max_attempts: int = 3, timeout: float = 60,
                 max_repairs: int = 2):
        self.llm = llm
        rules_all = qparse.parse_rulebook(rulebook)
        self.helper_status = rule_anchors.status(rules_all)
        helpers.RULE_FLAGS.update(rule_anchors.flags(rules_all))
        self.withdrawn = {h for h, st in self.helper_status.items() if st == "changed"}
        self.helper_rules = {h: (st, rule_anchors.rule_text(rules_all, h)) for h, st in self.helper_status.items()}
        self.system = SYSTEM.format(helpers=helpers_doc(self.withdrawn), rulebook=rulebook, types=list(ANSWER_TYPES))
        cal = pd.read_csv(calendar_csv)
        cal["date"] = pd.to_datetime(cal["date"])
        self.cal = cal
        self.max_attempts = max_attempts
        self.max_repairs = max_repairs
        self.timeout = timeout
        self._ctx: dict[str, tuple[dict, str, str]] = {}
        self.rules = qparse.parse_rulebook(rulebook) if os.environ.get("DM_RULE_HINTS", "1") != "0" else []
        self._industry: dict[str, str | None] = {}

    def set_business(self, biz_id: str, tables: dict[str, pd.DataFrame], meta: dict, source: str,
                     readme: str = ""):
        """`readme`: the shop's README.md. Its conventions (how the shop records commissions, instalments,
        warranty...) are shown to the QA model; without them it guesses definitions the README states."""
        desc = describe_tables(tables, meta)
        readme = readme_conventions(readme) if os.environ.get("DM_QA_README", "1") != "0" else ""
        if readme:
            desc += ("\n\n## Shop README (written for the RAW files; the tables above are the cleaned version, so "
                     "raw column names differ — use it for the shop's own conventions and definitions)\n" + readme)
        self._ctx[biz_id] = (tables, desc, source)
        dom = meta.get("domain") if isinstance(meta.get("domain"), dict) else {}
        self._industry[biz_id] = qparse.industry_of(
            self.rules, " ".join([str(meta.get("industry") or ""), str(dom.get("industry") or ""),
                                  readme.split("\n## ")[0]])) if self.rules else None

    def _env(self, tables, as_of):
        env = {"t": tables, "cal": self.cal, "AS_OF": as_of, "pd": pd, "np": np, "re": re,
               "fold_text": parsers.fold_text, "norm_text": parsers.norm_text}
        for n in ("valid_orders", "sales_lines", "month_range", "week_range", "tet_window", "match", "pct",
                  "due_dates"):
            env[n] = getattr(helpers, n)
        env["qty_in"] = lambda df, unit, convert=False: helpers.qty_in(tables, df, unit, convert)
        env["avg_price"] = lambda df, unit: helpers.avg_price(tables, df, unit)
        env["unit_factor"] = lambda product, unit: helpers.unit_factor(tables, product, unit)
        env["stock_at"] = lambda at=None: helpers.stock_at(tables, at, as_of)
        if "orders" in tables:
            env["with_customer"] = lambda df: helpers.with_customer(tables, df)
            env["top_customers"] = lambda df, value="line_amount", n=None, ascending=False: helpers.top_customers(
                tables, df, value, n, ascending)
            env["debts"] = lambda at=None: helpers.debts(tables, at if at is not None else as_of, as_of)
            env["debt_flow"] = lambda start=None, end=None: helpers.debt_flow(tables, start, end, as_of)
            env["sales"] = lambda start=None, end=None, valid_only=True: helpers.sales(tables, start, end, as_of,
                                                                                       valid_only)
            env["orders"] = lambda start=None, end=None, valid_only=True: helpers.orders(tables, start, end, as_of,
                                                                                         valid_only)
        return env

    def _env_checked(self, tables, as_of):
        env = self._env(tables, as_of)
        for h in self.withdrawn:
            env.pop(h, None)
        return env

    def _log(self, q, t0, pred, source, thinking=None, attempts=(), **extra) -> dict:
        last = next((a for a in reversed(attempts) if "code" in a), None)
        # Required log fields (organizer guide §3.3): id, answer, elapsed_ms, thinking, code, output.
        return {"id": q["id"], "answer": pred, "elapsed_ms": int((time.time() - t0) * 1000),
                "thinking": thinking if thinking and any(thinking) else None,
                "code": last["code"] if last else None, "output": last["output"] if last else None,
                "business_id": q["business_id"], "question": q.get("question_raw", q["question"]),
                "question_norm": q["question"], "as_of_date": q["as_of_date"],
                "source": source, "final": pred, "attempts": list(attempts), **extra}

    def answer(self, q: dict) -> QAResult:
        """One greedy solve, or self-consistency when DM_SC=K>1: K independent solves (sample 0 greedy, the rest
        sampled at DM_SC_TEMP), majority vote over the gated answers; fewer than DM_SC_MIN agreeing -> abstain."""
        t0 = time.time()
        tables, desc, source = self._ctx[q["business_id"]]
        q = {**q, "question_raw": q["question"], "question": qnorm.normalize(q["question"])}
        missing = checks.gate_tables(q["question"], tables) if source == "semantic" else None
        if missing and os.environ.get("DM_GATE_HARD", "0") != "1":
            # keyword evidence only: tell the model what the layer lacks and let it judge from the tables
            q["data_note"] = (f"Data check: this question may need {missing.split(',')[0].replace('needs ', '')}. "
                              "If the tables listed above really do not record it, answer unanswerable; if they do "
                              "(maybe under another column or an extra_* table), compute it.")
            missing = None
        early = checks.gate_source(source) or missing
        if early:
            # the shop records no such data: rulebook answer is 'unanswerable' (11/11 on train); DM_GATE_UNANS=0 -> abstain
            pred = {"type": "unanswerable", "reason": missing} if missing and early == missing \
                and os.environ.get("DM_GATE_UNANS", "1") != "0" else {"type": "abstain"}
            return QAResult(pred, self._log(q, t0, pred, source, gate=early))
        k = int(os.environ.get("DM_SC") or 1)
        if k <= 1:
            return self._solve(q, t0, tables, desc, source, 0)
        temp = float(os.environ.get("DM_SC_TEMP") or 0.7)
        need = int(os.environ.get("DM_SC_MIN") or (k // 2 + 1))
        runs = [self._solve(q, t0, tables, desc, source, i, temperature=0.0 if i == 0 else temp) for i in range(k)]
        final, votes = vote([r.pred for r in runs], need)
        log = dict(runs[0].log)
        log.update(answer=final, final=final, elapsed_ms=int((time.time() - t0) * 1000), sc_votes=votes,
                   sc_samples=[{"pred": r.pred, "pre_gate": r.log.get("pre_gate"), "gate": r.log.get("gate"),
                                "code": r.log.get("code")} for r in runs])
        return QAResult(final, log)

    def _solve(self, q, t0, tables, desc, source, sample: int, temperature: float = 0.0) -> QAResult:
        think = os.environ.get("DM_QA_THINK", "0") == "1"
        max_out = int(os.environ.get("DM_QA_MAX_TOKENS") or (16000 if think else 4000))
        as_of = pd.Timestamp(q["as_of_date"])
        pre = qparse.analyze(q["question"], as_of, tables, self.cal, self.rules,
                             self._industry.get(q["business_id"]), self.helper_rules)
        if q.get("data_note"):
            pre = (pre + "\n" if pre else "") + q["data_note"]
        periods = qparse.resolve_periods(q["question"], as_of, self.cal)
        user = (f"## Tables of business {q['business_id']} (source: {source})\n{desc}\n\n"
                f"## Question\nAS_OF (hôm nay) = {as_of.date()} ({as_of.day_name()})\n{q.get('question_raw', q['question'])}"
                + (f"\n(viết rõ: {q['question']})" if q.get("question_raw", q["question"]) != q["question"] else "")
                + (f"\n\n## Pre-analysis (computed by code from the question)\n{pre}" if pre else "")
                + ("\n\nNote: raw sheets only — sales()/orders()/valid_orders()/sales_lines() are NOT available; "
                   "compute directly on t[...] following the rulebook." if "orders" not in tables else ""))
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        env = self._env_checked(tables, as_of)
        attempts, thinking = [], []
        pred = {"type": "abstain"}
        issues: list[checks.ValueIssue] = []
        value_only_prev: str | None = None  # code of the last 'only fix the values' round
        errors_left, repairs_left = self.max_attempts, self.max_repairs
        i = 0
        while errors_left > 0:
            i += 1
            res = self.llm.chat(messages, temperature=temperature, max_tokens=max_out, seed=42 + sample,
                                thinking=True if think else None)
            thinking.append(res.reasoning)
            messages.append({"role": "assistant", "content": res.content})
            code = extract_code(res.content)
            if code is None:
                errors_left -= 1
                attempts.append({"attempt": i, "result": "no_code"})
                messages.append({"role": "user", "content": "Return the code in one ```python block."})
                continue
            status, payload = run_code(code, env, self.timeout)
            attempts.append({"attempt": i, "code": code, "status": status, "output": payload})
            if status != "ok":
                errors_left -= 1
                messages.append({"role": "user", "content": f"Running solve() failed:\n```\n{payload}\n```\n"
                                                            "Fix it and return the full code."})
                continue
            pred = checks.coerce(q["question"], payload)
            if value_only_prev is not None:
                if checks.skeleton(code) != checks.skeleton(value_only_prev):
                    attempts[-1]["skeleton_changed"] = True
                value_only_prev = None
            issues = checks.value_checks(code, tables)
            if issues:
                fixed, applied = checks.auto_fix(code, issues)
                if applied:
                    st2, out2 = run_code(fixed, env, self.timeout)
                    attempts.append({"attempt": i, "auto_fix": applied, "code": fixed, "status": st2,
                                     "output": out2})
                    if st2 == "ok":
                        code, pred = fixed, checks.coerce(q["question"], out2)
                        issues = checks.value_checks(code, tables)
            probs = checks.review(q["question"], pred, issues, tables, periods) if repairs_left > 0 else []
            if not probs:
                break
            repairs_left -= 1
            attempts[-1]["review"] = probs
            only_values = bool(issues) and len(probs) == 1
            if only_values:
                value_only_prev = code
            keep = ("\nOnly the listed values are wrong: change those literals (or the column they are compared to) "
                    "and keep the rest of the code as it is." if only_values else "")
            messages.append({"role": "user", "content": f"solve() returned {json.dumps(pred, ensure_ascii=False)}"
                                                        "\nProblems found by checking it against the data:\n"
                                                        + "\n".join(probs) + keep
                                                        + "\nReturn the full corrected code."})
        reason = checks.gate(pred, source, issues, q["question"])
        final = {"type": "abstain"} if reason else pred
        log = self._log(q, t0, final, source, thinking, attempts, gate=reason, pre_gate=pred,
                        value_issues=[it.describe() for it in issues])
        return QAResult(final, log)


def _vote_key(pred: dict) -> str:
    a = pred.get("answer")
    if pred.get("type") == "unanswerable":
        return "unanswerable"
    if isinstance(a, str):
        a = parsers.fold_text(a) or a
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        a = round(float(a), 2)
    return json.dumps([pred.get("type"), a], ensure_ascii=False, sort_keys=True, default=str)


def vote(preds: list[dict], need: int) -> tuple[dict, dict]:
    """Majority answer among non-abstain predictions; abstain when the top answer has fewer than `need` votes."""
    groups: dict[str, list[dict]] = {}
    for p in preds:
        if p.get("type") != "abstain":
            groups.setdefault(_vote_key(p), []).append(p)
    votes = {key: len(v) for key, v in groups.items()}
    if not groups:
        return {"type": "abstain"}, votes
    best = max(groups, key=lambda key: (len(groups[key]), -list(groups).index(key)))
    return (groups[best][0] if len(groups[best]) >= need else {"type": "abstain"}), votes


def load_business(work: str | Path, split: str | Path, biz_id: str) -> tuple[dict, dict, str]:
    """Semantic layer if it was built successfully, else raw sheets."""
    from ..semantic.mapper import load_semantic
    meta_p = Path(work) / biz_id / "meta.json"
    if meta_p.exists() and json.loads(meta_p.read_text()).get("ok"):
        tables, meta = load_semantic(work, biz_id)
        return tables, meta, "semantic"
    return raw_tables(str(Path(split) / "data" / biz_id)), {}, "raw"
