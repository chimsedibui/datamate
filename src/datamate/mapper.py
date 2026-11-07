"""LLM-generated mapping: raw business files -> canonical semantic tables.

Flow per business (run once, cached in the work dir):
    1. profile the folder (README + raw rows + column samples)
    2. LLM writes `load(raw)` Python code using our helper library
    3. execute it, finalize dtypes, validate
    4. on exceptions / validation errors, send feedback and retry
    5. one review round: LLM sees the load report and may fix subtle mistakes
The accepted code, report and tables (parquet) are saved under <work>/<business_id>/.
"""
from __future__ import annotations

import json
import re
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import parsers
from .llm import LLM
from .profile import profile_business
from .rawio import RawStore
from .schema import schema_prompt
from .validate import finalize, validate

HELPERS_DOC = '''
`raw` (RawStore for this business folder):
- raw.list_files() -> list[str]                file names (README excluded; names already utf-8 repaired)
- raw.sheet_names(file) -> list                xlsx sheet names / sqlite table names / [None] for csv
- raw.read_raw(file, sheet=None) -> DataFrame  header=None, cells as stored (Excel dates may be datetime or serial int)
- raw.read_table(file, sheet=None, header_rows=1, skip_rows=None) -> DataFrame
      skip_rows=None auto-detects the header row per sheet (blank/title rows above it skipped) — prefer this;
      else skip `skip_rows` rows. The next `header_rows` rows become column names (multi-row headers flattened:
      lowest label wins, duplicates become 'Upper|Lower'), empty rows dropped, strings NFC-stripped
- raw.read_sql(file, query) -> DataFrame       for .sqlite files

Parsers (already imported, all accept a Series and return a Series):
- parse_money(s)            '55.000đ', '25k', '1,01tr', '1,500.0000', '(5.000)', 22000 -> float VND
- parse_number(s)           same rules, for quantities/factors
- parse_datetime(s, default_year=None)  day-first; Excel serials, '01/02/2025', '1 thg 5 2025', '2025-02-01 06:00',
                            '1/5' (needs default_year; may be a Series aligned with s, e.g. year from sheet name)
- parse_date(s, default_year=None)      parse_datetime normalised to midnight
- parse_qty_unit(s)         '5 chai' -> DataFrame[qty, unit]
- norm_text(x) / fold_text(x)  scalar: NFC+lower+collapse spaces / same without diacritics (for matching labels)
- ffill_group(df, key, cols)   fill blank cells of `cols` in follow-up rows of each `key` group (rulebook §8)
- year_from_text(s)         4-digit year inside a sheet/file name, e.g. 'Thang 02 2025' -> 2025
Also available: pd, np, re.
'''

RULES = '''
Mapping rules:
1. Read the README first; its column descriptions and conventions override your guesses from the samples.
2. Map EVERY sales file/sheet (all months, all numbered parts _1, _2...). Concatenate same-layout sheets.
   Detect title rows / multi-row headers from the raw rows shown in the profile.
3. Order-level cells left blank on follow-up lines of the same order inherit the first line (use ffill_group).
4. orders.status must be one of completed / cancelled / returned / delivering, mapped from the shop's labels
   (e.g. 'đã xong','đã lấy','ok','hoàn thành' -> completed; 'hủy','huỷ','đã hủy đơn' -> cancelled;
   'trả hàng','hàng trả về','đổi trả','hoàn' -> returned; 'đang giao','chờ giao' -> delivering).
   Inspect ALL VALUES in the profile and map every one. If there is no status column, the status is in the note
   column; a note that does not mention a status means completed. Keep the original text in status_raw.
   An order with ANY returned item (even partial; e.g. a separate negative-quantity return row, a return slip)
   is a returned order: set its status to 'returned'. Do not keep negative return rows as order_lines
   (put them in the `returns` table if useful).
5. One row per order in `orders` (order_id unique). If the file has no order code, build a stable one from the
   fields that identify an order per the README (e.g. date+time+customer) and say so in meta['notes'].
6. Walk-in customers ('Khách lẻ', 'khách vãng lai', blank) -> customer_id and customer_name null.
   customer_id only when the shop really records customer codes.
7. order_lines.qty_base = qty × factor of the line's unit (unit_conversions / README conversion notes); the base
   unit has factor 1. If the file already has a converted quantity column, use it.
8. order_lines.line_amount = line total after line-level discount (thành tiền). If absent, compute
   qty*unit_price - line_discount. Never subtract order-level vouchers from lines.
9. Rows that are not merchandise but order-level adjustments (voucher, giảm giá đơn, phí ship, VAT rows) go to the
   matching orders column (order_discount positive, shipping_fee, vat), NOT to order_lines.
   'Phí phục vụ' (service fee) lines stay in order_lines (finalize marks is_fee).
10. Keep names exactly as written in the data (full Vietnamese diacritics); do not translate or clean them.
11. Never invent data. If the shop does not record something, omit that column/table. Do not derive a column from
    another unless the README says how.
12. Map payment methods to cash / transfer / ewallet / card / cod / debt / installment / other
    (ghi nợ, công nợ -> debt; COD, giao hàng thu tiền -> cod; trả góp -> installment).
13. Map other useful data (catalogue, conversions, customers, payments, purchases, stock, batches, batch_sales,
    price_history, expenses, returns, deliveries) to their canonical tables when present.
14. Data that fits no canonical table (repair tickets, trade-ins, installment schedules, shift cash reports,
    loyalty points, staff lists...) -> return it cleaned as an extra table named 'extra_<short_name>' with
    snake_case columns and parsed types, and describe it in meta['extra_desc'][name].
15. Any paid transaction is an order: a repair visit, service ticket, booking, appointment or rental is one
    `orders` row, its parts/services are `order_lines`. Omit `orders` only if the business records no paid
    transactions at all; then return everything as extra tables.
16. Code must be deterministic and run in < 60 s. No file writes, no network, no printing large data.
'''

OUTPUT_SPEC = '''
Return ONE ```python code block defining:

def load(raw):
    ...
    return tables, meta

- tables: dict table_name -> DataFrame using ONLY the canonical column names (omit unknown/unrecorded columns)
- meta: dict with keys
    industry: short label (tạp hoá, cà phê, nhà hàng, nhà thuốc, vật liệu xây dựng, mẹ và bé, sửa xe, điện thoại, ...)
    has_customer_codes: bool   (customers identified by code, else by name)
    line_no_from_file: bool    (True only if the file has a real line-number column)
    data_end: 'YYYY-MM-DD' if the README states the last data day, else None
    notes: list of short strings — quirks that matter when answering questions (units, how debt is recorded...)
    extra_desc: dict extra_table_name -> description (may be empty)
'''

SYSTEM = f"""You are a senior data engineer. You write pandas code that maps one small Vietnamese business's raw
export files into a fixed canonical schema (the semantic layer used to answer business questions).

## Helper API
{HELPERS_DOC}
## Canonical schema
{{schema}}
{RULES}
## Business rulebook (defines meanings; the loader must keep enough information for these rules)
{{rulebook}}
{OUTPUT_SPEC}"""


def extract_code(text: str) -> str | None:
    m = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.S)
    return max(m, key=len) if m else None


def columns_overview(store: RawStore) -> str:
    """One line per distinct column layout, listing the sheets that share it."""
    layouts: dict[tuple, list[str]] = {}
    for f in store.list_files():
        for sh in store.sheet_names(f):
            try:
                cols = tuple(store.read_table(f, sh).columns)
            except Exception as e:  # noqa: BLE001
                cols = (f"<unreadable: {e}>",)
            layouts.setdefault(cols, []).append(f if sh is None else f"{f}::{sh}")
    lines = []
    for cols, where in layouts.items():
        shown = where if len(where) <= 4 else where[:2] + [f"… {len(where)} total"]
        lines.append(f"- {shown}: {list(cols)}")
    return "\n".join(lines)


def run_loader(code: str, store: RawStore):
    ns = {"pd": pd, "np": np, "re": re, "raw": store}
    for n in ("parse_money", "parse_number", "parse_datetime", "parse_date", "parse_qty_unit",
              "norm_text", "fold_text", "ffill_group", "year_from_text"):
        ns[n] = getattr(parsers, n)
    exec(compile(code, "<loader>", "exec"), ns)  # noqa: S102 - generated loader runs offline, read-only data
    tables, meta = ns["load"](store)
    meta = dict(meta or {})
    meta.setdefault("extra_desc", {})
    return tables, meta


class Mapper:
    def __init__(self, llm: LLM, rulebook: str, work_dir: str | Path, max_attempts: int = 5, review: bool = True):
        self.llm = llm
        self.rulebook = rulebook
        self.work = Path(work_dir)
        self.max_attempts = max_attempts
        self.review = review

    def system_prompt(self) -> str:
        return SYSTEM.replace("{schema}", schema_prompt()).replace("{rulebook}", self.rulebook)

    def build(self, biz_id: str, biz_dir: str, force: bool = False) -> dict:
        out = self.work / biz_id
        out.mkdir(parents=True, exist_ok=True)
        if not force and (out / "loader.py").exists() and (out / "meta.json").exists():
            return json.loads((out / "meta.json").read_text())
        store = RawStore(biz_dir)
        profile = profile_business(store)
        (out / "profile.md").write_text(profile)
        messages = [{"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": f"Business id: {biz_id}\n\n{profile}\n\nWrite load(raw)."}]
        log = []
        best = None
        reviewed = False
        for attempt in range(1, self.max_attempts + 1):
            t = time.time()
            res = self.llm.chat(messages, max_tokens=16000)
            code = extract_code(res.content)
            entry = {"attempt": attempt, "llm_s": res.usage.get("latency_s"), "cached": res.cached,
                     "reasoning": res.reasoning}
            messages.append({"role": "assistant", "content": res.content})
            if code is None:
                if reviewed and best is not None and "OK" in res.content.upper():
                    entry["result"] = "review_ok"
                    log.append(entry)
                    break
                messages.append({"role": "user", "content": "Return the full code in one ```python block."})
                entry["result"] = "no_code"
                log.append(entry)
                continue
            try:
                tables, meta = run_loader(code, store)
                tables = finalize(tables, meta)
                errs, report = validate(tables, meta)
            except Exception:  # noqa: BLE001
                tb = traceback.format_exc(limit=6)
                entry.update(result="exception", error=tb[-3000:])
                log.append(entry)
                hint = ("\nExact columns returned by raw.read_table(file, sheet) with default arguments:\n"
                        + columns_overview(store)) if "KeyError" in tb else ""
                messages.append({"role": "user", "content":
                                 f"The code raised an exception:\n```\n{tb[-3000:]}\n```\n{hint}\n"
                                 "Fix it and return the full code."})
                continue
            entry.update(result="errors" if errs else "valid", errors=errs, run_s=round(time.time() - t, 1))
            log.append(entry)
            if not errs:
                best = (code, tables, meta, report)
                if not self.review or reviewed:
                    break
                reviewed = True
                messages.append({"role": "user", "content":
                                 "The code ran and passed validation. Load report:\n" + report +
                                 "\n\nCheck the report against the profile and README: row counts per month, status "
                                 "mapping, revenue scale, quantities, walk-in handling, all files/sheets included. "
                                 "If something is wrong return the full corrected code; otherwise reply exactly OK."})
                continue
            messages.append({"role": "user", "content":
                             "Validation errors:\n- " + "\n- ".join(errs) + "\n\nLoad report:\n" + report +
                             "\n\nFix the code and return the full code."})
        (out / "build_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str))
        (out / "messages.json").write_text(json.dumps(messages, ensure_ascii=False, indent=1))
        if best is None:
            meta = {"biz_id": biz_id, "ok": False}
            (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
            return meta
        code, tables, meta, report = best
        (out / "loader.py").write_text(code)
        (out / "report.txt").write_text(report)
        tdir = out / "tables"
        tdir.mkdir(exist_ok=True)
        for name, df in tables.items():
            df.to_parquet(tdir / f"{name}.parquet", index=False)
        meta.update(biz_id=biz_id, ok=True, tables=sorted(tables))
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
        return meta


def load_semantic(work_dir: str | Path, biz_id: str) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load the cached canonical tables + meta of one business."""
    out = Path(work_dir) / biz_id
    meta = json.loads((out / "meta.json").read_text())
    tables = {p.stem: pd.read_parquet(p) for p in sorted((out / "tables").glob("*.parquet"))}
    return tables, meta
