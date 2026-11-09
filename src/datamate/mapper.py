"""LLM-generated mapping: raw business files -> canonical semantic tables.

Flow per business (run once, cached in the work dir):
    1. profile the folder (README + raw rows + column samples)
    2. LLM writes `load(raw)` Python code using our helper library
    3. execute it, sanitize unambiguous structural slips (validate.sanitize), finalize dtypes, validate
    4. on exceptions / validation errors, send feedback and retry
    5. one review round: LLM sees the load report and may fix subtle mistakes
The accepted code, report and tables (parquet) are saved under <work>/<business_id>/.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import loader_examples, parsers
from .llm import LLM
from .profile import profile_business
from .rawio import RawStore
from .schema import CORE_TABLES, schema_prompt
from .validate import finalize, repair_tables, sanitize, validate

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
- map_status(s, mapping)    status from raw labels: keyword hints (hủy/huỷ/cancel -> cancelled; trả hàng/hoàn trả/
                            đổi trả -> returned; đang giao/chờ giao -> delivering) first, then your `mapping` dict
                            (label -> status) for the rest, e.g. {'xong': 'completed', 'ok': 'completed'};
                            unknown labels stay null. finalize() also re-applies the keyword hints to status_raw.
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
   Use `status = map_status(status_raw, {...})` listing every non-keyword label from the profile (labels can
   differ across sheets/files); never `.fillna('completed')` — an unmapped label must stay null so validation
   reports it. If there is no status column, the status is in the note
   column; a note that does not mention a status means completed. Keep the original text in status_raw.
   An order with ANY returned item (even partial; e.g. a separate negative-quantity return row, a return slip)
   is a returned order: set its status to 'returned'. Do not keep negative return rows as order_lines
   (put them in the `returns` table if useful).
5. One row per order in `orders` (order_id unique). If the file has no order code, build a stable one from the
   fields that identify an order per the README (e.g. date+time+customer) and say so in meta['notes']. If nothing
   identifies an order (no code, no time, no customer), every row is its own order: order_id = row number.
   Never put product or status into a synthesised order_id. Build `orders` with a groupby/dedup on a COPY;
   order_lines must keep every line of the line-level frame.
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
16. Build every non-core table (anything except orders / order_lines) inside its own
    `try: ... except Exception as e: meta.setdefault('table_errors', {})[name] = repr(e)` so a mistake there never
    breaks orders / order_lines. Do not wrap orders / order_lines.
17. Never drop duplicate sales rows yourself (e.g. `df = df.drop_duplicates()` on the raw sales sheet):
    finalize() applies rulebook §8. Deduplicating a catalogue / projected columns is fine.
18. In raw.read_sql quote identifiers that are SQL keywords ("group", "order", ...).
19. Code must be deterministic and run in < 60 s. No file writes, no network, no printing large data.
20. Files/sheets that are copies of other data (archive, history, backup, bản sao, lưu trữ, _old, _bak, (1)) are
    mapped only when the README says they hold sales not present elsewhere; never add them on top of the same
    sales (double counting).
21. purchases.amount is the LINE amount (qty × unit_cost after line discount); a receipt total repeated on every
    line must not go into amount.
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
    order_codes_from_file: bool (True only if order_id comes from a real order-code column; False if synthesised)
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


def lint(code: str) -> list[str]:
    """Static checks on loader code for mistakes validation cannot see."""
    errs = []
    m = re.search(r"\b(\w+)\s*=\s*\1\.drop_duplicates\(\s*\)", code)
    if m:
        errs.append(f"loader: `{m.group(0)}` removes legitimate repeated sales rows (identical rows of different "
                    "orders, or of a table without order codes, are separate sales); remove it — finalize() applies "
                    "rulebook §8 to order_lines")
    if re.search(r"\.drop_duplicates\((?:[^)]*,\s*)?inplace\s*=\s*True", code) and \
            not re.search(r"\.drop_duplicates\(\s*subset", code):
        errs.append("loader: `drop_duplicates(inplace=True)` on raw rows removes legitimate repeated sales; remove it")
    for m in re.finditer(r"\b(\w+)\s*=\s*\1\.drop_duplicates\(\s*subset\s*=\s*\[?\s*['\"]order_id['\"]\s*\]?\s*\)", code):
        if re.search(rf"\border_lines\w*\s*=\s*{m.group(1)}\b", code[m.end():]):
            errs.append(f"loader: `{m.group(0)}` keeps only the first line of each order, then order_lines is built "
                        f"from `{m.group(1)}` — build order_lines from the full line-level frame and dedup only orders")
    if re.search(r"dropna\(\s*subset\s*=\s*\[?\s*['\"]order_id['\"]\s*\]?\s*\)", code) and "ffill" not in code:
        errs.append("loader: `dropna(subset=['order_id'])` drops follow-up lines whose order code is blank — by rulebook "
                    "§8 a blank cell in a follow-up row repeats the row above: forward-fill the order code first "
                    "(ffill_group / ffill), never drop those rows")
    return errs


def run_loader(code: str, store: RawStore):
    ns = {"pd": pd, "np": np, "re": re, "raw": store}
    for n in ("parse_money", "parse_number", "parse_datetime", "parse_date", "parse_qty_unit",
              "norm_text", "fold_text", "ffill_group", "year_from_text", "map_status", "status_hint"):
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

    def domain_summary(self, profile: str) -> dict | None:
        """One short LLM call before writing the loader: industry, core entities, units and quirks.

        Units/conversions are the main payoff (thùng/lon, xe 4m3, bao 50kg): wrong conversions were a leading cause
        of wrong quantity/price answers. Failure is harmless (returns None)."""
        prompt = ("Read this small Vietnamese business's README and file profile. Return ONLY a JSON object with keys:\n"
                  '  "industry": short label,\n'
                  '  "core_entities": list of what is sold/recorded (orders, products, customers, debts, stock...),\n'
                  '  "units": list of {"product_or_group": str, "base_unit": str, "larger_units": {unit: factor}} '
                  "stated or evident in the data (only what is written; empty list if none),\n"
                  '  "quirks": list of short notes that matter for mapping (status labels, money formats, split files, '
                  "header rows, walk-in markers, debt recording)\n\n" + profile[:20000])
        try:
            res = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=1500, thinking=False)
            m = re.search(r"\{.*\}", res.content, flags=re.S)
            d = json.loads(m.group(0)) if m else None
            return d if isinstance(d, dict) else None
        except Exception:  # noqa: BLE001
            return None

    def system_prompt(self) -> str:
        return SYSTEM.replace("{schema}", schema_prompt()).replace("{rulebook}", self.rulebook)

    def fingerprint(self, biz_dir: str) -> str:
        """Hash of the build inputs: data files + README and rulebook.

        The build model and mapping code / prompt changes are deliberately excluded (they would force paid rebuilds of every
        business); use --retry-failed or --force after improving the mapper.
        """
        h = hashlib.sha256()
        h.update(self.rulebook.encode())
        for f in sorted(os.listdir(biz_dir)):
            p = Path(biz_dir) / f
            if p.is_file():
                h.update(f.encode())
                h.update(hashlib.sha256(p.read_bytes()).digest())
        return h.hexdigest()[:16]

    def build(self, biz_id: str, biz_dir: str, force: bool = False, retry_failed: bool = False) -> dict:
        """Build (or reuse) the semantic layer. Reused only if the fingerprint matches.

        Failed builds are cached as well (re-running the same prompts gives the same failure) unless
        retry_failed. If only non-core tables are wrong, those tables are dropped and the build is
        accepted as partial (meta['dropped_tables']).
        """
        out = self.work / biz_id
        out.mkdir(parents=True, exist_ok=True)
        fp = self.fingerprint(biz_dir)
        meta_p = out / "meta.json"
        if not force and meta_p.exists():
            old = json.loads(meta_p.read_text())
            if old.get("fingerprint") == fp and (
                    (old.get("ok") and (out / "loader.py").exists()) or (not old.get("ok") and not retry_failed)):
                return old
        for stale in ("meta.json", "loader.py", "report.txt"):  # keep the previous build for manual recovery
            if (out / stale).exists():
                (out / stale).replace(out / f"{stale}.prev")
        if (out / "tables").exists():
            shutil.rmtree(out / "tables")
        store = RawStore(biz_dir)
        profile = profile_business(store)
        domain = self.domain_summary(profile)
        (out / "profile.md").write_text(profile)
        dom = ("Domain summary (first read of README + profile; the README stays authoritative):\n"
               + json.dumps(domain, ensure_ascii=False) + "\n\n") if domain else ""
        examples = (loader_examples.pick(store, industry=(domain or {}).get("industry"))
                    if os.environ.get("BUILD_EXAMPLES", "1") != "0" else "")
        messages = [{"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": f"Business id: {biz_id}\n\n{dom}{profile}\n\n{examples}"
                                                "Write load(raw) for THIS business."}]
        log = []
        best = None
        partial = None
        lenient = None  # core tables structurally valid, only data-quality warnings left
        reviewed = False
        seen: dict[str, int] = {}  # code -> first attempt that produced it
        t_start = time.time()
        max_s = float(os.environ.get("BUILD_MAX_S", "1500"))
        max_tokens = int(os.environ.get("BUILD_MAX_TOKENS", "12000"))
        for attempt in range(1, self.max_attempts + 1):
            t = time.time()
            if attempt > 1 and t - t_start > max_s:
                log.append({"attempt": attempt, "result": "time_cap", "elapsed_s": round(t - t_start)})
                break
            # Only the task, the latest attempt and its feedback are sent: older attempts add nothing a small
            # model can use and would overflow its context window.
            call = messages if len(messages) <= 4 else messages[:2] + messages[-2:]
            # Thinking needs a large output budget, but prompt + output must fit the served context window.
            est_prompt = int(sum(len(m["content"]) for m in call) / 2.5)  # Vietnamese + code: ~2.5 chars/token
            budget = max(4000, min(max_tokens, int(os.environ.get("LLM_CONTEXT", "40960")) - est_prompt - 512))
            # Retries sample (seeded, so runs stay reproducible): at temperature 0 with the trimmed history a retry
            # can be byte-identical to the previous request and return the same failing code from cache.
            res = self.llm.chat(call, max_tokens=budget, thinking=True,
                                temperature=0.0 if attempt == 1 else 0.6, seed=42 + attempt)
            code = extract_code(res.content)
            repeat = ""
            if code is not None:
                key = code.strip()
                if key in seen:
                    repeat = (f"\n\nThis is the same code as attempt {seen[key]}, which already failed this way. Do not "
                              "resend it: use the exact column names listed for THIS business, not the ones in the "
                              "example loaders.")
                seen.setdefault(key, attempt)
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
                tables, meta = sanitize(*run_loader(code, store))
                tables = finalize(tables, meta)
                errs, report = validate(tables, meta)
                errs += lint(code)
            except Exception:  # noqa: BLE001
                tb = traceback.format_exc(limit=6)
                entry.update(result="exception", error=tb[-3000:])
                log.append(entry)
                hint = ("\nExact columns returned by raw.read_table(file, sheet) with default arguments (the example "
                        "loaders were written for other shops; their column names do not exist here):\n"
                        + columns_overview(store)) if "KeyError" in tb else ""
                messages.append({"role": "user", "content":
                                 f"The code raised an exception:\n```\n{tb[-3000:]}\n```\n{hint}\n"
                                 "Fix it and return the full code." + repeat})
                continue
            entry.update(result="errors" if errs else "valid", errors=errs, run_s=round(time.time() - t, 1))
            log.append(entry)
            if errs and best is None:
                cand = _drop_bad_tables(tables, meta, errs)
                if cand is not None and (partial is None or len(cand[0]) > len(partial[1])):
                    partial = (code, *cand)
                    entry["partial_ok"] = sorted(cand[0])
                soft = _accept_with_warnings(tables, meta, errs)
                if soft is not None and (lenient is None or len(soft[1]["soft_errors"]) < len(lenient[2]["soft_errors"])):
                    lenient = (code, *soft)
                    entry["lenient_ok"] = sorted(soft[0])
            if not errs:
                best = (code, tables, meta, report)
                if not self.review or reviewed:
                    break
                # A review round costs about as much as the attempt that passed; on slow GPUs keep the time for
                # other businesses (an L4 test spent 342 s on a review that answered OK).
                if time.time() - t_start > 0.4 * max_s:
                    entry["review"] = "skipped_time"
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
                             "\n\nFix the code and return the full code." + repeat})
        (out / "build_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str))
        (out / "messages.json").write_text(json.dumps(messages, ensure_ascii=False, indent=1))
        if best is None and partial is not None:
            best = partial
        if best is None and lenient is not None:
            best = lenient
        if best is None:
            meta = {"biz_id": biz_id, "ok": False, "fingerprint": fp}
            (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
            return meta
        code, tables, meta, report = best
        (out / "loader.py").write_text(code)
        (out / "report.txt").write_text(report)
        tdir = out / "tables"
        tdir.mkdir(exist_ok=True)
        for name, df in tables.items():
            df.to_parquet(tdir / f"{name}.parquet", index=False)
        meta.update(biz_id=biz_id, ok=True, tables=sorted(tables), fingerprint=fp, build_model=self.llm.model,
                    build_examples=bool(examples))
        if domain:
            meta["domain"] = domain
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
        return meta


_SOFT = (
    re.compile(r"monthly completed revenue varies"),
    re.compile(r"matches orders\.goods_total for only"),
    re.compile(r"orders\.status mapping looks wrong"),
    re.compile(r"is >100x the 99th percentile"),
)


def _is_soft(err: str, tables: dict) -> bool:
    """Data-quality warnings a usable layer may still have (vs structural errors that break every answer)."""
    if any(rx.search(err) for rx in _SOFT):
        return True
    m = re.match(r"order_lines\.(qty|qty_base|line_amount): ([\d.]+)% null", err)
    if m:
        return float(m.group(2)) <= 5
    m = re.match(r"order_lines: ([\d.]+)% of lines reference an order_id missing", err)
    if m:
        return float(m.group(1)) <= 2
    m = re.match(r"order_lines\.line_amount: (\d+) lines are null", err)
    if m and "order_lines" in tables:
        return int(m.group(1)) <= max(20, 0.01 * len(tables["order_lines"]))
    return False


def _accept_with_warnings(tables: dict, meta: dict, errs: list[str]):
    """Last resort when no attempt validated: keep the layer if every remaining error is a soft warning or belongs
    to a non-core table (dropped). A slightly imperfect layer answers far better than the raw-sheet fallback, which
    the answerability gate turns into abstain. Returns (tables, meta, report) or None."""
    if not {"orders", "order_lines"} <= set(tables):
        return None
    hard = [e for e in errs if not _is_soft(e, tables)]
    bad = {_error_table(e) for e in hard}
    if None in bad or bad & (set(CORE_TABLES) | {"loader"}):
        return None
    kept = {k: v for k, v in tables.items() if k not in bad}
    meta = dict(meta, dropped_tables=sorted(bad), soft_errors=[e for e in errs if _is_soft(e, tables)])
    meta.pop("table_errors", None)
    errs2, report = validate(kept, meta)
    if any(not _is_soft(e, kept) for e in errs2):
        return None
    return kept, meta, report


def _error_table(err: str) -> str | None:
    m = re.match(r"(?:table '|meta\['extra_desc'\] missing description for \[')?([a-z_]+)[.:' ]", err)
    return m.group(1) if m else None


def _drop_bad_tables(tables: dict, meta: dict, errs: list[str]):
    """If every error belongs to a non-core table, drop those tables and re-validate.

    Returns (tables, meta, report) of the reduced set, or None if a core table is wrong.
    """
    bad = {_error_table(e) for e in errs}
    if None in bad or bad & (set(CORE_TABLES) | {"loader"}):
        return None
    kept = {k: v for k, v in tables.items() if k not in bad}
    meta = dict(meta, dropped_tables=sorted(bad), dropped_reasons=errs)
    meta.pop("table_errors", None)
    errs2, report = validate(kept, meta)
    return None if errs2 else (kept, meta, report)


def refresh(work_dir: str | Path, biz_id: str, biz_dir: str) -> tuple[bool, list[str]]:
    """Re-run a cached loader with the current parsers/finalize/validate (no LLM). Saves tables if still valid.

    A build that now fails validation is marked ok=False (so the agent falls back to raw data) until rebuilt.
    """
    out = Path(work_dir) / biz_id
    meta_p = out / "meta.json"
    old = json.loads(meta_p.read_text())
    code = (out / "loader.py").read_text()
    tables, meta = sanitize(*run_loader(code, RawStore(biz_dir)))
    tables = finalize(tables, meta)
    errs, report = validate(tables, meta)
    errs += lint(code)
    keep = {k: old[k] for k in ("fingerprint", "biz_id") if k in old}
    if errs:
        meta_p.write_text(json.dumps({**old, "ok": False, "refresh_errors": errs}, ensure_ascii=False, indent=1,
                                     default=str))
        return False, errs
    (out / "report.txt").write_text(report)
    if (out / "tables").exists():
        shutil.rmtree(out / "tables")
    (out / "tables").mkdir()
    for name, df in tables.items():
        df.to_parquet(out / "tables" / f"{name}.parquet", index=False)
    meta.update(keep, ok=True, tables=sorted(tables))
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    return True, []


def load_semantic(work_dir: str | Path, biz_id: str) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load the cached canonical tables + meta of one business."""
    out = Path(work_dir) / biz_id
    meta = json.loads((out / "meta.json").read_text())
    tables = {p.stem: pd.read_parquet(p) for p in sorted((out / "tables").glob("*.parquet"))}
    return repair_tables(tables), meta
