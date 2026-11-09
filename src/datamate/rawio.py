"""Raw file access for a business folder (csv / xlsx / sqlite), with caching.

Helpers exposed to LLM-generated loader code:
    list_files()                       -> list of file names in the business folder
    sheet_names(file)                  -> sheet names of an xlsx (or table names of a sqlite)
    read_raw(file, sheet=None)         -> DataFrame with header=None, every cell as read
    read_table(file, sheet=None, header_rows=1, skip_rows=0)
                                       -> DataFrame with column names taken from header rows
    read_sql(file, query)              -> DataFrame from a sqlite file
"""
from __future__ import annotations

import os
import sqlite3
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd


def _nfc(s):
    return unicodedata.normalize("NFC", s) if isinstance(s, str) else s


def fix_filename(name: str) -> str:
    """Repair zip-extracted names decoded as cp437 instead of utf-8 (e.g. 'Danh s├ích')."""
    try:
        fixed = name.encode("cp437").decode("utf-8")
        return _nfc(fixed)
    except (UnicodeEncodeError, UnicodeDecodeError):
        return _nfc(name)


class RawStore:
    """Read-only view over one business folder."""

    def __init__(self, biz_dir: str):
        self.dir = biz_dir
        self._real = {}
        for f in sorted(os.listdir(biz_dir)):
            self._real[fix_filename(f)] = f

    def list_files(self) -> list[str]:
        return [f for f in self._real if not f.lower().endswith(".md")]

    @staticmethod
    def _split(file: str, sheet=None):
        """Accept 'file.xlsx::Sheet' (how the profile labels sheets) as well as (file, sheet)."""
        if isinstance(file, str) and "::" in file:
            file, sh = file.split("::", 1)
            sheet = sheet if sheet is not None else (sh or None)
        return file, sheet

    def path(self, file: str) -> str:
        file, _ = self._split(file)
        real = self._real.get(_nfc(file), file)
        return os.path.join(self.dir, real)

    def readme(self) -> str:
        p = os.path.join(self.dir, "README.md")
        return open(p, encoding="utf-8").read() if os.path.exists(p) else ""

    def sheet_names(self, file: str) -> list[str]:
        return self._sheet_names(self._split(file)[0])

    @lru_cache(maxsize=None)
    def _sheet_names(self, file: str) -> list[str]:
        p = self.path(file)
        if p.endswith(".xlsx") or p.endswith(".xls"):
            return list(pd.ExcelFile(p, engine="calamine").sheet_names)
        if p.endswith(".sqlite") or p.endswith(".db"):
            with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as c:
                return [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
        return [None]

    @lru_cache(maxsize=None)
    def _xlsx_all(self, file: str) -> dict:
        return pd.read_excel(self.path(file), sheet_name=None, header=None, engine="calamine", dtype=object)

    def read_raw(self, file: str, sheet=None) -> pd.DataFrame:
        """Cells exactly as stored, no header inference (row 0 = first row of the sheet)."""
        file, sheet = self._split(file, sheet)
        p = self.path(file)
        if p.endswith(".xlsx") or p.endswith(".xls"):
            sheets = self._xlsx_all(file)
            if sheet is None:
                sheet = next(iter(sheets))
            if isinstance(sheet, int):
                sheet = list(sheets)[sheet]
            return sheets[sheet].copy()
        if p.endswith(".csv"):
            return self._csv(file).copy()
        if p.endswith(".sqlite") or p.endswith(".db"):
            df = self.read_sql(file, f'select * from "{sheet}"')
            return pd.concat([pd.DataFrame([df.columns]), pd.DataFrame(df.values)], ignore_index=True)
        raise ValueError(f"unsupported file {file}")

    @lru_cache(maxsize=None)
    def _csv(self, file: str) -> pd.DataFrame:
        p = self.path(file)
        for enc in ("utf-8-sig", "utf-16", "cp1258", "latin-1"):
            try:
                return pd.read_csv(p, header=None, dtype=object, encoding=enc, keep_default_na=True,
                                   sep=None, engine="python")
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"cannot decode {file}")

    def read_table(self, file: str, sheet=None, header_rows: int = 1, skip_rows: int | None = None) -> pd.DataFrame:
        """Read a sheet using `header_rows` rows (after `skip_rows`) as the header.

        `skip_rows=None` (default) auto-detects the header row per sheet, so blank or
        title rows above the header are skipped even when they differ between sheets.
        Multi-row headers are flattened: the lowest non-empty label wins; duplicates
        get the upper label prefixed ('Tiền|ĐG'); remaining duplicates get '.1', '.2'.
        Fully empty rows are dropped.
        """
        file, sheet = self._split(file, sheet)
        raw = self.read_raw(file, sheet)
        if skip_rows is None:
            from .profile import guess_header_row
            # the guessed row is the lowest (most specific) header row, so a multi-row header ends there
            skip_rows = max(0, guess_header_row(raw) - header_rows + 1)
        hdr = raw.iloc[skip_rows: skip_rows + header_rows].copy()
        body = raw.iloc[skip_rows + header_rows:].copy()
        hdr = hdr.ffill(axis=1) if header_rows > 1 else hdr
        names = []
        for j in range(raw.shape[1]):
            labels = [str(_nfc(v)).strip() for v in hdr.iloc[:, j] if not _blank(v)]
            names.append(labels[-1] if labels else f"col{j}")
        dup = {n for n in names if names.count(n) > 1}
        if header_rows > 1 and dup:
            for j, n in enumerate(names):
                if n in dup:
                    labels = [str(_nfc(v)).strip() for v in hdr.iloc[:, j] if not _blank(v)]
                    names[j] = "|".join(labels)
        seen: dict[str, int] = {}
        for j, n in enumerate(names):
            if n in seen:
                seen[n] += 1
                names[j] = f"{n}.{seen[n]}"
            else:
                seen[n] = 0
        body.columns = names
        body = body.dropna(how="all")
        body = body.map(lambda v: _nfc(v.strip()) if isinstance(v, str) else v)
        return body.reset_index(drop=True)

    def read_sql(self, file: str, query: str) -> pd.DataFrame:
        with sqlite3.connect(f"file:{self.path(file)}?mode=ro", uri=True) as c:
            return pd.read_sql_query(query, c)


def _blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return isinstance(v, str) and not v.strip()
