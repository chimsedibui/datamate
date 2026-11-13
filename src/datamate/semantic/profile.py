"""Compact textual profile of a business folder, used as LLM context for mapping.

The profile shows, per file/sheet: size, the first raw rows (so the model can see
title rows and multi-row headers) and, per column of the guessed header, a few
distinct sample values spread across the whole sheet (so the model sees every
format variant: '55.000đ', '25k', '1,01tr', '1 thg 5 2025'...).

Files/sheets sharing the same header signature are collapsed into one entry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.rawio import RawStore, _blank, guess_header_row

HEAD_ROWS = 6
SAMPLES = 8


def _cell(v, width=40) -> str:
    if _blank(v):
        return "·"
    s = str(v).replace("\n", " ")
    return s if len(s) <= width else s[: width - 1] + "…"


def _distinct_samples(s: pd.Series, k: int = SAMPLES) -> tuple[list, float]:
    nonnull = s[~s.map(_blank)]
    null_ratio = 1 - len(nonnull) / max(len(s), 1)
    if nonnull.empty:
        return [], null_ratio
    # spread over the column so late-appearing formats show up; prefer distinct "shapes"
    idx = np.linspace(0, len(nonnull) - 1, num=min(len(nonnull), 400)).astype(int)
    seen, shapes, out = set(), set(), []
    for v in nonnull.iloc[idx]:
        key = str(v)
        shape = "".join("9" if ch.isdigit() else ch for ch in key)[:20] if not isinstance(v, str) or any(c.isdigit() for c in key) else key
        if key in seen:
            continue
        seen.add(key)
        if shape in shapes and len(out) >= k // 2:
            continue
        shapes.add(shape)
        out.append(v)
        if len(out) >= k:
            break
    return out, null_ratio


def profile_frame(raw: pd.DataFrame, title: str) -> str:
    lines = [f"#### {title} — {len(raw)} rows × {raw.shape[1]} cols"]
    lines.append("first raw rows (row index: cells; '·' = empty):")
    for i in range(min(HEAD_ROWS, len(raw))):
        lines.append(f"  r{i}: " + " | ".join(_cell(v, 28) for v in raw.iloc[i]))
    h = guess_header_row(raw)
    body = raw.iloc[h + 1:]
    lines.append(f"columns (header guessed at r{h}; samples spread over all rows):")
    for j in range(raw.shape[1]):
        name = _cell(raw.iloc[h, j], 30) if len(raw) else f"col{j}"
        col = body.iloc[:, j]
        samples, nr = _distinct_samples(col)
        strs = col[col.map(lambda v: isinstance(v, str) and not _blank(v))]
        vc = strs.value_counts()
        if 0 < len(vc) <= 25 and len(strs) > 3 * len(vc):
            sm = "ALL VALUES " + ", ".join(f"{v!r}×{n}" for v, n in vc.items())
            others = [v for v in samples if not isinstance(v, str)]
            if others:
                sm += " ; non-text: " + ", ".join(_cell(v) for v in others[:4])
        else:
            sm = ", ".join(repr(v) if isinstance(v, str) else _cell(v) for v in samples)
        lines.append(f"  [{j}] {name} (null {nr:.0%}): {sm}")
    return "\n".join(lines)


def _signature(raw: pd.DataFrame) -> tuple:
    h = guess_header_row(raw)
    return tuple(str(v) for v in raw.iloc[h]) if len(raw) else ()


def profile_business(store: RawStore) -> str:
    parts = [f"# Business folder: {store.dir}", "## README.md", store.readme().strip(), "## Files"]
    groups: dict[tuple, list[tuple[str, str, pd.DataFrame]]] = {}
    for f in store.list_files():
        for sh in store.sheet_names(f):
            try:
                raw = store.read_raw(f, sh)
            except Exception as e:  # noqa: BLE001
                parts.append(f"#### {f} [{sh}] — unreadable: {e}")
                continue
            groups.setdefault((f.split("_")[0] if f.endswith(".xlsx") else f, _signature(raw)), []).append((f, sh, raw))
    for members in groups.values():
        f, sh, raw = members[0]
        loc = f if sh is None else f"{f} :: sheet/table '{sh}'"
        if len(members) > 1:
            total = sum(len(m[2]) for m in members)
            others = [m[0] if m[1] is None else f"{m[0]}::{m[1]}" for m in members]
            shown = others if len(others) <= 8 else others[:4] + ["…"] + others[-3:]
            parts.append(f"(same header in {len(members)} sheets/files, {total} rows total: {shown})")
        parts.append(profile_frame(raw, loc))
    return "\n\n".join(parts)
