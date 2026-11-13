#!/usr/bin/env python3
"""Check a /result folder against the organizer guide (sections 3.2, 3.3, 10). Standard library only.

Usage:
    python3 scripts/release/validate_output.py --questions data/public_test/questions.jsonl --result result/
    python3 scripts/release/validate_output.py --questions ... --result run1/ --compare run2/    # reproducibility

Checks: every question has exactly one submission line in the published format; every question has a log record
with id, answer, elapsed_ms (non-negative int) and thinking (str, list of str or null); code/output present when
code ran; with --compare, how many answers differ between two runs. Exit code 1 if a check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VALUE_TYPES = {"number", "percent", "string", "date", "boolean", "list", "table"}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def check_entry(e: dict) -> str | None:
    t = e.get("type")
    if t in VALUE_TYPES:
        return None if "answer" in e else "value type without 'answer'"
    if t in ("unanswerable", "abstain"):
        return None
    return f"unknown type {t!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--compare", help="second result folder for a reproducibility diff")
    a = ap.parse_args()

    qids = [q["id"] for q in read_jsonl(Path(a.questions))]
    res = Path(a.result)
    errors: list[str] = []

    sub_p = res / "submission.jsonl"
    if not sub_p.exists():
        print(f"FAIL: {sub_p} missing")
        return 1
    sub = read_jsonl(sub_p)
    by_id: dict[str, dict] = {}
    for e in sub:
        if e.get("id") in by_id:
            errors.append(f"duplicate submission id {e.get('id')}")
        by_id[e.get("id")] = e
        bad = check_entry(e)
        if bad:
            errors.append(f"{e.get('id')}: {bad}")
    missing = [q for q in qids if q not in by_id]
    extra = [i for i in by_id if i not in set(qids)]
    types: dict[str, int] = {}
    for e in by_id.values():
        types[e.get("type")] = types.get(e.get("type"), 0) + 1

    log_missing, log_bad, elapsed = [], [], []
    for q in qids:
        lp = res / "logs" / f"{q}.json"
        if not lp.exists():
            log_missing.append(q)
            continue
        try:
            d = json.loads(lp.read_text())
        except json.JSONDecodeError:
            log_bad.append(f"{q}: not one JSON record")
            continue
        for k in ("id", "answer", "elapsed_ms", "thinking"):
            if k not in d:
                log_bad.append(f"{q}: missing '{k}'")
        em = d.get("elapsed_ms")
        if not (isinstance(em, int) and em >= 0):
            log_bad.append(f"{q}: elapsed_ms {em!r}")
        else:
            elapsed.append(em)
        th = d.get("thinking")
        if not (th is None or isinstance(th, str) or (isinstance(th, list) and all(isinstance(x, (str, type(None)))
                                                                                     for x in th))):
            log_bad.append(f"{q}: thinking has type {type(th).__name__}")
        if d.get("code") is not None and "output" not in d:
            log_bad.append(f"{q}: code without output")

    print(f"questions={len(qids)} submission_lines={len(sub)} missing={len(missing)} extra={len(extra)}")
    print(f"types: {dict(sorted(types.items(), key=lambda x: -x[1]))}")
    print(f"logs: missing={len(log_missing)} invalid={len(log_bad)}")
    if elapsed:
        elapsed.sort()
        print(f"elapsed_ms: median={elapsed[len(elapsed) // 2]} p95={elapsed[int(len(elapsed) * 0.95)]} "
              f"max={elapsed[-1]}")
    run_log = res / "logs" / "_run" / "predict_sh.log"
    if run_log.exists():
        print("predict.sh:", " | ".join(x.split("] ", 1)[-1] for x in run_log.read_text().splitlines()[-3:]))

    if a.compare:
        other = {e["id"]: e for e in read_jsonl(Path(a.compare) / "submission.jsonl")}
        diff = [q for q in qids if json.dumps(by_id.get(q), sort_keys=True) != json.dumps(other.get(q), sort_keys=True)]
        print(f"reproducibility: {len(diff)}/{len(qids)} answers differ ({len(diff) / max(len(qids), 1):.1%})"
              + (f", e.g. {diff[:5]}" if diff else ""))

    for msg in (errors[:10] + [f"missing submission: {missing[:10]}"] * bool(missing)
                + [f"missing logs: {log_missing[:10]}"] * bool(log_missing) + log_bad[:10]):
        print("FAIL:", msg)
    ok = not (errors or missing or log_missing or log_bad)
    print("OK" if ok else "NOT OK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
