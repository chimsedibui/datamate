#!/usr/bin/env python3
"""Score per-question logs (<out>/logs/*.json) against answers.jsonl: total, per business, per answer type.

Works on partial runs and on runs re-done for a few businesses (submission.jsonl only holds the last run).

    python scripts/dev/score_logs.py --logs out/dev_base/qa/logs --split data/train_dev [--biz biz_101 biz_102]
    python scripts/dev/score_logs.py ... --compare out/other/logs     # same questions, side by side
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from datamate.core.scoring import score_one  # noqa: E402


def load(logs: Path, refs: dict) -> dict[str, dict]:
    out = {}
    for f in logs.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("id") in refs:
            out[d["id"]] = d.get("answer") or d.get("final") or {"type": "abstain"}
    return out


def fmt(v: list[int]) -> str:
    return f"PA={sum(v) / len(v):+.3f} n={len(v):4d}  +{v.count(1):4d} -{v.count(-1):4d} 0={v.count(0):4d}" if v else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--biz", nargs="*")
    ap.add_argument("--compare")
    a = ap.parse_args()
    split = Path(a.split)
    refs = {d["id"]: d for d in map(json.loads, open(split / "answers.jsonl"))}
    biz = {d["id"]: d["business_id"] for d in map(json.loads, open(split / "questions.jsonl"))}
    runs = [load(Path(a.logs), refs)] + ([load(Path(a.compare), refs)] if a.compare else [])
    ids = set(runs[0]).intersection(*runs[1:]) if a.compare else set(runs[0])
    if a.biz:
        ids = {i for i in ids if biz[i] in a.biz}
    for title, key in (("business", lambda i: biz[i]), ("type", lambda i: refs[i]["type"])):
        groups = collections.defaultdict(lambda: [[] for _ in runs])
        for i in ids:
            for k, r in enumerate(runs):
                groups[key(i)][k].append(score_one(r[i], refs[i]))
        if title == "business":
            tot = [[s for g in groups.values() for s in g[k]] for k in range(len(runs))]
            print("TOTAL".ljust(14) + "   |   ".join(fmt(t) for t in tot))
        print(f"-- by {title}")
        for g in sorted(groups):
            print(f"  {g:12s}" + "   |   ".join(fmt(v) for v in groups[g]))


if __name__ == "__main__":
    main()
