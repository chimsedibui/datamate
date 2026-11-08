#!/usr/bin/env python3
"""Make a smaller split folder (same layout as /data) for smoke tests and quick evaluations. Standard library only.

Usage:
    python3 scripts/make_subset.py --split data/train --out /tmp/train_sample --per-biz 5      # = predict.py --per-biz 5
    python3 scripts/make_subset.py --split data/public_test --out /tmp/smoke --limit 20
    python3 scripts/make_subset.py --split data/train --out /tmp/train10 --per-biz 5 --max-biz 10

Business folders, rulebook.md, calendar.csv and answers.jsonl (if present) are copied (not symlinked: links that
point outside the folder break once it is mounted as /data in the container); questions.jsonl is filtered. --per-biz uses the same sampling as src/predict.py (seed 0), so scores compare with earlier runs.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-biz", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-biz", type=int, default=0, help="keep N businesses, spread evenly over the sorted list")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    split, out = Path(a.split).resolve(), Path(a.out)
    qs = [json.loads(x) for x in (split / "questions.jsonl").read_text().splitlines() if x.strip()]
    if a.max_biz:
        biz = sorted({q["business_id"] for q in qs})
        keep = {biz[i * len(biz) // a.max_biz] for i in range(min(a.max_biz, len(biz)))}
        qs = [q for q in qs if q["business_id"] in keep]
    if a.per_biz:
        by = defaultdict(list)
        for q in qs:
            by[q["business_id"]].append(q)
        rnd = random.Random(a.seed)
        qs = [q for b in sorted(by) for q in rnd.sample(by[b], min(a.per_biz, len(by[b])))]
    if a.limit:
        qs = qs[:a.limit]

    (out / "data").mkdir(parents=True, exist_ok=True)
    for name in ("rulebook.md", "calendar.csv", "answers.jsonl"):
        if (split / name).exists() and not (out / name).exists():
            shutil.copy2(split / name, out / name)
    for b in sorted({q["business_id"] for q in qs}):
        if not (out / "data" / b).exists():
            shutil.copytree(split / "data" / b, out / "data" / b)
    (out / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs))
    print(f"{out}: {len(qs)} questions, {len({q['business_id'] for q in qs})} businesses")


if __name__ == "__main__":
    main()
