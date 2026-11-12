#!/usr/bin/env python3
"""Offline sweep of self-consistency settings from a DM_SC run's logs (no LLM calls).

For each sample count k (first k samples) and vote threshold `need`, re-run the vote and score against answers.jsonl.

    python scripts/sc_sweep.py --logs out/dev_sc/logs --split data/train_dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datamate.agent import vote  # noqa: E402
from datamate.scoring import score_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    refs = {d["id"]: d for d in map(json.loads, open(Path(a.split) / "answers.jsonl")) if d}
    runs = []
    for f in Path(a.logs).glob("*.json"):
        d = json.loads(f.read_text())
        if d["id"] in refs:
            runs.append((d["id"], [s["pred"] for s in d.get("sc_samples") or []] or [d.get("final")]))
    kmax = max(len(s) for _, s in runs)
    print(f"{len(runs)} questions, up to {kmax} samples\n  k need   PA    +1   -1    0")
    for k in range(1, kmax + 1):
        for need in range(1, k + 1):
            sc = [score_one(vote(s[:k], need)[0], refs[q]) for q, s in runs]
            print(f"{k:3d} {need:4d} {sum(sc) / len(sc):6.3f} {sc.count(1):5d} {sc.count(-1):4d} {sc.count(0):4d}")


if __name__ == "__main__":
    main()
