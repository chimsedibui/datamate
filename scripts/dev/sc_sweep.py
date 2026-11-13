#!/usr/bin/env python3
"""Offline sweep of self-consistency settings from a DM_SC run's logs (no LLM calls).

For each max sample count k and vote threshold `need`, replay the agent's early-stopping vote over the logged
samples (in order) and score against answers.jsonl. `calls` is the mean number of samples the agent would have
drawn; `short` counts questions whose log holds fewer samples than that setting needs (an earlier early stop),
which are voted on what is there. Run the agent with DM_SC_ALL=1 (e.g. DM_SC=5 DM_SC_ALL=1) to log every sample.

    python scripts/dev/sc_sweep.py --logs out/dev_sc/logs --split data/train_dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from datamate.qa.agent import decided, vote  # noqa: E402
from datamate.core.scoring import score_one  # noqa: E402


def replay(samples: list[dict], k: int, need: int) -> tuple[dict, int, bool]:
    used = []
    for i, p in enumerate(samples[:k]):
        used.append(p)
        if decided(used, need, k - i - 1):
            return vote(used, need)[0], len(used), False
    return vote(used, need)[0], len(used), len(used) < k


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
    print(f"{len(runs)} questions, up to {kmax} samples\n  k need    PA    +1   -1    0  calls short")
    for k in range(1, kmax + 1):
        for need in range(1, k + 1):
            res = [(score_one(p, refs[q]), n, short) for q, s in runs for p, n, short in [replay(s, k, need)]]
            sc = [r[0] for r in res]
            print(f"{k:3d} {need:4d} {sum(sc) / len(sc):6.3f} {sc.count(1):5d} {sc.count(-1):4d} {sc.count(0):4d}"
                  f" {sum(r[1] for r in res) / len(res):6.2f} {sum(r[2] for r in res):5d}")


if __name__ == "__main__":
    main()
