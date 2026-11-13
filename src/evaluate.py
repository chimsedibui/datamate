"""Score a submission against reference answers (Penalized Accuracy, competition rules).

Usage:
    python src/evaluate.py --pred out/train/submission.jsonl --split data/train
    python src/evaluate.py --pred out/train/submission.jsonl --split data/train --all   # unanswered = abstain

By default only questions present in the submission are scored (useful with --per-biz sampling).
Writes <pred dir>/eval.json with per-question scores and the wrong cases (prediction vs reference).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datamate.core.scoring import score_all  # noqa: E402


def read_jsonl(p: Path) -> dict[str, dict]:
    return {d["id"]: d for d in (json.loads(x) for x in p.read_text().splitlines() if x.strip())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    split = Path(a.split)
    preds = read_jsonl(Path(a.pred))
    refs = read_jsonl(split / "answers.jsonl")
    qs = read_jsonl(split / "questions.jsonl")
    if not a.all:
        refs = {k: v for k, v in refs.items() if k in preds}
    r = score_all(preds, refs)

    by_biz = defaultdict(list)
    for qid, s in r["scores"].items():
        by_biz[qs[qid]["business_id"]].append(s)

    print(f"PA = {r['PA']:.4f}   n={r['n']}  correct={r['correct']} wrong={r['wrong']} abstain={r['abstain']}  "
          f"acc(answered)={r['accuracy_answered']:.3f}")
    print("\nper reference type:")
    for t, c in r["per_type"].items():
        n = sum(c.values())
        print(f"  {t:13s} n={n:4d}  correct={c.get('correct', 0):4d}  wrong={c.get('wrong', 0):4d}  "
              f"abstain={c.get('abstain', 0):4d}  PA={(c.get('correct', 0) - c.get('wrong', 0)) / n:+.3f}")
    print("\nper business:")
    for b in sorted(by_biz):
        s = by_biz[b]
        print(f"  {b}: n={len(s):3d}  PA={sum(s) / len(s):+.3f}")

    wrong = [{"id": q, "business_id": qs[q]["business_id"], "question": qs[q]["question"],
              "as_of_date": qs[q]["as_of_date"], "pred": preds.get(q), "ref": refs[q]}
             for q, s in r["scores"].items() if s < 0]
    out = Path(a.pred).parent / "eval.json"
    out.write_text(json.dumps({k: v for k, v in r.items() if k != "scores"} | {"scores": r["scores"], "wrong": wrong},
                              ensure_ascii=False, indent=1))
    print(f"\ndetails -> {out}")


if __name__ == "__main__":
    main()
