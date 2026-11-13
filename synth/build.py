"""Build a synthetic split laid out like data/train, so every pipeline command runs on it unchanged:

    OUT/questions.jsonl  OUT/answers.jsonl  OUT/rulebook.md  OUT/calendar.csv  OUT/data/<shop>/{files, README.md}

The rulebook and calendar are the train split's. Validation shops come from industries that no training shop uses, so they
measure transfer to new industries the way the public and private tests do.

    python -m synth.build --out data/synth_train --shops 200 --seed 0
    python -m synth.build --out data/synth_val --shops 24 --seed 1 --val
"""
import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

from synth import catalog
from synth.questions import make_questions
from synth.render import render
from synth.world import simulate

VAL_INDUSTRIES = ["hoa_tuoi", "giay_dep", "thu_cung"]      # held out from training shops


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shops", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--questions", type=int, default=60, help="questions per shop")
    ap.add_argument("--val", action="store_true", help="validation split: only the held-out industries")
    ap.add_argument("--train-data", default="data/train", help="where rulebook.md and calendar.csv are copied from")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new folder")
    (out / "data").mkdir(parents=True)
    shutil.copy(Path(args.train_data) / "rulebook.md", out / "rulebook.md")
    shutil.copy(Path(args.train_data) / "calendar.csv", out / "calendar.csv")
    pool = VAL_INDUSTRIES if args.val else [k for k in catalog.INDUSTRIES if k not in VAL_INDUSTRIES]
    weights = np.array([0.5 if k in catalog.TRAIN_LIKE else 1.0 for k in pool])
    weights /= weights.sum()
    prefix = "sv" if args.val else "st"
    rng = np.random.default_rng(args.seed)
    stats, kinds, t0 = Counter(), Counter(), time.time()
    with open(out / "questions.jsonl", "w", encoding="utf-8") as fq, open(out / "answers.jsonl", "w", encoding="utf-8") as fa:
        for i in range(args.shops):
            sid = f"{prefix}_{args.seed:02d}{i + 1:03d}"
            industry = str(rng.choice(pool, p=weights))
            srng = np.random.default_rng([args.seed, i])
            shop = simulate(sid, industry, srng)
            kind, smeta = render(shop, srng, out / "data" / sid)
            kinds[kind] += 1
            smeta.update({"industry": industry, "start": shop.start.isoformat(), "end": shop.end.isoformat()})
            (out / "_meta").mkdir(exist_ok=True)
            (out / "_meta" / f"{sid}.json").write_text(json.dumps(smeta, ensure_ascii=False, indent=1), encoding="utf-8")
            for q, a in make_questions(shop, srng, args.questions):
                fq.write(json.dumps(q, ensure_ascii=False) + "\n")
                fa.write(json.dumps(a, ensure_ascii=False) + "\n")
                stats[a["type"]] += 1
            print(f"{sid} {industry:15} orders {len(shop.orders):6} lines {len(shop.lines):6}  {time.time() - t0:5.0f}s", flush=True)
    meta = {"shops": args.shops, "seed": args.seed, "val": args.val, "answer_types": stats, "archetypes": kinds}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
