"""SFT mix for LoRA v5 (trained from v3): generator-v2 rows (new rule definitions, tables beyond sales, new question shapes,
prompts with the rule block and the cost column), a sample of the OpenAI-teacher rows on v1 synthetic shops, and a sample of
the v3 train rows as replay so the adapter keeps what v3 knows.

    python training/mix_v5.py --out data/sft/sql_v5_mix.jsonl
    python training/mix_v5.py --keep data/sft/synth3_rows_variant.jsonl data/sft/synth3_rows_other.jsonl --out data/sft/sql_v5_mix.jsonl
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--v2", type=Path, nargs="+", default=[Path("data/sft/synth2_rows.jsonl"), Path("data/sft/synth2_rows_boost.jsonl")])
ap.add_argument("--teacher", type=Path, default=Path("data/sft/sql_v4_mix_fixed.jsonl"), help="the v4 mix; its rows with source=openai are the teacher rows")
ap.add_argument("--train", type=Path, default=Path("data/sft/sql_rft_v3_notes.jsonl"))
ap.add_argument("--n-teacher", type=int, default=500)
ap.add_argument("--n-train", type=int, default=500)
ap.add_argument("--keep", type=Path, nargs="*", default=[], help="generator rows kept in full (they count toward --max-v2), e.g. a kind added later")
ap.add_argument("--max-v2", type=int, default=3300)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]          # noqa: E731
keep = [r for p in a.keep for r in load(p)]
v2 = [r for p in a.v2 for r in load(p)]
rng.shuffle(v2)
v2 = keep + v2[: max(0, a.max_v2 - len(keep))]
teacher = [r for r in load(a.teacher) if r.get("source") == "openai"]
train = load(a.train)
rows = v2 + rng.sample(teacher, min(a.n_teacher, len(teacher))) + rng.sample(train, min(a.n_train, len(train)))
rows = [{k: r[k] for k in ("system", "user", "reply", "id", "thinking") if k in r} | {"route": r.get("route", ""), "source": r.get("source", "train")} for r in rows]
rng.shuffle(rows)
a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(json.dumps({"rows": len(rows), "by source": Counter(r["source"] for r in rows), "thinking rows": sum(bool(r["thinking"]) for r in rows),
                  "v2 kinds": Counter(r.get("kind", "") for r in v2)}, ensure_ascii=False))
