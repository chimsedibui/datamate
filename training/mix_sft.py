"""SFT mix for LoRA v4: the v3 rows plus the OpenAI-teacher rows on synthetic shops.

Teacher rows follow build_sft.py's route preference: per question, replies that read the canonical views are kept first;
a reply that reads raw tables is kept only when no reply of that question reached the answer through the views (raw SQL is
taught only where nothing else worked). At most --per-question rows per question. Unanswerable rows are capped at
--max-unanswerable of the teacher rows so the adapter does not learn to give up.

    python training/mix_sft.py --base data/sft/sql_rft_v3_notes.jsonl --teacher runs/teacher_b1/sft.jsonl --out data/sft/sql_v4_mix.jsonl
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

VIEW = re.compile(r'\b"?v_(sales|orders)(_all)?"?\b|debt_at\(')


def reads_views(reply: str) -> bool:
    return bool(VIEW.search(reply.split("```sql")[-1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, nargs="+", required=True)
    ap.add_argument("--answers", type=Path, default=Path("data/synth_train/answers.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-question", type=int, default=1)
    ap.add_argument("--max-unanswerable", type=float, default=0.04)
    ap.add_argument("--max-teacher", type=int, help="cap on teacher rows, spread over shops")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    gold = {json.loads(l)["id"]: json.loads(l)["type"] for l in open(a.answers, encoding="utf-8")}
    base = [json.loads(l) for l in open(a.base, encoding="utf-8")]
    by_q = defaultdict(list)
    for path in a.teacher:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            by_q[r["id"]].append(r)
    picked, stats = [], Counter()
    for qid, rows in by_q.items():
        via_views = [r for r in rows if reads_views(r["reply"])]
        chosen = via_views or rows
        stats["view route" if via_views else "raw route"] += 1
        rng.shuffle(chosen)
        picked += chosen[: a.per_question]
    unans = [r for r in picked if gold.get(r["id"]) == "unanswerable"]
    rest = [r for r in picked if gold.get(r["id"]) != "unanswerable"]
    cap = int(a.max_unanswerable * len(rest) / (1 - a.max_unanswerable))
    rng.shuffle(unans)
    teacher = rest + unans[:cap]
    if a.max_teacher and len(teacher) > a.max_teacher:   # round-robin over shops so every schema stays in
        shops = defaultdict(list)
        for r in teacher:
            m = re.search(r"Dữ liệu (s[tv]_\d+)", r["user"])
            shops[m.group(1) if m else ""].append(r)
        order, i = list(shops.values()), 0
        teacher = []
        while len(teacher) < a.max_teacher and any(order):
            g = order[i % len(order)]
            if g:
                teacher.append(g.pop())
            i += 1
    stats["teacher rows"] = len(teacher)
    stats["unanswerable kept"] = min(cap, len(unans))
    rows = [dict(r, route=r.get("route") or ("views" if reads_views(r["reply"]) else "raw")) for r in base + teacher]
    rng.shuffle(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"base rows": len(base), **stats, "total": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
