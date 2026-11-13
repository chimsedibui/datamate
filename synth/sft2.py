"""Training and few-shot material from the generator-v2 splits.

    python -m synth.sft2 bank --splits data/synth2 --teacher runs/teacher_b1/sft.jsonl --out assets/fewshot_bank_v2.jsonl
    python -m synth.sft2 rows --splits data/synth2 --per-shop 12 --fewshot assets/fewshot_bank_v2.jsonl --out data/sft/synth2_rows.jsonl

bank: the train few-shot bank plus verified synthetic examples that read only the canonical views and whose question is far
from every train example (best match under --novel), so the examples the train questions already get do not change. Rule-
defined terms are left out: their definition changes from split to split, and an example must never carry one over.
rows: one SFT row per chosen question, the prompt built by the pipeline itself on the split (its rulebook variant, the gold
views, hints, the rule rows the question names, few-shot examples on every second row), the reply = notes, TYPE and the SQL
that was verified against the computed answer.
"""
import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datamate import fewshot, sqlexec

FIXED_KINDS = {"shape", "price_effect"}            # v2 kinds whose meaning does not depend on the split's rule set


def best_sim(bank: fewshot.Bank, question: str) -> float:
    f = fewshot.feats(question)
    nq = math.sqrt(sum((c * bank.idf.get(x, 1.0)) ** 2 for x, c in f.items())) or 1.0
    return max(sum(c * g[x] * bank.idf[x] ** 2 for x, c in f.items() if x in g) / (nq * bank.norm[i] or 1.0) for i, g in enumerate(bank.f))


def cmd_bank(a) -> None:
    base = fewshot.Bank(a.base)
    cands = []
    for d in sorted(a.splits.glob("r*")):
        qs = {json.loads(l)["id"]: json.loads(l) for l in open(d / "questions.jsonl", encoding="utf-8")}
        for line in open(d / "answers.jsonl", encoding="utf-8"):
            r = json.loads(line)
            ok = r["kind"] in FIXED_KINDS or (r["kind"] == "profit" and r["type"] != "percent")
            if ok and r["type"] != "unanswerable":
                cands.append({"id": r["id"], "business_id": qs[r["id"]]["business_id"], "question": r["clean_question"], "type": r["type"], "sql": r["sql"], "kind": r["kind"]})
    if a.teacher:
        tq = {json.loads(l)["id"]: json.loads(l) for l in open(a.teacher_split / "questions.jsonl", encoding="utf-8")}
        good = {json.loads(l)["id"] for l in open(a.teacher_answers, encoding="utf-8")} if a.teacher_answers else None
        for line in open(a.teacher, encoding="utf-8"):
            r = json.loads(line)
            kind, sql = sqlexec.parse(r["reply"])
            if r["id"] not in tq or (good is not None and r["id"] not in good) or kind == "unanswerable" or not sql:
                continue
            cands.append({"id": r["id"], "business_id": tq[r["id"]]["business_id"], "question": tq[r["id"]]["question"], "type": kind, "sql": sql.strip().rstrip(";"), "kind": "v1"})
    random.Random(0).shuffle(cands)
    kept, seen, per = [], set(), Counter()
    for c in cands:
        tabs = fewshot.tables_of(c["sql"])
        sk = (fewshot.skeleton(c["sql"]), re.sub(r"\d+", "#", c["question"].lower())[:60])
        if not tabs or not tabs <= fewshot.CANON or sk in seen or per[c["kind"]] >= a.per_kind:
            continue
        # sales shapes must be far from every train example; cost / price-effect examples are kept anyway (the train bank has
        # one cost example), up to 150 a kind
        if (best_sim(base, c["question"]) >= a.novel) if c["kind"] in ("v1", "shape") else per[c["kind"]] >= 150:
            continue
        seen.add(sk)
        per[c["kind"]] += 1
        kept.append({k: c[k] for k in ("id", "business_id", "question", "type", "sql")})
    rows = base.rows + kept
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"train examples": len(base.rows), "candidates": len(cands), "added": len(kept), "by kind": per, "total": len(rows)}, ensure_ascii=False))


def cmd_rows(a) -> None:
    from synth.teacher import Prompts
    rng = random.Random(a.seed)
    out, stats = [], Counter()
    skip = {json.loads(l)["id"] for f in a.exclude for l in open(f, encoding="utf-8")}
    for d in sorted(a.splits.glob("r*")):
        P = Prompts(d, d / "views", extra=["--rule-lines"], fewshot=str(a.fewshot))
        ans = {json.loads(l)["id"]: json.loads(l) for l in open(d / "answers.jsonl", encoding="utf-8")}
        qs = [json.loads(l) for l in open(d / "questions.jsonl", encoding="utf-8")]
        by = defaultdict(lambda: defaultdict(list))
        for q in qs:
            if q["id"] not in skip and (not a.kinds or ans[q["id"]]["kind"] in a.kinds):
                by[q["business_id"]][ans[q["id"]]["kind"]].append(q)
        chosen = []
        for shop, kinds in by.items():                       # round-robin over the kinds of each shop
            pools = [rng.sample(v, len(v)) for v in kinds.values()]
            rng.shuffle(pools)
            picked = []
            while len(picked) < a.per_shop and any(pools):
                for pool in pools:
                    if pool and len(picked) < a.per_shop:
                        picked.append(pool.pop())
            chosen += picked
        for shop, todo, chats in P.shops(chosen):
            for q, cs in zip(todo, chats):
                j = rng.randrange(2)
                r = ans[q["id"]]
                out.append({"system": cs[j][0]["content"], "user": cs[j][1]["content"], "reply": r["reply"], "id": q["id"], "thinking": False,
                            "route": "views" if fewshot.tables_of(r["sql"]) <= fewshot.CANON else "raw", "source": "synth2", "kind": r["kind"]})
                stats[r["kind"]] += 1
                stats["rows with a rule block"] += "RULEBOOK ROWS NAMED IN THE QUESTION" in cs[j][1]["content"]
        print(d.name, len(out), flush=True)
        del P                                            # one variant's shops at a time: free their databases before the next
        import gc
        gc.collect()
    rng.shuffle(out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(json.dumps({"rows": len(out), **stats}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["bank", "rows"])
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=Path("assets/fewshot_bank.jsonl"))
    ap.add_argument("--teacher", type=Path)
    ap.add_argument("--teacher-split", type=Path, default=Path("data/synth_train"))
    ap.add_argument("--teacher-answers", type=Path, help="keep teacher rows only for ids in this answers file")
    ap.add_argument("--novel", type=float, default=0.6)
    ap.add_argument("--per-kind", type=int, default=400)
    ap.add_argument("--per-shop", type=int, default=12)
    ap.add_argument("--fewshot", type=Path, default=Path("assets/fewshot_bank.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", nargs="*", default=[], help="rows: only these question kinds")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[], help="rows: leave out the ids of these row files")
    a = ap.parse_args()
    {"bank": cmd_bank, "rows": cmd_rows}[a.mode](a)


if __name__ == "__main__":
    main()
