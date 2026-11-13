"""Build generator-v2 splits: several small splits, each with its own rulebook variant (synth/rules.py), shops with tables
beyond sales (synth/world2.py), and questions whose SQL has been run on the rendered files and matches the computed answer.

    OUT/rNN/rulebook.md calendar.csv questions.jsonl answers.jsonl rules.json meta.json data/<shop>/ _meta/<shop>.json
    OUT/rNN/views/logs/_views_<shop>.json     the correct canonical view of each shop (`--views-from OUT/rNN/views`)

A question is kept only when (1) the shop's gold view reproduces the generator's revenue and (2) the question's SQL, executed
by the pipeline's own executor on the shop's database, scores as right against the computed answer. answers.jsonl keeps the
SQL and the full reply (notes, TYPE, SQL) for few-shot and training rows.

    python -m synth.build2 --out data/synth2 --variants 10 --shops 20 --seed 100 --questions 60 --workers 4
    python -m synth.build2 --out data/synth2_val --variants 3 --shops 8 --seed 900 --questions 50 --val
"""
import argparse
import json
import shutil
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from synth import catalog
from synth.build import VAL_INDUSTRIES

A = None


def one_shop(job):
    out, seed, i, sid, industry, n_questions = job
    from datamate import sqlexec, views
    from datamate.ingest import load_business
    from datamate.scoring import score
    from synth import goldviews
    from synth.questions import Facts, month_bounds
    from synth.questions2 import make2
    from synth.render import render
    from synth.world2 import simulate2
    rules = json.loads((out / "rules.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng([seed, i])
    shop = simulate2(sid, industry, rng, rules.get("variant"), rules.get("books"))
    kind, meta = render(shop, rng, out / "data" / sid)
    meta.update({"industry": industry, "start": shop.start.isoformat(), "end": shop.end.isoformat()})
    (out / "_meta" / f"{sid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    stats = Counter()
    try:
        db = load_business(out, sid, canon=True)[0]
        sql = {"flat": goldviews.flat_sql, "pos": goldviews.pos_sql, "ledger": goldviews.ledger_sql}[kind](db, meta)
        views.build(db, sql)
        facts = Facts(shop, shop.end)
        for y, m in {(shop.start.year, shop.start.month), (shop.end.year, shop.end.month)}:
            a, b = month_bounds(y, m)
            got = db.execute("SELECT round(sum(amount)) FROM v_sales WHERE order_date BETWEEN ? AND ?", [a, b]).fetchone()[0] or 0
            if int(got) != facts.revenue(a, b):
                raise ValueError(f"gold view revenue {int(got)} != {facts.revenue(a, b)} in {y}-{m:02d}")
    except Exception as e:  # noqa: BLE001
        return sid, industry, kind, [], Counter({"shop failed": 1}), [f"{sid}: {str(e)[:200]}"]
    (out / "views" / "logs" / f"_views_{sid}.json").write_text(json.dumps({"agree": 8, "k": 8, "used": True, "sql": sql, "errors": [], "drafts": []},
                                                                           ensure_ascii=False), encoding="utf-8")
    rows, bad = [], []
    for q, a in make2(shop, meta, rules, rng, n_questions):
        try:
            pred = sqlexec.execute(db, a["reply"], q["question"])
            ok = score(pred, a) == 1
            why = "" if ok else f"got {json.dumps(pred, ensure_ascii=False)[:80]} want {a['type']} {str(a['answer'])[:60]}"
        except Exception as e:  # noqa: BLE001
            ok, why = False, "error " + str(e)[:140]
        stats[a["kind"] + (" ok" if ok else " FAIL")] += 1
        if ok:
            rows.append((q, a))
        elif len(bad) < 3:
            bad.append(f"{sid} [{a['kind']}] {a['clean_question'][:70]} | {why} | {a['sql'][:260]}")
    return sid, industry, kind, rows, stats, bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--shops", type=int, default=10, help="shops per variant")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--questions", type=int, default=60)
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--names", action="store_true", help="each split also draws an attribute written in the item names (rules.choose_name_rule)")
    ap.add_argument("--books", action="store_true", help="each split also draws side books tied to the sales book: orders paid in periods and a book of "
                    "amounts per order (rules.choose_books)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train-data", type=Path, default=Path("data/train"))
    a = ap.parse_args()
    from synth.rules import choose, rulebook
    train_rules = (a.train_data / "rulebook.md").read_text(encoding="utf-8-sig")
    pool = VAL_INDUSTRIES if a.val else [k for k in catalog.INDUSTRIES if k not in VAL_INDUSTRIES]
    total, t0, fails = Counter(), time.time(), []
    for v in range(a.variants):
        out = a.out / f"r{v:02d}"
        if out.exists():
            raise SystemExit(f"{out} exists; pick a new folder")
        for d in ("data", "_meta", "views/logs"):
            (out / d).mkdir(parents=True)
        seed = a.seed + v
        rng = np.random.default_rng(seed)
        rules = choose(rng, val=a.val, names=a.names, books=a.books)
        (out / "rules.json").write_text(json.dumps(rules, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "rulebook.md").write_text(rulebook(train_rules, rules), encoding="utf-8")
        shutil.copy(a.train_data / "calendar.csv", out / "calendar.csv")
        (out / "meta.json").write_text(json.dumps({"seed": seed, "val": a.val, "variant": v}))
        jobs = [(out, seed, i, f"{'v2' if a.val else 't2'}_{v:02d}{i + 1:03d}", str(rng.choice(pool)), a.questions) for i in range(a.shops)]
        with Pool(a.workers) as p, open(out / "questions.jsonl", "w", encoding="utf-8") as fq, open(out / "answers.jsonl", "w", encoding="utf-8") as fa:
            for sid, industry, kind, rows, stats, bad in p.imap(one_shop, jobs):
                for q, ans in rows:
                    fq.write(json.dumps(q, ensure_ascii=False) + "\n")
                    fa.write(json.dumps(ans, ensure_ascii=False) + "\n")
                total.update(stats)
                fails += bad
                print(f"{sid} {industry:15} {kind:6} kept {len(rows):3}  {time.time() - t0:5.0f}s", flush=True)
    kinds = sorted({k.rsplit(" ", 1)[0] for k in total if k != "shop failed"})
    print("\nkind                 ok  fail")
    for k in kinds:
        print(f"{k:18} {total[k + ' ok']:5} {total[k + ' FAIL']:5}")
    print("shops failed:", total["shop failed"], "| kept", sum(v for k, v in total.items() if k.endswith(" ok")), "| dropped", sum(v for k, v in total.items() if k.endswith(" FAIL")))
    for f in fails[:40]:
        print("  FAIL", f)


if __name__ == "__main__":
    main()
