"""Pool the samples of several saved runs of the same questions and vote once: an ensemble of their adapters, without the model.

    python analysis/pool_runs.py --member runs/sql_v9b_public runs/sql_v9_public --member runs/public_v4 \
        --data data/public_test --layers configs/public_v9 --out runs/pool_v3_v4

A member is the run dirs of one setup that together cover the questions (a dir holds trace.jsonl or shard*/trace.jsonl). Each
member's voters are picked as in its own run (a debt question on a shop with the debt layer: only the samples calling a debt
macro), then every voter of every member votes together with the run's own rule (voting.vote, final_answer): the share is
counted over all voters, so two adapters that disagree end in an abstain instead of a wrong answer. --layers is the dir whose
logs/_debt_<shop>.json say which shops have the debt layer. Writes OUT/submission.jsonl and OUT/pool.jsonl (per question: the
pooled vote and each member's own vote). With DATA/answers.jsonl it also prints the score of each member and of the pool.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.hints import DEBT_WORDS  # noqa: E402
from datamate.ingest import norm  # noqa: E402
from datamate.submission import write_submission  # noqa: E402
from datamate.voting import debt_pool, final_answer, vote  # noqa: E402

OUTSIDE = "period outside the shop's data dates"


def load(dirs):
    rows = {}
    for d in dirs:
        files = [f for f in [d / "trace.jsonl", *sorted(d.glob("shard*/trace.jsonl"))] if f.exists()]
        if not files:
            raise SystemExit(f"{d}: no trace.jsonl")
        for f in files:
            for line in f.open(encoding="utf-8"):
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--member", nargs="+", type=Path, action="append", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--layers", type=Path, nargs="+", help="dir(s) with logs/_debt_<shop>.json or shard*/logs/_debt_<shop>.json (default: no debt routing)")
    ap.add_argument("--logs", type=Path, help="per-question log dir (<id>.json, Docker /result/logs): its answer becomes the pooled one, with the members' votes")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids", type=Path, help="only these question ids")
    ap.add_argument("--k", type=int, default=8, help="plain samples per question in each run (thinking samples follow them)")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    qs = [json.loads(l) for l in (a.data / "questions.jsonl").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    members = [load(m) for m in a.member]
    if a.ids:
        keep = set(a.ids.read_text().split())
        qs = [q for q in qs if q["id"] in keep]
    qs = [q for q in qs if any(q["id"] in m for m in members)]
    has_debt = {}
    for d in a.layers or []:
        for f in [*(d / "logs").glob("_debt_*.json"), *d.glob("shard*/logs/_debt_*.json")]:
            has_debt[f.stem[len("_debt_"):]] = bool(json.loads(f.read_text(encoding="utf-8")).get("used"))
    results, stats, moves = {}, Counter(), Counter()
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "pool.jsonl").open("w", encoding="utf-8") as log:
        for q in qs:
            dq = has_debt.get(q["business_id"], False) and bool(DEBT_WORDS.search(norm(q["question"])))
            voters, own, oor = [], [], False
            for m in members:
                t = m.get(q["id"])
                if t is None:
                    stats["missing in a member"] += 1
                    own.append(None)
                    continue
                texts, preds = [s["text"] for s in t["samples"]], [s["pred"] for s in t["samples"]]
                voters += [preds[j] for j in debt_pool(texts, a.k, dq)]
                oor |= t["final"].get("reason") == OUTSIDE
                own.append(t["final"])
            ans, conf = vote(voters, len(voters)) if voters else ({"type": "abstain"}, 0.0)
            r = final_answer(ans, conf, a.threshold, oor, True)
            results[q["id"]] = r
            kinds = ["abstain" if o is None or o["type"] == "abstain" else "answer" for o in own]
            same = len({json.dumps(o, sort_keys=True, ensure_ascii=False) for o in own if o and o["type"] != "abstain"}) <= 1
            moves[("pool answers" if r["type"] != "abstain" else "pool abstains") + " | members: " +
                  ("all abstain" if "answer" not in kinds else ("agree" if same else "DISAGREE") + f" ({kinds.count('answer')}/{len(kinds)} answer)")] += 1
            log.write(json.dumps({"id": q["id"], "business_id": q["business_id"], "final": r, "vote": ans, "conf": round(conf, 4),
                                  "voters": len(voters), "members": own}, ensure_ascii=False, default=str) + "\n")
    write_submission(a.out / "submission.jsonl", qs, results)
    if a.logs:  # the submitted answer and its log record must agree
        n_logs = 0
        for line in (a.out / "pool.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            f = a.logs / f"{r['id']}.json"
            if not f.exists():
                continue
            rec = json.loads(f.read_text(encoding="utf-8"))
            rec["answer"] = r["final"]
            rec["ensemble"] = {"members": r["members"], "vote": r["vote"], "vote_share": r["conf"], "voters": r["voters"],
                               "note": "the answer pools the samples of every member (one adapter each) and votes once"}
            f.write_text(json.dumps(rec, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            n_logs += 1
        print(f"{n_logs} log records updated in {a.logs}")
    n = len(qs)
    print(f"{n} questions, {len(members)} members, pool answers {sum(r['type'] != 'abstain' for r in results.values())}; "
          + ", ".join(f"member {i + 1} answers {sum(1 for q in qs if q['id'] in m and m[q['id']]['final']['type'] != 'abstain')}" for i, m in enumerate(members)))
    for k, v in sorted(moves.items()):
        print(f"  {v:5}  {k}")
    if stats:
        print(dict(stats))
    gold_file = a.data / "answers.jsonl"
    if gold_file.exists():
        from datamate.scoring import score
        gold = {json.loads(l)["id"]: json.loads(l) for l in gold_file.open(encoding="utf-8")}
        def tally(get):
            c = Counter()
            for q in qs:
                p = get(q["id"])
                c[0 if p is None or p["type"] == "abstain" else score(p, gold[q["id"]])] += 1
            return f"right {c[1]} wrong {c[-1]} abstain {c[0]} PA {(c[1] - c[-1]) / max(1, n):.3f}"
        for i, m in enumerate(members):
            print(f"member {i + 1}: " + tally(lambda id: m[id]["final"] if id in m else None))
        print("pool:     " + tally(results.get))


if __name__ == "__main__":
    main()
