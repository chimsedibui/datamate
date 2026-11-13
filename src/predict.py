"""Answer every question of a split and write submission.jsonl + per-question logs.

Usage:
    python src/predict.py --split data/train --work work/train --out out/train
    python src/predict.py --split data/train --work work/train --out out/train --per-biz 5 --jobs 4
    python src/predict.py --split /data --work /tmp/work --out /result --build      # Docker: build semantic layer first

Outputs:
    <out>/submission.jsonl  always complete and in question order: it starts as abstain for every question and is
                            rewritten (atomically) each time an answer is ready, so a run killed at any moment
                            leaves a valid submission.
    <out>/logs/<id>.json    one single-line JSON record per question with id, answer, elapsed_ms, thinking, code,
                            output (+ attempts and context).
Questions already logged in <out>/logs are reused unless --force (lets an interrupted run resume); logs of
questions that failed with an error or were cut by the deadline are answered again.

Time budget: --deadline is a unix timestamp. Semantic builds stop starting after --build-deadline (those
businesses fall back to raw sheets); questions not started before --deadline are written as abstain.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from datamate.qa.agent import Agent, load_business  # noqa: E402
from datamate.core.llm import LLM  # noqa: E402

ABSTAIN = {"type": "abstain"}


def select(questions, per_biz, bizs, seed=0):
    if bizs:
        questions = [q for q in questions if q["business_id"] in bizs]
    if not per_biz:
        return questions
    by = defaultdict(list)
    for q in questions:
        by[q["business_id"]].append(q)
    rnd = random.Random(seed)
    return [q for b in sorted(by) for q in rnd.sample(by[b], min(per_biz, len(by[b])))]


def build_all(llm, rulebook, work, split, bizs, jobs, deadline):
    from datamate.semantic.mapper import Mapper, automap_enabled
    mapper = Mapper(llm, rulebook, work)

    def one(b):
        t = time.time()
        if deadline and time.time() > deadline:
            if automap_enabled():
                r = mapper.auto_build(b, str(split / "data" / b))
                print(f"{b}: build deadline reached -> automap ok={r.get('ok')} {time.time() - t:.0f}s", flush=True)
            else:
                print(f"{b}: build skipped (build deadline reached) -> raw", flush=True)
            return
        try:
            r = mapper.build(b, str(split / "data" / b))
            print(f"{b}: build ok={r.get('ok') if isinstance(r, dict) else r} {time.time() - t:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{b}: semantic build failed: {e!r}", flush=True)
            if automap_enabled():
                r = mapper.auto_build(b, str(split / "data" / b))
                print(f"{b}: automap ok={r.get('ok')} {time.time() - t:.0f}s", flush=True)

    with ThreadPoolExecutor(max(1, jobs)) as ex:
        list(ex.map(one, bizs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--biz", nargs="*")
    ap.add_argument("--per-biz", type=int, default=0, help="sample N questions per business (0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="answer only the first N questions (smoke test)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--build", action="store_true", help="build missing semantic layers before answering")
    ap.add_argument("--build-jobs", type=int, default=4)
    ap.add_argument("--build-deadline", type=float, default=0, help="unix time after which no build starts")
    ap.add_argument("--deadline", type=float, default=0, help="unix time after which no question starts")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    split, out = Path(a.split), Path(a.out)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    questions = select([json.loads(x) for x in (split / "questions.jsonl").read_text().splitlines() if x.strip()],
                       a.per_biz, a.biz)
    if a.limit:
        questions = questions[:a.limit]
    rulebook = (split / "rulebook.md").read_text()
    llm = LLM()

    sub_path = out / "submission.jsonl"
    preds: dict[str, dict] = {}
    lock = threading.Lock()

    def write_submission():
        tmp = sub_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for q in questions:
                f.write(json.dumps({"id": q["id"], **preds.get(q["id"], ABSTAIN)}, ensure_ascii=False) + "\n")
        tmp.replace(sub_path)

    write_submission()  # abstain everywhere until answers arrive: valid even if the run dies during the build

    bizs = sorted({q["business_id"] for q in questions})
    if a.build:
        build_all(llm, rulebook, a.work, split, bizs, a.build_jobs, a.build_deadline)

    agent = Agent(llm, rulebook, str(split / "calendar.csv"))
    for b in bizs:
        tables, meta, source = load_business(a.work, split, b)
        agent.set_business(b, tables, meta, source)
        print(f"{b}: {source} ({len(tables)} tables)", flush=True)

    t0 = time.time()

    def logged(lp):
        """A finished answer from an earlier run; None for no log, an unreadable one, or one that ended in an error."""
        if a.force or not lp.exists():
            return None
        try:
            d = json.loads(lp.read_text())
        except ValueError:
            return None
        return None if d.get("error") else d.get("answer") or d.get("final") or ABSTAIN

    def one(q):
        lp = logs / f"{q['id']}.json"
        pred = logged(lp)
        if pred is not None:
            pass
        elif a.deadline and time.time() > a.deadline:
            pred = ABSTAIN
            lp.write_text(json.dumps({"id": q["id"], "answer": pred, "elapsed_ms": 0, "thinking": None,
                                      "code": None, "output": None, "error": "deadline reached"}))
        else:
            t = time.time()
            try:
                r = agent.answer(q)
                pred, log = r.pred, r.log
            except Exception as e:  # noqa: BLE001
                pred = ABSTAIN
                log = {"id": q["id"], "answer": pred, "elapsed_ms": int((time.time() - t) * 1000),
                       "thinking": None, "code": None, "output": None, "error": repr(e)}
            lp.write_text(json.dumps(log, ensure_ascii=False, default=str))
        with lock:
            preds[q["id"]] = pred
            write_submission()
        return pred

    with ThreadPoolExecutor(a.jobs) as ex:
        for i, _ in enumerate(ex.map(one, questions), 1):
            if i % 20 == 0 or i == len(questions):
                print(f"{i}/{len(questions)} answered, {time.time() - t0:.0f}s", flush=True)

    print(f"wrote {sub_path} ({len(preds)} answers)")


if __name__ == "__main__":
    main()
