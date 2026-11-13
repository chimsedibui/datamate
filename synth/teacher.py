"""OpenAI as the teacher for synthetic shops: the exact prompts inference sends, answered by an OpenAI model, executed on the
shop's DuckDB, scored against the generator's answers. Only replies that reach the right answer become SFT rows.

    # pilot: a few questions, called directly, to measure accuracy and cost before spending the budget
    python -m synth.teacher pilot --split data/synth_train --views runs/goldviews_synth_train --n 40 --model gpt-6-luna --out runs/teacher_pilot
    # batch: write the requests, submit them (Batch API, half price), then collect, execute and score
    python -m synth.teacher submit  --split data/synth_train --views runs/goldviews_synth_train --ids IDS --model gpt-6-luna --out runs/teacher_b1
    python -m synth.teacher collect --split data/synth_train --views runs/goldviews_synth_train --out runs/teacher_b1

The key is read from OPENAI_API_KEY, or from the repository's .env (git-ignored); it is never printed or written.
Prompts carry only synthetic data and the train rulebook: no public or private test data is ever sent.
"""
import argparse
import json
import os
import random
import re
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PRICES = {  # USD per 1M tokens: input, cached input, output (standard tier; Batch API is half)
    "gpt-6-luna": (0.10, 0.01, 0.50), "gpt-5-mini": (0.25, 0.025, 2.00), "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.4-mini": (0.75, 0.075, 4.50), "gpt-6.1-terra": (1.00, 0.05, 5.00), "gpt-6.1-sol": (2.00, 0.10, 10.00),
}
FLAGS = ["--notes", "--canon-names", "--answer-unanswerable", "--fewshot", "assets/fewshot_bank.jsonl", "--fewshot-k", "4"]


def load_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*)\s*$", line)
            if m:
                os.environ["OPENAI_API_KEY"] = m.group(1).strip().strip('"').strip("'")


def cost(model: str, usage: dict, batch: bool = False) -> float:
    pin, pcache, pout = PRICES.get(model, (0, 0, 0))
    cached = usage.get("cached", 0)
    c = ((usage.get("prompt", 0) - cached) * pin + cached * pcache + usage.get("completion", 0) * pout) / 1e6
    return c / 2 if batch else c


class Prompts:
    """Pipeline prompts for synthetic questions, built offline (no model) on the gold views."""

    def __init__(self, split: Path, views_dir: Path, extra: list | None = None, fewshot: str | None = None):
        from datamate.pipeline import Pipeline
        from datamate.predict import parse_args
        self.tmp = Path(tempfile.mkdtemp())
        flags = list(FLAGS) + list(extra or [])
        if fewshot:
            flags[flags.index("--fewshot") + 1] = fewshot
        a = parse_args(["--model", "none", "--output", str(self.tmp / "out"), "--data", str(split), "--views-from", str(views_dir)] + flags)
        (a.output / "logs").mkdir(parents=True)
        self.pipe, self.split = Pipeline(a), split
        self.gold = {json.loads(l)["id"]: json.loads(l) for l in open(split / "answers.jsonl", encoding="utf-8")}
        self.views = {p.stem.replace("_views_", "") for p in (views_dir / "logs").glob("_views_*.json")}

    def shops(self, questions: list[dict]):
        """Yield (shop, its questions, their chats); each chat is [system, user] for one of the two prompts."""
        by = defaultdict(list)
        for q in questions:
            if q["business_id"] in self.views:
                by[q["business_id"]].append(q)
        loaded = self.pipe.load_shops(sorted(by))
        self.pipe.reuse_layers(loaded)
        for b, (db, meta) in loaded.items():
            shop = self.pipe.setup_shop(b, db, meta)
            todo = by[b]
            yield shop, todo, self.pipe.build_chats(shop, todo)
            db.close()                                   # the caller is done with this shop when it asks for the next one


def check(shop, q: dict, text: str, gold: dict) -> tuple[bool, str]:
    from datamate.scoring import score
    from datamate.sqlexec import execute
    try:
        pred = execute(shop.db, text, q["question"])
    except Exception as e:  # noqa: BLE001
        return False, "error: " + str(e)[:160]
    return score(pred, gold) == 1, json.dumps(pred, ensure_ascii=False)[:160]


def sft_row(chat: list, reply: str, q: dict) -> dict:
    return {"system": chat[0]["content"], "user": chat[1]["content"], "reply": reply.strip(), "id": q["id"], "thinking": False,
            "source": "openai"}


def pick_questions(split: Path, views: set, n: int | None, ids: Path | None, seed: int = 0) -> list[dict]:
    qs = [json.loads(l) for l in open(split / "questions.jsonl", encoding="utf-8")]
    qs = [q for q in qs if q["business_id"] in views]
    if ids:
        keep = set(ids.read_text().split())
        return [q for q in qs if q["id"] in keep]
    random.Random(seed).shuffle(qs)
    per = Counter()
    out = []
    for q in qs:                                    # spread over shops
        if per[q["business_id"]] < 2:
            out.append(q)
            per[q["business_id"]] += 1
        if n and len(out) >= n:
            break
    return out


def call(client, model: str, messages: list, max_tokens: int, effort: str | None) -> tuple[str, dict]:
    kw = {"model": model, "messages": messages, "max_completion_tokens": max_tokens}
    if effort:
        kw["reasoning_effort"] = effort
    r = client.chat.completions.create(**kw)
    u = r.usage
    details = getattr(u, "prompt_tokens_details", None)
    usage = {"prompt": u.prompt_tokens, "completion": u.completion_tokens, "cached": getattr(details, "cached_tokens", 0) or 0}
    return r.choices[0].message.content or "", usage


def pilot(a) -> None:
    load_key()
    from openai import OpenAI
    client = OpenAI()
    P = Prompts(a.split, a.views)
    qs = pick_questions(a.split, P.views, a.n, a.ids)
    a.out.mkdir(parents=True, exist_ok=True)
    rows, recs, total = [], [], Counter()
    usage_sum = Counter()
    t0 = time.time()
    for shop, todo, chats in P.shops(qs):
        jobs = [(q, j, chat) for q, cs in zip(todo, chats) for j, chat in enumerate(cs)]
        with ThreadPoolExecutor(8) as ex:
            outs = list(ex.map(lambda x: call(client, a.model, x[2], a.max_tokens, a.effort), jobs))
        for (q, j, chat), (text, usage) in zip(jobs, outs):
            ok, info = check(shop, q, text, P.gold[q["id"]])
            usage_sum.update(usage)
            total["samples"] += 1
            total["right"] += ok
            recs.append({"id": q["id"], "j": j, "ok": ok, "gold_type": P.gold[q["id"]]["type"], "info": info, "usage": usage, "reply": text})
            if ok:
                rows.append(sft_row(chat, text, q))
        print(f"{shop.name}: {sum(r['ok'] for r in recs if r['id'] in {q['id'] for q in todo})}/{2 * len(todo)} right  {time.time() - t0:.0f}s", flush=True)
    by_type = defaultdict(Counter)
    for r in recs:
        by_type[r["gold_type"]]["n"] += 1
        by_type[r["gold_type"]]["ok"] += r["ok"]
    spent = cost(a.model, usage_sum)
    summary = {"model": a.model, "effort": a.effort, "samples": total["samples"], "right": total["right"],
               "usage": dict(usage_sum), "cost_usd": round(spent, 4), "cost_per_right_usd": round(spent / max(1, total["right"]), 5),
               "by_type": {k: f"{v['ok']}/{v['n']}" for k, v in by_type.items()}}
    (a.out / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    (a.out / "sft.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def submit(a) -> None:
    load_key()
    from openai import OpenAI
    client = OpenAI()
    P = Prompts(a.split, a.views)
    qs = pick_questions(a.split, P.views, a.n, a.ids)
    a.out.mkdir(parents=True, exist_ok=True)
    req = a.out / "requests.jsonl"
    with open(req, "w", encoding="utf-8") as f, open(a.out / "chats.jsonl", "w", encoding="utf-8") as fc:
        for shop, todo, chats in P.shops(qs):
            for q, cs in zip(todo, chats):
                for j, chat in enumerate(cs):
                    body = {"model": a.model, "messages": chat, "max_completion_tokens": a.max_tokens}
                    if a.effort:
                        body["reasoning_effort"] = a.effort
                    f.write(json.dumps({"custom_id": f"{q['id']}|{j}", "method": "POST", "url": "/v1/chat/completions", "body": body},
                                       ensure_ascii=False) + "\n")
                    fc.write(json.dumps({"custom_id": f"{q['id']}|{j}", "q": q, "chat": chat}, ensure_ascii=False) + "\n")
    upload(client, a.out, a.model)


def upload(client, out: Path, model: str, per_file: int = 2000) -> None:
    """Submit requests.jsonl as several batches: the Batch API refuses input files over 200 MiB."""
    lines = (out / "requests.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    ids = []
    for k in range(0, len(lines), per_file):
        part = out / f"requests_{k // per_file:02d}.jsonl"
        part.write_text("".join(lines[k:k + per_file]), encoding="utf-8")
        up = client.files.create(file=open(part, "rb"), purpose="batch")
        ids.append(client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h").id)
    (out / "batch.json").write_text(json.dumps({"batch_ids": ids, "model": model, "requests": len(lines)}))
    print(f"submitted {len(ids)} batches: {len(lines)} requests")


def resubmit(a) -> None:
    load_key()
    from openai import OpenAI
    upload(OpenAI(), a.out, a.model)


def collect(a) -> None:
    load_key()
    from openai import OpenAI
    client = OpenAI()
    info = json.loads((a.out / "batch.json").read_text())
    batches = [client.batches.retrieve(i) for i in info.get("batch_ids") or [info["batch_id"]]]
    for b in batches:
        print("batch", b.id, b.status, b.request_counts)
    if any(b.status == "failed" for b in batches):          # refused at validation: there is no output at all
        print("batch", "-", "failed")
        return
    # a cancelled or expired batch still returns the requests it finished: use them (partial results)
    done = ("completed", "cancelled", "expired")
    if any(b.status not in done for b in batches):
        return
    if any(b.status != "completed" for b in batches):
        print("partial results:", sum(b.request_counts.completed for b in batches), "of", sum(b.request_counts.total for b in batches))
    lines = [l for b in batches if b.output_file_id for l in client.files.content(b.output_file_id).text.splitlines()]
    replies = {}
    usage_sum = Counter()
    for line in lines:
        r = json.loads(line)
        body = (r.get("response") or {}).get("body") or {}
        if not body.get("choices"):
            continue
        replies[r["custom_id"]] = body["choices"][0]["message"]["content"] or ""
        u = body.get("usage", {})
        usage_sum.update({"prompt": u.get("prompt_tokens", 0), "completion": u.get("completion_tokens", 0),
                          "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)})
    chats = {json.loads(l)["custom_id"]: json.loads(l) for l in open(a.out / "chats.jsonl", encoding="utf-8")}
    P = Prompts(a.split, a.views)
    qs = [c["q"] for c in chats.values()]
    uniq = list({q["id"]: q for q in qs}.values())
    rows, n_ok, n = [], 0, 0
    for shop, todo, _ in P.shops(uniq):
        for q in todo:
            for j in (0, 1):
                cid = f"{q['id']}|{j}"
                if cid not in replies:
                    continue
                n += 1
                ok, _ = check(shop, q, replies[cid], P.gold[q["id"]])
                if ok:
                    n_ok += 1
                    rows.append(sft_row(chats[cid]["chat"], replies[cid], q))
    (a.out / "sft.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    spent = cost(info["model"], usage_sum, batch=True)
    print(json.dumps({"replies": len(replies), "scored": n, "right": n_ok, "rows": len(rows), "usage": dict(usage_sum), "cost_usd": round(spent, 4)}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["pilot", "submit", "resubmit", "collect"])
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--views", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--n", type=int)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--effort", choices=["minimal", "low", "medium", "high"], help="reasoning effort, for reasoning models")
    a = ap.parse_args()
    {"pilot": pilot, "submit": submit, "resubmit": resubmit, "collect": collect}[a.mode](a)


if __name__ == "__main__":
    main()
