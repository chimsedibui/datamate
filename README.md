# DataMate — Zalo AI Challenge 2026

Answers Vietnamese questions over small-business data files (CSV / XLSX / SQLite) by having a local LLM
write and execute pandas code. Fully offline inside the Docker image.

## Run

```bash
bash /code/predict.sh          # no arguments; reads /data, writes /result
```

- Output: `/result/submission.jsonl`, written answer by answer and rewritten in question order at the end.
- Logs: `/result/logs/<question_id>.json`, one single-line record per question with `id`, `answer`, `elapsed_ms`,
  `thinking` (raw model reasoning per call, or `null`), `code` (the executed code), `output`, and every attempt.
- Run logs: `/result/logs/_run/` (vLLM server, semantic builds, progress).

Fixed settings, in `predict.sh`:
- vLLM 0.10.1.1, Qwen3-14B bf16, tensor parallel 2, prefix caching.
- Greedy decoding (temperature 0, seed 42 per request, `--seed 0` on the server), thinking mode off.
- 16 concurrent questions.
- Time guards: no new semantic build after 1.5 h; no new question in the last 15 min of the 6 h limit.

## Approach

1. **Semantic layer (per business, once).**
   - The LLM reads the business README, a profile of every file and sheet, and the rulebook.
   - It writes a `load(raw)` function that maps the raw files into a canonical schema:
     `orders`, `order_lines`, `products`, `unit_conversions`, `customers`, `payments`, `purchases`, `stock`,
     `returns`, `deliveries`, … plus `extra_*` tables for anything else.
   - `src/datamate/validate.py` checks the result: types, required columns, status mapping, money scale,
     monthly revenue consistency and duplicate handling.
   - Static lint catches common loader mistakes.
   - Errors go back to the LLM, up to 5 attempts and one review round. Non-core tables that keep failing are dropped.
   - If no valid layer is produced, questions use the raw sheets instead.
2. **Question answering.**
   - For each question the LLM gets the rulebook rules, a summary of the tables and the question with `as_of_date`.
   - It writes `solve()` using packaged helpers (`src/datamate/helpers.py`: valid orders, sales lines,
     week/month ranges, Tết window from `calendar.csv`, fuzzy name matching).
   - The code runs in a forked process with a timeout. On error it is retried with the traceback (3 attempts).
   - The returned value is validated and normalized to the submission format (`src/datamate/agent.py`).
   - If no attempt succeeds, the answer is `abstain`.
3. Nothing is keyed on question or business ids. Each business's mapping is generated from its own files at run time.

Code map:
- `src/predict.py`: entry point.
- `src/build_semantic.py`: build only.
- `src/evaluate.py`: local scorer replicating the competition rules.
- `src/datamate/`: library (`mapper`, `validate`, `parsers`, `rawio`, `profile`, `schema`, `agent`, `helpers`, `llm`, `scoring`).

## Models

| Name | Source | Revision | Released | License | Params |
|---|---|---|---|---|---|
| Qwen3-14B | https://huggingface.co/Qwen/Qwen3-14B | TODO: commit sha | 2025-04-29 | Apache-2.0 | 14.8B |

This is the only model used at inference (total ≤ 15B). It is not fine-tuned. Weights are in `/code/models/Qwen3-14B`
and were downloaded with `scripts/download_model.sh`; checksums are in `models/Qwen3-14B.sha256`.

## Data

- No training and no data generation were performed. See `training_code/README.md`.
- The provided training split (30 businesses, 3000 questions) was used only to develop prompts, validation rules
  and helpers, and to measure accuracy with `src/evaluate.py`.
- Public/private test answers were never used.

## Build the image (developer)

```bash
bash scripts/download_model.sh
docker build --platform linux/amd64 -t zaic2026_datamate_<team_name>:final .
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
  -v $PWD/data/public_test:/data:ro -v $PWD/result:/result zaic2026_datamate_<team_name>:final /code/predict.sh
```

Local development uses any OpenAI-compatible endpoint via `.env` (see `.env.example`).
