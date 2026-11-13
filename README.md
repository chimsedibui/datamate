# DataMate — Zalo AI Challenge 2026

Answers Vietnamese questions over small-business data files (CSV / XLSX / SQLite) by having a local LLM
write and execute pandas code. Fully offline inside the Docker image.

## Run

```bash
bash /code/predict.sh          # no arguments; reads /data, writes /result
```

- Output: `/result/submission.jsonl`, always complete and in question order: abstain for every question at the start,
  rewritten each time an answer is ready.
- Logs: `/result/logs/<question_id>.json`, one single-line record per question with `id`, `answer`, `elapsed_ms`,
  `thinking` (raw model reasoning per call, or `null`), `code` (the executed code), `output`, and every attempt.
- Run logs: `/result/logs/_run/` (vLLM server, semantic builds, progress).

Fixed settings, in `predict.sh`:
- vLLM 0.10.1.1, Qwen3-14B bf16, tensor parallel 2, prefix caching.
- Self-consistency vote: up to 3 solves per question (the first greedy, the others at temperature 0.7 with seeds
  43, 44, each with its own prompt note: plain / plan first / settle unit and scope; `--seed 0` on the server),
  thinking mode off. The answer is kept only when 2 solves agree, compared the
  way the scorer compares answers; otherwise `abstain`. The third solve runs only when the first two disagree.
- 16 concurrent questions.
- Time guards: no new semantic build after 1.5 h; one greedy solve (no vote) for questions started in the last
  45 min; no new question in the last 15 min of the 6 h limit; the answering
  process is stopped 5 min before the limit (questions still running stay `abstain`).

## Approach

1. **Semantic layer (per business, once).**
   - A short first LLM call summarises the README and file profile (industry, entities, units and conversions,
     quirks); the summary is added to the loader prompt and the units to the question-answering context.
   - The LLM reads the business README, a profile of every file and sheet, and the rulebook, plus the 2 most similar
     validated loaders from the training businesses as few-shot examples (`src/datamate/semantic/loader_examples.py`).
   - Per business: at most 25 min, 12k thinking tokens per attempt; only the latest attempt and its feedback are
     resent. If no attempt validates, a layer whose core tables are structurally sound and whose remaining issues are
     data-quality warnings is accepted (`meta['soft_errors']`) rather than falling back to raw sheets.
   - It writes a `load(raw)` function that maps the raw files into a canonical schema:
     `orders`, `order_lines`, `products`, `unit_conversions`, `customers`, `payments`, `purchases`, `stock`,
     `returns`, `deliveries`, … plus `extra_*` tables for anything else.
   - `src/datamate/semantic/validate.py` checks the result: types, required columns, status mapping, money scale,
     monthly revenue consistency and duplicate handling.
   - Static lint catches common loader mistakes. Unambiguous structural slips are repaired instead of failing the
     attempt (duplicate column names coalesced, non-schema columns dropped, unknown tables renamed `extra_*`), and
     every load normalises walk-in customer markers to null and rebuilds purchase line amounts stored as receipt
     totals.
   - Errors go back to the LLM, up to 5 attempts and one review round. Non-core tables that keep failing are dropped.
   - If no valid layer is produced, questions use the raw sheets instead.
2. **Question answering.**
   - Deterministic pre-analysis (`src/datamate/qa/qparse.py`): periods resolved per the rulebook, data values matching
     names in the question, likely answer type, chat abbreviations expanded (dthu, bn, hqua…), and which packaged
     helper fits the question (debt, quantity in a unit, price per unit, customer ranking).
   - The LLM gets the rulebook rules, a summary of the tables, the pre-analysis and the question with `as_of_date`,
     and writes `solve()` using packaged helpers (`src/datamate/qa/helpers.py`): valid orders and sales lines,
     week/month ranges, Tết window from `calendar.csv`, fuzzy name matching, `qty_in` / `avg_price` (units and
     conversions), `with_customer` (customer identity per rulebook §3), `debts` / `debt_flow` (rulebook §4).
   - The code runs in a forked process with a timeout (process isolation only: the container itself is offline). On error it is retried with the traceback (3 attempts).
   - The result is checked against the data (`src/datamate/qa/checks.py`):
     - every string literal the code filters on must exist in its column; a literal that only differs by case,
       diacritics or spacing from one stored value is replaced automatically, otherwise the stored candidates are
       sent back to the LLM;
     - 0/empty results, shares outside 0–100 %, wrong number of list items, walk-in/null entries in rankings and
       "unanswerable" claims contradicted by the data range get one feedback round each (2 rounds in total).
   - The returned value is normalized to the submission format (`src/datamate/qa/agent.py`); money ≥ 1000 is rounded to
     whole VND.
   - Answerability gate: the answer becomes `abstain` when the evidence says it is probably wrong (0/empty result,
     filter values not found in the data, impossible share) or when the business has no valid semantic layer.
     If no attempt succeeds, the answer is `abstain`. A gated answer whose periods all lie outside the shop's order
     dates becomes `unanswerable` (rulebook §7). Numbers round half up (rulebook: "nửa đồng làm tròn lên").
   - Self-consistency (`DM_SC=3`): steps above run up to 3 times (greedy, then sampled). Answers are grouped by the
     scorer's own equality (number exact, percent to 2 decimals, normalised strings, ordered lists, tables with
     rows in any order); the answer needs 2 agreeing solves, else `abstain`. Sampling stops as soon as further
     solves cannot change the outcome.
3. Nothing is keyed on question or business ids. Each business's mapping is generated from its own files at run time.

Code map:
- `predict.sh`: container entry point (starts vLLM, then `src/predict.py --build`).
- `src/predict.py`: build missing semantic layers, answer every question. `src/build_semantic.py`: build only.
  `src/evaluate.py`: local scorer replicating the competition rules.
- `src/datamate/core/`: shared infrastructure — `llm` (client + disk cache), `rawio` (CSV/XLSX/SQLite access),
  `parsers` (money, dates, quantities, status), `pdcompat` (pandas 3 shims), `scoring`.
- `src/datamate/semantic/`: semantic layer — `mapper` (prompts, build loop), `validate`, `schema`, `profile`,
  `automap` (deterministic fallback, off by default), `loader_examples` (+ `.json`, the few-shot library).
- `src/datamate/qa/`: question answering — `qnorm`, `qparse` (pre-analysis), `agent`, `helpers`, `checks`.
- `scripts/release/`: model download, image checks and export, output validation, few-shot library builder.
  `scripts/dev/`: API runs, subsets, log scoring. `scripts/gcp/`: test VM.
- `training_code/`: provenance of the few-shot loaders.

## Models

| Name | Source | Revision | Released | License | Params |
|---|---|---|---|---|---|
| Qwen3-14B | https://huggingface.co/Qwen/Qwen3-14B | 40c069824f4251a91eefaf281ebe4c544efd3e18 | 2025-04-29 | Apache-2.0 | 14.8B |

This is the only model used at inference (total ≤ 15B). It is not fine-tuned. Weights are in `/code/models/Qwen3-14B`
and were downloaded with `scripts/release/download_model.sh`; checksums are in `models/Qwen3-14B.sha256`.

## Data

- No training was performed. Validated loaders built for the training businesses during development ship as
  few-shot examples (not training data); generator, prompts/responses and checksums are in `training_code/`.
- The provided training split (30 businesses, 3000 questions) was used only to develop prompts, validation rules
  and helpers, and to measure accuracy with `src/evaluate.py`.
- Public/private test answers were never used.

## Build, check and export the image (developer)

```bash
bash scripts/release/download_model.sh                       # pinned Qwen3-14B revision + checksums -> models/
docker build --platform linux/amd64 -t zaic2026_datamate_<team_name>:final .
TEAM=<team_name> bash scripts/release/check_submission.sh static                              # platform, size, layout, offline imports, weight checksums
TEAM=<team_name> bash scripts/release/check_submission.sh run data/public_test result/        # official scoring command (2 GPUs) + output check
TEAM=<team_name> bash scripts/release/check_submission.sh compare data/public_test run1/ run2/ # reproducibility
TEAM=<team_name> bash scripts/release/export_image.sh export/                                 # .tar.gz, uncompressed size, .md5
```

`scripts/dev/make_subset.py` builds smaller /data folders for smoke tests; `scripts/release/validate_output.py` checks a /result
folder (submission coverage and format, required log fields). `scripts/gcp/` runs all of this on a 2 × L4 test VM.

Local development uses any OpenAI-compatible endpoint via `.env` (see `.env.example`).
