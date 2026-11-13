# DataMate solution (Zalo AI Challenge 2026)

## Run

```bash
bash /code/predict.sh        # no arguments: reads /data, writes /result
```

All settings are fixed in `/code/flags.sh` and `predict.sh`: model, adapters, seed, temperatures, number of samples, batch limits and timeouts.

`predict.sh` runs the pipeline twice, one LoRA adapter per pass, and pools the votes:

1. **Pass 1 (LoRA v3).** Drafts every business's layers (views, debt macros) and answers every question. The submission is
   complete after this pass.
2. **Pass 2 (LoRA v5).** Answers the same questions on pass 1's layers, as long as time is left. A business it does not reach keeps
   pass 1's answers.
3. **Pooled vote** (`src/analysis/pool_runs.py`). Every sample of both passes votes once, with the same rule as a single pass. Where
   the two adapters disagree, the answer is abstained. The submission and each log record's `answer` become the pooled answer.

Each pass starts one inference process per visible GPU, each on its own set of businesses. There is no tensor parallelism.
Everything stops at 5 h 50 min. Outputs:

| Path | Content |
| --- | --- |
| `/result/submission.jsonl` | One line per question id: `id`, `type`, `answer` (`reason` for unanswerable). Each process rewrites it after every business, so answers saved before a stop count |
| `/result/logs/<id>.json` | One log record per question, on one line (fields below) |
| `/result/work/shard<i>/` | Per-GPU working files: the canonical views and debt macros each business got (`logs/_views_<biz>.json`, `logs/_debt_<biz>.json`), `trace.jsonl`, `submission.jsonl` |
| `/result/work/shard<i>.log` | Process log: model load time, layer drafting, per-business timings |
| `/result/work2/shard<i>/`, `work2/shard<i>.log` | Pass 2 traces and process logs (it reads pass 1's layers) |
| `/result/work/pool/` | The pooled vote: `submission.jsonl` and `pool.jsonl` (per question: the pooled vote and each pass's answer) |

**Log record fields:**

| Field | Content |
| --- | --- |
| `id` | Question id |
| `answer` | The final answer object, as in the submission |
| `elapsed_ms` | Milliseconds from the question entering its batch to its final answer. All questions of a business share one batch, which also covers reading the data and building its views. The per-business view drafts made at the start are shared preparation, like loading the model |
| `thinking` | One string per model call for this question, in call order, as generated: the text inside `<think>…</think>` for thinking-mode samples, and the notes written before the SQL for plain samples and repairs. `null` when no call produced reasoning |
| `code` | Every SQL query run against the business's DuckDB database, in order, as executed |
| `output` | The typed result of each query, or `{"error": …}` |
| `calls` | Every generation in full: kind, sample index, thinking mode, adapter on/off |
| `executions` | Every execution attempt, including replies rejected before running |
| `decision` | The vote: `src/datamate/voting.py::vote, final_answer`, with its inputs, the winning answer and its share |
| `shop_layers` | Files with the views and debt macros the SQL reads |
| `ensemble` | Added by the pooled vote: each pass's own answer, the pooled vote, its share and the number of voters. The other fields describe pass 1 |

## Approach

1. **Reading data** (`src/datamate/ingest.py`).
   - Each business's CSV, XLSX and SQLite files are loaded into an in-memory DuckDB database. Headers, dates, amounts and names are
     normalised.
   - Per business, the model drafts the mapping of the raw tables onto two canonical views, `v_sales` and `v_orders`
     (`views.py`). It writes 8 drafts (8 more when they disagree), which are checked against the data (failed checks come back together), repaired up to three times and grouped by result. A draft also returns the quantity and unit as recorded, so a failed unit conversion is redone from the prices.
     A view is used when at least 2 of the 8 drafts agree.
   - A debt layer is built the same way (`debts.py`): macros `debt_at`, `debt_new` and `debt_repaid`.
   - The prompt always keeps the README and the full schema of the raw tables. The views are added on top.
2. **Generating and executing code** (`src/datamate/pipeline.py`, `sqlexec.py`).
   - **Prompt.** Rules, the split's rulebook, README and schema, the views and debt macros, and hints computed from the question (date
     windows, matching values). One of the two prompts also gets 4 similar train examples, retrieved by word and character-trigram
     overlap with no extra model.
   - **Plain samples.** The model writes 8 samples: short notes, a `TYPE:` line, then one DuckDB SELECT. Settings: temperature 0.7,
     adapter on.
   - **Execution.** Each query runs read-only, with a 20 s timeout and no external access.
   - **Repairs.** A failed sample gets up to 2 greedy repair turns with the error message.
   - **Thinking samples.** A question whose vote is under 75%, and a debt question, gets 4 thinking-mode samples (16,384-token budget).
3. **Choosing the answer** (`src/datamate/voting.py`).
   - The typed results vote.
   - The answer is given when the largest group holds at least half of the samples. Otherwise the question is abstained. Zero or
     empty answers are also abstained.
   - `unanswerable` is answered when every period the question names lies outside the business's data, or when unanswerable wins the vote.

## Models

| Model | Source | Revision | Original release | License | Parameters |
| --- | --- | --- | --- | --- | ---: |
| Qwen3.5-9B (base, `/code/models/Qwen3.5-9B`) | https://huggingface.co/Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 2026-02-27 (repo); revision 2026-03-02 | Apache-2.0 | 9,653,104,368 |
| LoRA v3 (ours, `/code/models/lora`, pass 1) | trained on Qwen3.5-9B, see `training_code/` | `training_code/data_manifest.json` | 2026-11-06 | MIT (ours), from an Apache-2.0 base | 43,278,336 |
| LoRA v5 (ours, `/code/models/lora_v5`, pass 2) | continued from v3, see `training_code/` | `training_code/data_manifest.json` | 2026-11-12 | MIT (ours), from an Apache-2.0 base | 43,278,336 |
| **Total at inference** | | | | | **9,739,661,040 (≤ 15B)** |

- vLLM 0.32.0 runs the model with fp8 online quantization and an fp8 KV cache.
- It is a single model with two adapters of the same rank, used one per pass. There are no embedding, reranking or classification models.
- **Data generation only:** avyukth/Qwen3.8-27B-AWQ-INT4 (revision `5a2ee524cbc9f3ac5df12729ecf0c4b57f311192`, 2026-08-15,
  Apache-2.0) was run locally on train questions. It produced replies for 26 train questions in the v3 data and is not in the image.

## Data

- **Sources.** The provided train split, data we generated from it, and synthetic shops from our own generator (LoRA v5). We never
  trained on public or private test data and never collected test answers. The public test was used only for leaderboard submissions.
- **Synthetic data (v5).** Our generator renders shops into messy files and computes every answer in code; a question is kept only when
  its SQL gives that answer on the rendered files. 500 rows were written by an OpenAI model on synthetic shops only (the organizers
  allowed API use for synthetic data); no public or private test data was ever sent.
- **SFT data.** The pipeline's own samples on train questions, kept when they match `answers.jsonl` (rejection sampling), plus the 26
  teacher replies above.
- **Counts.** v1 1,800 rows (1,636 trained), v2 2,400 (2,398), v3 2,910 (2,909), v5 4,300 (4,300). Details are in `training_code/README.md`.
- **Validation.** 632 questions of 6 train shops held out of v1 and v2.
- **Few-shot bank.** `/code/assets/fewshot_bank.jsonl` holds train questions whose SQL reads only the canonical views and ran to the
  train answer. It is built by `training_code/data_generation/build_fewshot_bank.py`.

## Folder layout

```
/code/predict.sh, flags.sh, README.md, requirements.txt, model_manifest.json
/code/src/datamate/          inference code (PYTHONPATH=/code/src)
/code/models/Qwen3.5-9B/     base model weights, tokenizer, config
/code/models/lora/           LoRA v3 adapter (pass 1)
/code/models/lora_v5/        LoRA v5 adapter (pass 2)
/code/src/analysis/pool_runs.py   the pooled vote of both passes
/code/assets/fewshot_bank.jsonl, fewshot_bank_v2.jsonl   few-shot banks of pass 1 and pass 2
/code/training_code/         training code, data, logs, checkpoints, data_manifest.json (its README explains how to retrain)
/opt/venv-train/             training environment (requirements in training_code/requirements.txt)
```

## Reproducibility and time

- **Seeds.** vLLM is seeded (seed 0) and every sampling request passes a fixed seed. Repairs are greedy.
- **Batching.** GPU batching can still change sampled outputs between runs. Before submitting, the image is run twice on the same input and the two submissions are compared (guide, section 6).
- **Public test run.** On one 24 GB GPU (RTX PRO 4000 Blackwell), 1,000 questions took about 1.3 h of layer drafting, 3 h for pass 1
  and 3 h for pass 2. Public score of this two-pass setup: 0.800.
- **Local check.** The same entry point runs outside the image with every path overridden by environment variables (`CODE_DIR`,
  `FLAGS_FILE`, `MODEL_DIR`, `LORA_DIR`, `LORA2_DIR`, `ASSETS_DIR`, `POOL_SCRIPT`, `PYTHON`, `DATA_DIR`, `RESULT_DIR`, `NUM_GPUS`).
- **Private test estimate.** On 2x RTX 5090 both passes should fit in 6 h; if pass 2 runs out of time, the businesses it reached are
  pooled and the others keep pass 1's answers.
- **Not yet validated.** The image has not yet been timed on an RTX 5090.
- **Time guard.** Each process skips thinking samples for a business when it falls behind its time budget, and stops at 5 h 50 min.
