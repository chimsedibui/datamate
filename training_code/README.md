# Training code

**No training or fine-tuning was performed.** Qwen3-14B is used as released; see `/code/README.md` for its source,
revision and checksums.

## Data generated during development (shipped, not used for training)

`src/datamate/semantic/loader_examples.json` holds 21 semantic-layer loaders (pandas `load(raw)` functions) that passed
validation on the **provided training businesses**. At inference the build step shows the 2 most similar ones to
Qwen3-14B as few-shot examples (`src/datamate/semantic/loader_examples.py`); no weights are trained on them.

- Generator: the same build pipeline (`src/datamate/semantic/mapper.py`, prompts in that file) run on `data/train` with an
  OpenAI-compatible API model during development (gpt-4o via Azure OpenAI, some builds gpt-5.4-mini), temperature 0.
- Prompts/responses, build logs and load reports of every included loader: `loader_examples/<business>/`
  (`messages.json`, `build_log.json`, `report.txt`, `loader.py`), with SHA-256 in `loader_examples/manifest.json`.
- Re-create the library from cached builds: `python scripts/release/build_loader_library.py --work work/train --split data/train`
  (loaders longer than 12000 characters are skipped to fit the prompt budget).
- Only the provided training split was used; public/private test data and answers were never used.

The training split was also used to develop and evaluate prompts and rules
(`python src/evaluate.py --pred <submission.jsonl> --split <train dir>`).
