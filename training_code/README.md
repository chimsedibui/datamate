# Training code

**No training, fine-tuning or data generation was performed.**

Qwen3-14B is used as released; see `/code/README.md` for its source, revision and checksums. The solution consists
of prompts, validation rules and helper code under `/code/src`, all of which ship with the inference image.

The provided training split was used only to develop and evaluate those prompts and rules
(`python src/evaluate.py --pred <submission.jsonl> --split <train dir>`). It was not used to train any weights.
