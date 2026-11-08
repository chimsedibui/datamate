#!/usr/bin/env bash
# Measure the pipeline as the organizers will run it: semantic layers built by Qwen3-14B (not gpt-4o), then QA.
# Uses qwen/qwen3-14b on OpenRouter (same model, 40960 context like the container) instead of a GPU VM.
#
# Usage:
#   bash scripts/measure_qwen_build.sh                 # build 30 train businesses + QA on the 150-question sample
#   EXAMPLES=0 bash scripts/measure_qwen_build.sh      # A/B: builds without few-shot loader examples
#   STAGES=build bash scripts/measure_qwen_build.sh    # build only (STAGES=qa for QA only)
#
# Cost (OpenRouter list price $0.12 / $0.24 per 1M tokens in/out): ~$1-1.5 for the builds, ~$0.5 for QA.
# LLM_BUDGET_USD caps each process; the prices below make that cap count real Qwen prices (not gpt-4o).
set -euo pipefail
cd "$(dirname "$0")/.."

EXAMPLES=${EXAMPLES:-1}
STAGES=${STAGES:-"build qa"}
TAG=${TAG:-qwen$([ "$EXAMPLES" = 1 ] && echo _ex || echo _noex)}
WORK=work/train_$TAG
OUT=out/train_$TAG
KEY=$(grep -E '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)
[ -n "$KEY" ] || { echo "OPENROUTER_API_KEY missing in .env"; exit 1; }

# Same model and switches as predict.sh inside the image; explicit env wins over .env (python-dotenv, no override).
export LLM_BASE_URL=https://openrouter.ai/api/v1 LLM_API_KEY="$KEY" LLM_MODEL=qwen/qwen3-14b
export BUILD_LLM_BASE_URL=https://openrouter.ai/api/v1 BUILD_LLM_API_KEY="$KEY" BUILD_LLM_MODEL=qwen/qwen3-14b
export LLM_ENABLE_THINKING=0 LLM_CONTEXT=40960 BUILD_EXAMPLES=$EXAMPLES
export LLM_PRICE_IN=0.12 LLM_PRICE_OUT=0.24 LLM_BUDGET_USD=${LLM_BUDGET_USD:-4}
unset LLM_FALLBACK_MODEL LLM_FALLBACK_BASE_URL LLM_FALLBACK_API_KEY
PY=${PY:-.venv/bin/python}
mkdir -p "$OUT"

if [[ " $STAGES " == *" build "* ]]; then
  echo "== build 30 train businesses with Qwen3-14B (examples=$EXAMPLES) -> $WORK"
  $PY src/build_semantic.py --split data/train --work "$WORK" --jobs 8 2>&1 | tee "$OUT/build.log" \
    | grep -E "ok=|businesses mapped" || true
fi

if [[ " $STAGES " == *" qa "* ]]; then
  echo "== QA on the 150-question sample (same as q14e) with layers from $WORK"
  $PY src/predict.py --split data/train --work "$WORK" --out "$OUT" --per-biz 5 --jobs 6 > "$OUT/predict.log" 2>&1
  $PY src/evaluate.py --pred "$OUT/submission.jsonl" --split data/train | head -1
fi

$PY - "$WORK" "$OUT" <<'EOF'
import json, sys
from pathlib import Path
work, out = Path(sys.argv[1]), Path(sys.argv[2])
metas = {p.parent.name: json.loads(p.read_text()) for p in work.glob("*/meta.json")}
ok = [b for b, m in metas.items() if m.get("ok")]
soft = [b for b in ok if metas[b].get("soft_errors")]
print(f"layers: {len(ok)}/{len(metas)} ok ({len(soft)} accepted with warnings); "
      f"baseline gpt-4o layers: 25/30")
ev = out / "eval.json"
if ev.exists():
    d = json.loads(ev.read_text())
    print(f"QA: PA={d['PA']:.4f} correct={d['correct']} wrong={d['wrong']} abstain={d['abstain']} "
          f"(q14e with gpt-4o layers: PA 0.447, 91/24/35)")
EOF
