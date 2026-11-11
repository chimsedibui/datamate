#!/usr/bin/env bash
# Public-test submissions through APIs (portal Step 2: "Based on 3rd APIs"): semantic layers are built once
# (same settings as predict.sh in the image), then QA runs on those same layers per model.
# Build provider: PROVIDER=ntq (default, Qwen 27B "nxchat" at api.ntq.ai, key NTQ_API_KEY in .env)
#                 PROVIDER=openrouter (Qwen3-14B, key OPENROUTER_API_KEY).
#
# Usage:
#   bash scripts/run_public.sh                       # build + qa_ntq (NTQ Qwen 27B)
#   STAGES="build qa_ntq" bash scripts/run_public.sh
#   PROVIDER=openrouter STAGES="build qa_qwen qa_gpt4o" bash scripts/run_public.sh
#   STAGES="build qa_qwen" bash scripts/run_public.sh
#   STAGES=qa_gpt4o GPT_BUDGET=30 bash scripts/run_public.sh
#   WORK=work/public_mix TAG=mix STAGES=qa_qwen bash scripts/run_public.sh   # QA on another layer folder
#
# Outputs: out/public_<model>/submission.jsonl (upload this) + logs/, layers in work/public_qwen.
# Cost (measured token use, list prices): build ~$0.3, qa_qwen ~$2-3, qa_gpt4o ~$30-50. Each stage has its own cap.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGES=${STAGES:-"build qa_ntq"}
PROVIDER=${PROVIDER:-ntq}
SPLIT=data/public_test
WORK=${WORK:-work/public_$PROVIDER}
PY=${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)}   # else the active env (e.g. conda)
key() { grep -E "^$1=" .env | cut -d= -f2-; }

# Explicit (even empty) values win over .env: python-dotenv does not override variables that are already set.
export BUILD_LLM_BASE_URL="" BUILD_LLM_API_KEY="" BUILD_LLM_MODEL="" BUILD_LLM_MAX_OUTPUT=""
export LLM_FALLBACK_MODEL="" LLM_CONTEXT=40960
export BUILD_MAX_S=1500 BUILD_MAX_TOKENS=12000 BUILD_EXAMPLES=1

qwen_env() {
  export LLM_BASE_URL=https://openrouter.ai/api/v1 LLM_API_KEY="$(key OPENROUTER_API_KEY)" LLM_MODEL=qwen/qwen3-14b
  export LLM_ENABLE_THINKING=0 LLM_PRICE_IN=0.12 LLM_PRICE_OUT=0.24 LLM_BUDGET_USD=${QWEN_BUDGET:-5}
}
ntq_env() {  # self-hosted vLLM, no per-token price; LLM_BUDGET_USD stays off
  export LLM_BASE_URL=https://api.ntq.ai/v1 LLM_API_KEY="$(key NTQ_API_KEY)" LLM_MODEL=nxchat
  export LLM_ENABLE_THINKING=0 LLM_PRICE_IN=0 LLM_PRICE_OUT=0 LLM_BUDGET_USD=0
}
build_env() { case $PROVIDER in ntq) ntq_env ;; openrouter) qwen_env ;; *) echo "unknown PROVIDER $PROVIDER"; exit 2 ;; esac; }
gpt4o_env() {
  export LLM_BASE_URL=https://api.openai.com/v1 LLM_API_KEY="$(key OPENAI_API_KEY)" LLM_MODEL=gpt-4o
  export LLM_ENABLE_THINKING="" LLM_PRICE_IN=2.5 LLM_PRICE_OUT=10 LLM_BUDGET_USD=${GPT_BUDGET:-45}
}

qa() {  # qa <name>
  local out=out/public_$1${TAG:+_$TAG}
  mkdir -p "$out"
  echo "== QA $1 ($LLM_MODEL, cap \$$LLM_BUDGET_USD) -> $out"
  $PY src/predict.py --split "$SPLIT" --work "$WORK" --out "$out" --jobs "${QA_JOBS:-8}" > "$out/predict.log" 2>&1 || true
  tail -n 3 "$out/predict.log"
  python3 scripts/validate_output.py --questions "$SPLIT/questions.jsonl" --result "$out" | tail -n 5
  python3 - "$out/submission.jsonl" <<'EOF'
import collections, json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"{len(rows)} answers:", dict(collections.Counter(r.get("type") for r in rows)))
EOF
}

for s in $STAGES; do
  case $s in
  build)
    build_env
    mkdir -p out/public_build
    echo "== build $(ls $SPLIT/data | wc -l) public businesses with $LLM_MODEL -> $WORK"
    $PY src/build_semantic.py --split "$SPLIT" --work "$WORK" --jobs "${BUILD_JOBS:-8}" --retry-failed 2>&1 \
      | tee -a out/public_build/build.log | grep -E "ok=|businesses mapped" || true ;;
  qa_ntq) ntq_env; qa ntq ;;
  qa_qwen) qwen_env; qa qwen ;;
  qa_gpt4o) gpt4o_env; qa gpt4o ;;
  *) echo "unknown stage $s"; exit 2 ;;
  esac
done
