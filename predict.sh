#!/usr/bin/env bash
# DataMate entry point: bash /code/predict.sh  (no arguments)
# Reads /data, writes /result/submission.jsonl and one log record per question in /result/logs/<id>.json.
# Run-level logs (vLLM server, build, predict) go to /result/logs/_run/.
#
# All settings below are fixed defaults. The optional env overrides exist only for local smoke tests on smaller
# GPUs; the organizers' run uses the defaults.
set -uo pipefail

START=$(date +%s)
CODE=/code
DATA=${DATA_DIR:-/data}
OUT=${RESULT_DIR:-/result}
WORK=/tmp/datamate/work
RUNLOG="$OUT/logs/_run"

MODEL_DIR=${VLLM_MODEL_DIR:-$CODE/models/Qwen3-14B}
SERVED_NAME=qwen3-14b
TP=${VLLM_TP:-2}
GPU_UTIL=${VLLM_GPU_UTIL:-0.88}
MAX_LEN=${VLLM_MAX_LEN:-40960}
PORT=8000

TIME_LIMIT=21600                              # 6 h, including model loading
BUILD_DEADLINE=$((START + 5400))              # no new semantic build after 1.5 h (falls back to raw sheets)
DEADLINE=$((START + TIME_LIMIT - 900))        # no new question in the last 15 min
QA_JOBS=${QA_JOBS:-16}
BUILD_JOBS=${BUILD_JOBS:-8}
export BUILD_MAX_S=1500 BUILD_MAX_TOKENS=12000 BUILD_EXAMPLES=1   # per-business build time cap, thinking cap, few-shot loaders

mkdir -p "$OUT/logs" "$RUNLOG" "$WORK" /tmp/datamate/llm_cache
log() { echo "[predict.sh] $(date -u +%FT%TZ) $*" | tee -a "$RUNLOG/predict_sh.log"; }

APP_PY="python3"
export PYTHONPATH="$CODE/pydeps"               # app deps live apart from vLLM's own environment
export PYTHONUNBUFFERED=1
export LLM_BASE_URL="http://127.0.0.1:$PORT/v1" LLM_API_KEY=EMPTY LLM_MODEL=$SERVED_NAME
export LLM_ENABLE_THINKING=0 LLM_CONTEXT=$MAX_LEN LLM_CACHE_DIR=/tmp/datamate/llm_cache
unset LLM_FALLBACK_MODEL LLM_FALLBACK_BASE_URL LLM_FALLBACK_API_KEY

write_abstain_all() {
  PYTHONPATH= python3 - "$DATA/questions.jsonl" "$OUT/submission.jsonl" <<'EOF'
import json, sys
qs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
with open(sys.argv[2], "w") as f:
    for q in qs:
        f.write(json.dumps({"id": q["id"], "type": "abstain"}) + "\n")
EOF
}

log "start; gpus=$(nvidia-smi -L 2>/dev/null | wc -l) tp=$TP model=$MODEL_DIR"

# vLLM runs with the image's system Python, without the app PYTHONPATH.
env -u PYTHONPATH python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP" --dtype bfloat16 --seed 0 \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching --max-num-seqs 32 \
  --reasoning-parser qwen3 --generation-config vllm \
  --disable-log-requests > "$RUNLOG/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null' EXIT

for _ in $(seq 1 240); do                     # up to 20 min
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  if ! kill -0 $VLLM_PID 2>/dev/null; then break; fi
  sleep 5
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
  log "vLLM failed to start (see $RUNLOG/vllm.log); writing abstain for every question"
  write_abstain_all
  exit 1
fi
log "vLLM ready after $(( $(date +%s) - START ))s"

# A build request holds ~30-40k tokens of KV cache. Builds beyond what fits only wait in vLLM's queue while their
# per-business time cap runs (L4 test: 67k-token cache, 8 builds, most hit the cap), so match jobs to the cache.
KV_TOKENS=$(grep -oE 'GPU KV cache size: [0-9,]+' "$RUNLOG/vllm.log" | head -1 | grep -oE '[0-9,]+$' | tr -d ,)
if [ -n "$KV_TOKENS" ]; then
  FIT=$(( KV_TOKENS / 36000 )); [ "$FIT" -lt 2 ] && FIT=2
  [ "$FIT" -lt "$BUILD_JOBS" ] && BUILD_JOBS=$FIT
fi
log "KV cache ${KV_TOKENS:-?} tokens -> build jobs $BUILD_JOBS"

$APP_PY "$CODE/src/predict.py" --split "$DATA" --work "$WORK" --out "$OUT" \
  --build --build-jobs "$BUILD_JOBS" --build-deadline "$BUILD_DEADLINE" \
  --deadline "$DEADLINE" --jobs "$QA_JOBS" 2>&1 | tee "$RUNLOG/predict.log"
RC=${PIPESTATUS[0]}

if [ ! -s "$OUT/submission.jsonl" ]; then
  log "no submission written; writing abstain for every question"
  write_abstain_all
fi
log "done rc=$RC total=$(( $(date +%s) - START ))s"
exit $RC
