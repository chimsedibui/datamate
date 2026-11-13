#!/usr/bin/env bash
# Entry point of the image: bash /code/predict.sh, no arguments (organizers' Docker guide, section 5). All settings are fixed in
# /code/flags.sh. Reads /data (read-only); writes into /result:
#   submission.jsonl  every question id; rewritten after each shop of pass 1 by every process, so answers saved before a stop
#                     still count; replaced at the end by the pooled vote of both passes
#   logs/<id>.json    one log record per question on one line: id, answer, elapsed_ms, thinking, code, output (+ the full trace);
#                     after pooling, answer = the pooled answer and "ensemble" holds each pass's own answer and the vote
#   work/shard<i>/    pass 1 per-GPU files: the shop layers (views, debt macros) the SQL reads, trace.jsonl, process log
#   work2/shard<i>/   pass 2 per-GPU files (same layers, read from work/)
# Two passes of the same pipeline, one LoRA adapter each (one base model, 9.74B parameters with both adapters):
#   pass 1  LoRA v3, drafts every shop's layers and answers every question (the submission is complete after it)
#   pass 2  LoRA v5 on pass 1's layers, as long as time is left (a shop it does not reach keeps pass 1's answer)
#   pool    every sample of both passes votes once (analysis/pool_runs.py): where the adapters disagree the answer is abstained
# One datamate.predict process per GPU and pass (data parallel over shops, no tensor parallelism). Everything stops at
# TIMEOUT_SECONDS (5 h 50 min), below the 6 h limit that includes model loading.
# Outside the image (a local check of the same flow), every path can be overridden: CODE_DIR, FLAGS_FILE, MODEL_DIR, LORA_DIR, LORA2_DIR,
# ASSETS_DIR, POOL_SCRIPT, PYTHON, DATA_DIR, RESULT_DIR, NUM_GPUS, TIMEOUT_SECONDS (see docker/README.md).
set -uo pipefail
cd "${CODE_DIR:-/code}"
source "${FLAGS_FILE:-./flags.sh}"
PY="${PYTHON:-python3}"
MODEL="${MODEL_DIR:-/code/models/Qwen3.5-9B}" LORA1="${LORA_DIR:-/code/models/lora}" LORA2="${LORA2_DIR:-/code/models/lora_v5}"
ASSETS="${ASSETS_DIR:-/code/assets}" POOL="${POOL_SCRIPT:-/code/src/analysis/pool_runs.py}"
DATA="${DATA_DIR:-/data}" OUT="${RESULT_DIR:-/result}"
NG="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
T="${TIMEOUT_SECONDS:-21000}"
mkdir -p "$OUT/logs" "$OUT/work" "$OUT/work2"
start=$(date +%s)
echo "datamate: $NG GPU(s), $(grep -c . "$DATA/questions.jsonl") questions, timeout ${T}s"
for i in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$i timeout --signal=TERM --kill-after=15s "${T}s" \
    "$PY" -m datamate.predict --data "$DATA" --output "$OUT/work/shard$i" --shard "$i/$NG" --deadline "$(( T - 900 ))" \
    --log-dir "$OUT/logs" --merge-into "$OUT" \
    --model "$MODEL" --lora "$LORA1" --fewshot "$ASSETS/fewshot_bank.jsonl" \
    "${DATAMATE_FLAGS[@]}" --engine "$DATAMATE_ENGINE" "$@" > "$OUT/work/shard$i.log" 2>&1 &
done
wait
"$PY" -m datamate.submission merge "$OUT" --shards "$OUT/work" --no-copy-logs --questions "$DATA/questions.jsonl"
"$PY" -m datamate.submission fill-logs "$OUT/logs" --questions "$DATA/questions.jsonl"
"$PY" -m datamate.submission check "$OUT/submission.jsonl" --questions "$DATA/questions.jsonl" || true
grep -h -E "model ready|loaded [0-9]+ shops|view drafts|debt drafts" "$OUT"/work/shard*.log || true
p1=$(( $(date +%s) - start )); left=$(( T - p1 ))
echo "datamate: pass 1 done in ${p1}s, ${left}s left"
if [ "$left" -gt 1800 ] && [ -f "$LORA2/adapter_model.safetensors" ]; then
  for i in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=$i timeout --signal=TERM --kill-after=15s "$(( left - 600 ))s" \
      "$PY" -m datamate.predict --data "$DATA" --output "$OUT/work2/shard$i" --shard "$i/$NG" --deadline "$(( left - 1200 ))" \
      --log-dir "$OUT/work2/logs" --views-from "$OUT/work/shard$i" \
      --model "$MODEL" --lora "$LORA2" --fewshot "$ASSETS/fewshot_bank_v2.jsonl" --rule-lines \
      "${DATAMATE_FLAGS[@]}" --engine "$DATAMATE_ENGINE" "$@" > "$OUT/work2/shard$i.log" 2>&1 &
  done
  wait
  echo "datamate: pass 2 done at $(( $(date +%s) - start ))s"
  if ls "$OUT"/work2/shard*/trace.jsonl >/dev/null 2>&1; then
    if "$PY" "$POOL" --member "$OUT/work" --member "$OUT/work2" --data "$DATA" --layers "$OUT/work" \
         --out "$OUT/work/pool" --logs "$OUT/logs" \
       && "$PY" -m datamate.submission check "$OUT/work/pool/submission.jsonl" --questions "$DATA/questions.jsonl"; then
      cp "$OUT/work/pool/submission.jsonl" "$OUT/.submission.pool.tmp" && mv "$OUT/.submission.pool.tmp" "$OUT/submission.jsonl"
      echo "datamate: submission = pooled vote of both passes"
    else
      echo "datamate: pooling failed, the submission stays pass 1's"
    fi
  fi
else
  echo "datamate: no pass 2 (time left ${left}s)"
fi
echo "datamate: done in $(( $(date +%s) - start ))s, $(ls "$OUT/logs" | wc -l) question logs"
