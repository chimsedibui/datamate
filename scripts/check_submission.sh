#!/usr/bin/env bash
# Pre-submission checks (organizer guide section 10) for the built image.
#
# Usage:
#   TEAM=<team> bash scripts/check_submission.sh static                          # platform, size, layout, offline imports
#   TEAM=<team> bash scripts/check_submission.sh run <data_dir> <result_dir>     # official scoring command + output check
#   TEAM=<team> bash scripts/check_submission.sh compare <data_dir> <run1> <run2> # reproducibility of two runs
#
# `run` needs 2 GPUs (device 0,1) and executes predict.sh exactly as the organizers do.
set -euo pipefail
DOCKER=${DOCKER:-docker}   # e.g. DOCKER="sudo docker"
TEAM=${TEAM:?set TEAM to the lowercase team identifier}
IMG="zaic2026_datamate_${TEAM}:final"
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-static}

case "$MODE" in
static)
  echo "== platform / size"
  $DOCKER image inspect -f '{{.Os}}/{{.Architecture}} {{.Size}} bytes' "$IMG"
  echo "== layout"
  $DOCKER run --rm --network none --entrypoint "" "$IMG" \
    bash -c 'cd /code && find . -maxdepth 2 -not -path "./models/*/*" -not -path "./pydeps/*" | sort; du -sh /code/models/*'
  echo "== offline imports and model files"
  $DOCKER run --rm --network none --entrypoint "" "$IMG" bash -c '
    set -e
    test -x /code/predict.sh
    test -f /code/models/Qwen3-14B/config.json
    test -s /code/models/Qwen3-14B.sha256
    (cd /code/models/Qwen3-14B && sha256sum --quiet -c ../Qwen3-14B.sha256) && echo "weights checksum OK"
    PYTHONPATH=/code/pydeps python3 -c "import sys; sys.path.insert(0, \"/code/src\"); import datamate.agent, datamate.mapper, pandas, openpyxl, pyarrow; print(\"app imports OK, pandas\", pandas.__version__)"
    python3 -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__)"
    grep -q "^set -uo pipefail" /code/predict.sh && echo "predict.sh present"'
  ;;
run)
  DATA=$(realpath "${2:?data dir}"); RES=$(realpath -m "${3:?result dir}")
  mkdir -p "$RES"
  START=$(date +%s)
  $DOCKER run --rm --gpus '"device=0,1"' --network none --shm-size=16g \
    --entrypoint bash \
    -v "$DATA":/data:ro \
    -v "$RES":/result \
    "$IMG" /code/predict.sh || echo "predict.sh exited with $?"
  echo "wall time: $(( $(date +%s) - START ))s for $(wc -l < "$DATA/questions.jsonl") questions"
  python3 "$HERE/validate_output.py" --questions "$DATA/questions.jsonl" --result "$RES"
  if [ -f "$DATA/answers.jsonl" ]; then
    $DOCKER run --rm --network none --entrypoint bash -v "$DATA":/data:ro -v "$RES":/result "$IMG" \
      -c "PYTHONPATH=/code/pydeps python3 /code/src/evaluate.py --pred /result/submission.jsonl --split /data --all" \
      | head -1
  fi
  ;;
compare)
  DATA=$(realpath "${2:?data dir}")
  python3 "$HERE/validate_output.py" --questions "$DATA/questions.jsonl" --result "${3:?run1}" --compare "${4:?run2}"
  ;;
*)
  echo "unknown mode $MODE (static | run | compare)"; exit 2 ;;
esac
