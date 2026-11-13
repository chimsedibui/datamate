#!/usr/bin/env bash
# Build, check and export the submission image, step by step as the organizers' guide asks (sections 9, 10 and 11).
# Needs Docker with BuildKit and the NVIDIA Container Toolkit (sudo or the docker group). Run from anywhere:
#   TEAM=<team> bash docker/build.sh pack      # git lfs pull + scripts/pack_training_code.py -> build/training_code
#   TEAM=<team> bash docker/build.sh build     # image zaic2026_datamate_<team>:final, then platform, size and /code layout
#   TEAM=<team> bash docker/build.sh test [N]  # scoring command (2 GPUs, no network, read-only /data, 16 GiB shm) on the
#                                              # public test -> result_docker/ (N = first N questions, a smoke test only)
#   TEAM=<team> bash docker/build.sh export    # .tar.gz, uncompressed .tar size in bytes, and the .md5 file to send
set -euo pipefail
cd "$(dirname "$0")/.."
TEAM=${TEAM:?set TEAM=<lowercase team id>}
IMAGE=zaic2026_datamate_${TEAM,,}:final
ARCHIVE=zaic2026_datamate_${TEAM,,}.tar.gz
DOCKER=${DOCKER:-docker}
case "${1:-}" in
  pack)
    git lfs pull
    python3 scripts/pack_training_code.py
    ;;
  build)
    for f in models/Qwen3.5-9B/config.json adapters/sql_rft_v3/adapter_model.safetensors adapters/sql_v5/adapter_model.safetensors build/training_code/data_manifest.json; do
      [ -s "$f" ] || { echo "missing $f (base model download, git lfs pull, docker/build.sh pack)" >&2; exit 1; }
    done
    DOCKER_BUILDKIT=1 $DOCKER build --platform linux/amd64 -f docker/Dockerfile -t "$IMAGE" .
    $DOCKER image inspect -f '{{.Os}}/{{.Architecture}} {{.Size}}' "$IMAGE"
    $DOCKER run --rm --network none --entrypoint "" "$IMAGE" find /code -maxdepth 2
    ;;
  test)
    data=${DATA:-$PWD/data/public_test}
    mkdir -p result_docker
    $DOCKER run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
      -v "$(realpath "$data")":/data:ro -v "$PWD/result_docker":/result "$IMAGE" /code/predict.sh ${2:+--limit "$2"}
    .venv/bin/python -m datamate.submission check result_docker/submission.jsonl --questions "$data/questions.jsonl"
    ;;
  export)
    set -o pipefail
    zip=$(command -v pigz >/dev/null && echo "pigz -p 8" || echo gzip)
    $DOCKER save "$IMAGE" | $zip > "$ARCHIVE"
    echo "uncompressed .tar bytes: $($zip -dc "$ARCHIVE" | wc -c)"
    md5sum "$ARCHIVE" > "$ARCHIVE.md5"
    cat "$ARCHIVE.md5"
    ;;
  *) echo "usage: TEAM=<team> bash docker/build.sh pack|build|test [N]|export" >&2; exit 2 ;;
esac
