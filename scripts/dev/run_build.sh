#!/usr/bin/env bash
# usage: scripts/dev/run_build.sh <work> <logfile> [build_semantic args...]
cd "$(dirname "$0")/../.." && source scripts/dev/dev_env.sh
W=$1 L=$2; shift 2
exec python src/build_semantic.py --split data/train_dev --work "$W" "$@" > "$L" 2>&1
