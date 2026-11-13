#!/usr/bin/env bash
# usage: scripts/dev/run_qa.sh <work> <out> <logname> [predict args...]
cd "$(dirname "$0")/../.." && source scripts/dev/dev_env.sh
W=$1 O=$2 N=$3; shift 3
mkdir -p "$O"
exec python src/predict.py --split data/train_dev --work "$W" --out "$O" "$@" > "$O/$N.log" 2>&1
