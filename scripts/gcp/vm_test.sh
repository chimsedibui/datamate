#!/usr/bin/env bash
# Runs ON the VM (via run_remote.sh test ...). Every stage uses the official scoring command (check_submission.sh run):
# 2 GPUs, no network, read-only /data, 16 GiB shm, predict.sh without arguments.
#
#   smoke   20 public_test questions: the image runs end to end and writes valid outputs (~15-25 min incl. build)
#   train   150 train questions (= predict.py --per-biz 5, seed 0), scored with answers; compare with q14e (PA 0.447,
#           semantic layers built by gpt-4o) to see the effect of building them with Qwen3-14B (~1.5-2 h)
#   repro   the same 150 questions a second time + diff with the first run (rule: |ΔPA| <= 0.03) (~1.5-2 h)
#   public  all 1000 public_test questions: wall time vs the 6 h limit, output format (~3-5 h on L4)
#
# Usage: bash scripts/gcp/vm_test.sh smoke train repro public      (default: smoke train public)
set -uo pipefail
cd "$(dirname "$0")/../.."
TEAM=${TEAM:-test}
export TEAM DOCKER="sudo docker"
R=$HOME/results
mkdir -p "$R"
stages=${*:-smoke train public}
log() { echo "[vm_test] $(date -u +%FT%TZ) $*" | tee -a "$R/vm_test.log"; }

for s in $stages; do
  log "stage $s start"
  case $s in
  smoke)
    python3 scripts/make_subset.py --split data/public_test --out "$R/data_smoke" --limit 20
    bash scripts/check_submission.sh run "$R/data_smoke" "$R/smoke" 2>&1 | tee "$R/smoke.txt" ;;
  train)
    [ -d "$R/data_train150" ] || python3 scripts/make_subset.py --split data/train --out "$R/data_train150" --per-biz 5
    bash scripts/check_submission.sh run "$R/data_train150" "$R/train1" 2>&1 | tee "$R/train1.txt" ;;
  repro)
    [ -d "$R/data_train150" ] || python3 scripts/make_subset.py --split data/train --out "$R/data_train150" --per-biz 5
    bash scripts/check_submission.sh run "$R/data_train150" "$R/train2" 2>&1 | tee "$R/train2.txt"
    bash scripts/check_submission.sh compare "$R/data_train150" "$R/train1" "$R/train2" 2>&1 | tee "$R/repro.txt" ;;
  public)
    bash scripts/check_submission.sh run data/public_test "$R/public" 2>&1 | tee "$R/public.txt" ;;
  *) log "unknown stage $s" ;;
  esac
  log "stage $s done"
done
log "all stages done: $stages"
