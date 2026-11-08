#!/usr/bin/env bash
# Copy ~/results (submissions, logs, eval output, stage reports) from the VM to out/gcp_<timestamp>/.
# Usage: bash scripts/gcp/fetch_results.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$(git rev-parse --show-toplevel)"
dest="out/gcp_$(date +%Y%m%d_%H%M)"
mkdir -p "$dest"
ssh_vm -- "sudo chown -R \$USER ~/results && tar -czf - -C ~/results --exclude='data_*' ." | tar -xzf - -C "$dest"
echo "results in $dest"
ls "$dest"
