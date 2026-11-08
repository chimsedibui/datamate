#!/usr/bin/env bash
# Copy the project (git-tracked + untracked-but-not-ignored files) and data/train, data/public_test to the VM.
# Secrets (.env*), work/, out/, models/ are never sent. Usage: bash scripts/gcp/sync_to_vm.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$(git rev-parse --show-toplevel)"

list=$(mktemp)
{ git ls-files; git ls-files --others --exclude-standard; } | grep -vE '^\.env' | sort -u > "$list"
find data/train data/public_test -type f >> "$list"
echo "sending $(wc -l < "$list") files ($(tar -cf - -T "$list" | wc -c | numfmt --to=iec))"
tar -czf - -T "$list" | ssh_vm -- "mkdir -p ~/$REMOTE_DIR && tar -xzf - -C ~/$REMOTE_DIR"
rm -f "$list"
echo "synced to $VM:~/$REMOTE_DIR"
