#!/usr/bin/env bash
# Download the inference model into models/ before `docker build` (the image itself never downloads anything).
# Usage: bash scripts/release/download_model.sh            # needs: pip install -U huggingface_hub
set -euo pipefail
REPO=${REPO:-Qwen/Qwen3-14B}
REV=${REV:-40c069824f4251a91eefaf281ebe4c544efd3e18}   # pinned Qwen/Qwen3-14B commit (2025-07-26)
DEST=${DEST:-models/Qwen3-14B}

# Python API instead of the `hf` CLI: its --exclude parsing differs between huggingface_hub versions.
python3 -c "import sys; from huggingface_hub import snapshot_download as s; \
s(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], ignore_patterns=['*.pth', 'original/*'])" \
  "$REPO" "$REV" "$DEST"
( cd "$DEST" && find . -type f ! -path "./.cache/*" -print0 | sort -z | xargs -0 sha256sum ) > "$DEST.sha256"
echo "downloaded $REPO@$REV -> $DEST; checksums in $DEST.sha256"
