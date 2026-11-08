#!/usr/bin/env bash
# Download the inference model into models/ before `docker build` (the image itself never downloads anything).
# Usage: bash scripts/download_model.sh            # needs: pip install -U "huggingface_hub[cli]"
set -euo pipefail
REPO=${REPO:-Qwen/Qwen3-14B}
REV=${REV:-40c069824f4251a91eefaf281ebe4c544efd3e18}   # pinned Qwen/Qwen3-14B commit (2025-07-26)
DEST=${DEST:-models/Qwen3-14B}

hf download "$REPO" --revision "$REV" --local-dir "$DEST" \
  --exclude "*.pth" "original/*"
( cd "$DEST" && find . -type f ! -path "./.cache/*" -print0 | sort -z | xargs -0 sha256sum ) > "$DEST.sha256"
echo "downloaded $REPO@$REV -> $DEST; checksums in $DEST.sha256"
