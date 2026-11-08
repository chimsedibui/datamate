#!/usr/bin/env bash
# Export the final image for upload (organizer guide section 11): .tar.gz, uncompressed size, md5 file.
# Usage: TEAM=<lowercase_team> bash scripts/export_image.sh [out_dir]
set -euo pipefail
DOCKER=${DOCKER:-docker}   # e.g. DOCKER="sudo docker"
TEAM=${TEAM:?set TEAM to the lowercase team identifier}
IMG="zaic2026_datamate_${TEAM}:final"
OUT=${1:-.}
NAME="zaic2026_datamate_${TEAM}.tar.gz"
mkdir -p "$OUT"
cd "$OUT"

$DOCKER image inspect -f '{{.Os}}/{{.Architecture}}' "$IMG" | grep -qx 'linux/amd64' \
  || { echo "image $IMG is not linux/amd64"; exit 1; }

ZIP=gzip; UNZIP="gzip -dc"
if command -v pigz >/dev/null; then ZIP="pigz -p 8"; UNZIP="pigz -dc"; fi

set -o pipefail   # without it a failed docker save still leaves a small archive and "succeeds"
$DOCKER save "$IMG" | $ZIP > "$NAME"
SIZE=$($UNZIP "$NAME" | wc -c)
md5sum "$NAME" > "$NAME.md5"

echo "archive:            $OUT/$NAME ($(du -h "$NAME" | cut -f1))"
echo "uncompressed bytes: $SIZE  (limit 100 GB = 100000000000)"
echo "md5:                $(cat "$NAME.md5")"
[ "$SIZE" -le 100000000000 ] || { echo "WARNING: uncompressed tar exceeds 100 GB"; exit 1; }
echo "Send the link, $NAME.md5 and the uncompressed size. Do not rename the archive."
