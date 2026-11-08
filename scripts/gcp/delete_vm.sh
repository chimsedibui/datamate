#!/usr/bin/env bash
# Delete the test VM and its boot disk (stops all billing for it). Fetch results first.
# Usage: bash scripts/gcp/delete_vm.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
gcloud compute instances delete "$VM" --project "$PROJECT" --zone "$(vm_zone)" --delete-disks=all
rm -f "$zone_file"
echo "deleted $VM"
