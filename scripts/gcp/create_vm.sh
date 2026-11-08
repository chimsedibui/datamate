#!/usr/bin/env bash
# Create the 2 x L4 test VM (billing starts now: ~$2.00/h on-demand, ~$1.20/h Spot).
# Usage: bash scripts/gcp/create_vm.sh          (SPOT=1 for a Spot VM, MAX_HOURS=N for the auto power-off)
set -euo pipefail
source "$(dirname "$0")/config.sh"

provisioning=(--provisioning-model=STANDARD)
[ "$SPOT" = "1" ] && provisioning=(--provisioning-model=SPOT --instance-termination-action=STOP)

# Safety net: power off after MAX_HOURS even if nobody deletes the VM (a stopped VM bills only its disk).
startup="#!/bin/bash
shutdown -h +$((MAX_HOURS * 60)) 'datamate cost cap reached' || true"

for zone in $ZONES; do
  echo "trying $zone ..."
  if gcloud compute instances create "$VM" --project "$PROJECT" --zone "$zone" \
      --machine-type "$MACHINE" \
      --accelerator "type=nvidia-l4,count=2" \
      --maintenance-policy TERMINATE "${provisioning[@]}" \
      --image-family "$IMAGE_FAMILY" --image-project "$IMAGE_PROJECT" \
      --boot-disk-size "${DISK_GB}GB" --boot-disk-type "$DISK_TYPE" \
      --metadata "install-nvidia-driver=True,startup-script=$startup" \
      --labels "purpose=datamate-test,owner=khanh-tran" \
      --scopes cloud-platform; then
    echo "$zone" > "$zone_file"
    echo "created $VM in $zone; first boot installs the NVIDIA driver (a few minutes)."
    echo "next: bash scripts/gcp/sync_to_vm.sh && bash scripts/gcp/run_remote.sh setup"
    exit 0
  fi
done
echo "could not create $VM in any of: $ZONES (L4 stockout or quota)"; exit 1
