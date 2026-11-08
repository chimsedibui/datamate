# Shared settings for scripts/gcp/*.sh (sourced, not executed). Override any value from the environment.
PROJECT=${PROJECT:-obk-sb-ai-analytics}
REGION=${REGION:-us-central1}
ZONES=${ZONES:-"us-central1-a us-central1-b us-central1-c"}   # L4 zones in us-central1, tried in order (stockouts)
VM=${VM:-obk-test}
MACHINE=${MACHINE:-g2-standard-24}                              # 24 vCPU, 96 GB RAM, 2 x NVIDIA L4 (48 GB VRAM)
IMAGE_FAMILY=${IMAGE_FAMILY:-ubuntu-accelerator-2204-amd64-with-nvidia-580}  # signed NVIDIA 580 driver: org policy requires Secure Boot
IMAGE_PROJECT=${IMAGE_PROJECT:-ubuntu-os-accelerator-images}
DISK_GB=${DISK_GB:-300}                                         # image ~60 GB + build cache + weights + export
DISK_TYPE=${DISK_TYPE:-pd-ssd}                                  # docker build writes ~90 GB; pd-balanced took 18 min to export one layer
SPOT=${SPOT:-0}                                                 # 1 = Spot VM (~40% cheaper, may be preempted)
MAX_HOURS=${MAX_HOURS:-12}                                      # VM powers itself off after this many hours (cost cap)
TEAM=${TEAM:-test}                                              # image tag zaic2026_datamate_<TEAM>:final
REMOTE_DIR=${REMOTE_DIR:-datamate}                              # project folder in the VM user's home

# Price list (USD/h, Cloud Billing catalog, us-central1, 2026-11): on-demand ~2.00, Spot ~1.20; disk ~$1/day.
zone_file="$(dirname "${BASH_SOURCE[0]}")/.zone"
vm_zone() { cat "$zone_file" 2>/dev/null || echo "${ZONES%% *}"; }
# Firewall allows SSH only from IAP (35.235.240.0/20); set IAP=0 to connect to the external IP directly.
iap_flag() { [ "${IAP:-1}" = "1" ] && echo --tunnel-through-iap; }
ssh_vm() { gcloud compute ssh "$VM" --project "$PROJECT" --zone "$(vm_zone)" $(iap_flag) "$@"; }
