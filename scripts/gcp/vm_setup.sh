#!/usr/bin/env bash
# Runs ON the VM (via run_remote.sh setup): check GPUs/Docker, download the pinned weights, build the submission
# image, run the static checks. Idempotent: weights and Docker layers are reused on a second run.
set -euo pipefail
cd "$(dirname "$0")/../.."
TEAM=${TEAM:-test}
IMG="zaic2026_datamate_${TEAM}:final"

echo "== waiting for the NVIDIA driver (installed on first boot)"
for _ in $(seq 1 60); do nvidia-smi -L >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

echo "== docker + NVIDIA container toolkit"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
fi
if ! sudo docker info 2>/dev/null | grep -qi nvidia; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
fi
sudo usermod -aG docker "$USER" || true
D="sudo docker"
$D run --rm --gpus all --entrypoint "" vllm/vllm-openai:v0.10.1.1 nvidia-smi -L

echo "== model weights (pinned revision, see scripts/download_model.sh)"
if [ ! -s models/Qwen3-14B.sha256 ]; then
  python3 -m pip install -q --user -U "huggingface_hub[cli]"
  export PATH="$HOME/.local/bin:$PATH"
  bash scripts/download_model.sh
fi
du -sh models/Qwen3-14B

echo "== build $IMG"
$D build --platform linux/amd64 -t "$IMG" .

echo "== static checks"
DOCKER="$D" TEAM="$TEAM" bash scripts/check_submission.sh static
