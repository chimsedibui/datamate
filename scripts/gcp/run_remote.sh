#!/usr/bin/env bash
# Start work on the VM from your machine. Long jobs run under nohup, so closing the SSH session does not stop them.
#
#   bash scripts/gcp/run_remote.sh setup                    # weights + docker build + static checks (~30-45 min)
#   bash scripts/gcp/run_remote.sh test smoke train public  # stages of vm_test.sh (see that file)
#   bash scripts/gcp/run_remote.sh status                   # progress: stage log, GPU use, latest predict.sh lines
#   bash scripts/gcp/run_remote.sh export                   # docker save | pigz + md5 (organizer guide §11)
set -euo pipefail
source "$(dirname "$0")/config.sh"
cmd=${1:?setup | test <stages> | status | export}; shift || true
D="cd ~/$REMOTE_DIR && export TEAM=$TEAM"

case $cmd in
setup)
  ssh_vm -- "$D && nohup bash scripts/gcp/vm_setup.sh > ~/setup.log 2>&1 & echo 'setup started; log: ~/setup.log'" ;;
test)
  ssh_vm -- "$D && nohup bash scripts/gcp/vm_test.sh $* > ~/test.log 2>&1 & echo 'test started; log: ~/test.log'" ;;
status)
  ssh_vm -- "tail -n 5 ~/setup.log 2>/dev/null; echo ---; tail -n 8 ~/results/vm_test.log 2>/dev/null; echo ---;
             nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader;
             for f in ~/results/*/logs/_run/predict.log; do [ -f \"\$f\" ] && echo \"\$f:\" && tail -n 2 \"\$f\"; done" ;;
export)
  ssh_vm -- "$D && sudo apt-get install -y -qq pigz >/dev/null; DOCKER='sudo docker' bash scripts/release/export_image.sh ~/export" ;;
*) echo "unknown command $cmd"; exit 2 ;;
esac
