# Test the submission image on a GCP VM (2 × L4, us-central1)

The local machine has one 8 GB GPU, so the image cannot run here. These scripts create a `g2-standard-24` VM
(2 × NVIDIA L4, 48 GB VRAM), build the submission image on it, and run it with the organizers' scoring command.

Settings are in `config.sh`: project `obk-sb-ai-analytics`, zones us-central1-a/b/c, team tag `test`. Override any of
them from the environment, e.g. `TEAM=myteam SPOT=1 bash scripts/gcp/create_vm.sh`.

## Steps

| # | Command (from the repo root) | What it does | Time |
|---|---|---|---|
| 1 | `bash scripts/gcp/create_vm.sh` | creates the VM (**billing starts**); auto power-off after `MAX_HOURS` (12) | 2 min |
| 2 | `bash scripts/gcp/sync_to_vm.sh` | sends code + data/train + data/public_test (no secrets, no work/out) | 1 min |
| 3 | `bash scripts/gcp/run_remote.sh setup` | driver/Docker checks, pinned weights download, `docker build`, static checks | 30–45 min |
| 4 | `bash scripts/gcp/run_remote.sh test smoke` | 20 public questions end to end, output format | 15–25 min |
| 5 | `bash scripts/gcp/run_remote.sh test train repro public` | 150 train questions scored, the same again for reproducibility, all 1000 public questions | 6–9 h |
| – | `bash scripts/gcp/run_remote.sh status` | progress (stage log, GPU use, predict.sh lines) | – |
| 6 | `bash scripts/gcp/fetch_results.sh` | copies results to `out/gcp_<timestamp>/` | 1 min |
| 7 | `TEAM=<team> bash scripts/gcp/run_remote.sh export` (optional) | `docker save` → .tar.gz + .md5 on the VM | 30–60 min |
| 8 | `bash scripts/gcp/delete_vm.sh` | deletes the VM and disk (**billing stops**) | 1 min |

Stages and what to read in the results (`vm_test.sh`):
- `smoke`: `validate_output.py` says OK; `predict_sh.log` shows vLLM ready.
- `train`: PA versus q14e (0.447 with semantic layers built by gpt-4o). This run builds the layers with Qwen3-14B,
  as the organizers' run will. A large drop means the build prompts need work before submission.
- `repro`: `reproducibility: N/150 answers differ` and the two PA values (rule: |ΔPA| ≤ 0.03).
- `public`: wall time for 1000 questions (the private test has ~500 on 2 × RTX 5090, which are faster than L4).

## Cost (price list, USD)

| Item | Rate | Full plan (≈ 8–10 h) |
|---|---|---|
| g2-standard-24 on-demand | $2.00/h | $16–20 |
| g2-standard-24 Spot (`SPOT=1`) | $1.20/h | $10–12, but a preemption kills a running stage |
| 300 GB pd-balanced disk | ~$1/day | <$1 |

Stopping the VM (or the auto power-off) stops the GPU charges; only `delete_vm.sh` stops the disk charge.
