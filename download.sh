#!/usr/bin/env bash
# Pre-download the checkpoints into this dir's HF cache, with retries.
#
#   ./download.sh          RadixArk target + DSpark drafter (what the default
#                          ./start.sh and ./start-sglang.sh serve)
#   ./download.sh --mtp    unsloth checkpoint (for VARIANT=mtp / DSPARK_TARGET=unsloth)
#   ./download.sh --all    everything
#
# HF_TOKEN is set (without export) in ~/.bashrc -> pick it up here.
set -u
cd "$(dirname "$0")"

DEFAULT_REPOS=("RadixArk/Qwen3.8-27B-NVFP4" "RadixArk/Qwen3.8-27B-DSpark")
MTP_REPOS=("unsloth/Qwen3.8-27B-NVFP4")
case "${1:-}" in
  "" | --sglang) REPOS=("${DEFAULT_REPOS[@]}") ;;
  --mtp)         REPOS=("${MTP_REPOS[@]}") ;;
  --all)         REPOS=("${DEFAULT_REPOS[@]}" "${MTP_REPOS[@]}") ;;
  *) echo "usage: download.sh [--mtp|--all]"; exit 1 ;;
esac

# Honor an HF_TOKEN already in the environment; fall back to ~/.bashrc
# (mirrors the start scripts).
if [[ -z "${HF_TOKEN:-}" && -f ~/.bashrc ]]; then
  HF_TOKEN="$(grep -oP '(?<=HF_TOKEN=)["'"'"']?[A-Za-z0-9_\-]+["'"'"']?' ~/.bashrc | head -1 | tr -d '"'"'"'')"
fi
export HF_TOKEN
# Plain HTTP downloads (no xet) resume cleanly after stalls.
export HF_HUB_DISABLE_XET=1

for repo in "${REPOS[@]}"; do
  ok=0
  for i in $(seq 1 10); do
    echo "[$repo] attempt $i $(date -Is)"
    HF_HOME="$PWD/.cache/huggingface" hf download "$repo"
    if [[ $? -eq 0 ]]; then ok=1; break; fi
    echo "retry in 10s"
    sleep 10
  done
  if [[ $ok -ne 1 ]]; then echo "DOWNLOAD-FAILED $repo"; exit 1; fi
done
echo "DOWNLOAD-OK"
