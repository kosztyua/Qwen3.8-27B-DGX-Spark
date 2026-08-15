#!/usr/bin/env bash
# Pre-download unsloth/Qwen3.8-27B-NVFP4 into this dir's HF cache, with retries.
# With --sglang, also fetch the DSpark drafter used by start-sglang.sh.
# HF_TOKEN is set (without export) in ~/.bashrc -> pick it up here.
set -u
cd "$(dirname "$0")"

REPOS=("unsloth/Qwen3.8-27B-NVFP4")
if [[ "${1:-}" == "--sglang" ]]; then
  # SGLang serves the RadixArk modelopt checkpoint plus the DSpark drafter.
  REPOS+=("RadixArk/Qwen3.8-27B-NVFP4" "RadixArk/Qwen3.8-27B-DSpark")
fi

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
