#!/usr/bin/env bash
# Pre-download unsloth/Qwen3.8-27B-NVFP4 into this dir's HF cache, with retries.
# HF_TOKEN is set (without export) in ~/.bashrc -> pick it up here.
set -u
cd "$(dirname "$0")"
export HF_TOKEN
HF_TOKEN="$(grep -oP '(?<=HF_TOKEN=)["'"'"']?[A-Za-z0-9_\-]+["'"'"']?' ~/.bashrc | head -1 | tr -d '"'"'"'')"
# Plain HTTP downloads (no xet) resume cleanly after stalls.
export HF_HUB_DISABLE_XET=1
for i in $(seq 1 10); do
  echo "attempt $i $(date -Is)"
  HF_HOME="$PWD/.cache/huggingface" hf download unsloth/Qwen3.8-27B-NVFP4
  if [[ $? -eq 0 ]]; then echo "DOWNLOAD-OK"; exit 0; fi
  echo "retry in 10s"
  sleep 10
done
echo "DOWNLOAD-FAILED"
exit 1
