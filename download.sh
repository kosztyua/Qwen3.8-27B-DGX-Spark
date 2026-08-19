#!/usr/bin/env bash
# Pre-download the checkpoints into this dir's HF cache, with retries.
#
#   ./download.sh          RadixArk target + DSpark drafter (what the default
#                          ./start.sh and ./start-sglang.sh serve)
#   ./download.sh --mtp    unsloth checkpoint (for VARIANT=mtp / DSPARK_TARGET=unsloth)
#   ./download.sh --dflash2
#                          RadixArk target + experimental DFlash 2 drafter
#   ./download.sh --all    everything
#
# HF_TOKEN is set (without export) in ~/.bashrc -> pick it up here.
set -u
cd "$(dirname "$0")"

DEFAULT_REPOS=("RadixArk/Qwen3.8-27B-NVFP4" "RadixArk/Qwen3.8-27B-DSpark")
MTP_REPOS=("unsloth/Qwen3.8-27B-NVFP4")
DFLASH2_REPOS=("z-lab/Qwen3.8-27B-DFlash2")
DFLASH2_REVISION="50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
case "${1:-}" in
  "" | --sglang) REPOS=("${DEFAULT_REPOS[@]}") ;;
  --mtp)         REPOS=("${MTP_REPOS[@]}") ;;
  --dflash2)     REPOS=("RadixArk/Qwen3.8-27B-NVFP4" "${DFLASH2_REPOS[@]}") ;;
  --all)         REPOS=("${DEFAULT_REPOS[@]}" "${MTP_REPOS[@]}" "${DFLASH2_REPOS[@]}") ;;
  *) echo "usage: download.sh [--mtp|--dflash2|--all]"; exit 1 ;;
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
    revision_args=()
    if [[ "$repo" == "${DFLASH2_REPOS[0]}" ]]; then
      revision_args=(--revision "${DFLASH2_REVISION}")
    fi
    HF_HOME="$PWD/.cache/huggingface" hf download "$repo" "${revision_args[@]}"
    if [[ $? -eq 0 ]]; then ok=1; break; fi
    echo "retry in 10s"
    sleep 10
  done
  if [[ $ok -ne 1 ]]; then echo "DOWNLOAD-FAILED $repo"; exit 1; fi
done
echo "DOWNLOAD-OK"
