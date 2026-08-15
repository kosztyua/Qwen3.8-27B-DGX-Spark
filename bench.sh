#!/usr/bin/env bash
# Benchmark the running server with `vllm bench serve` (runs inside the
# serving container, results land in .cache/huggingface/bench-results/<label>).
#
#   ./bench.sh <label> [--scenarios dec1,dec4,pre16k,long128k]
#
# Scenarios (random dataset, thinking-mode sampling, fixed seeds so prompts
# are identical across runs; restart the server between configs so the prefix
# cache is cold):
#   dec1     c=1, 1k-token prompts, 1k-token generations  (interactive decode)
#   dec4     c=4, same shape                              (batched decode)
#   pre16k   c=4, 16k-token prompts, 16-token generations (prefill/TTFT)
#   long128k c=1, 128k-token prompts, 256-token generations (long context)
#
# MTP acceptance is content-dependent: compare configs on ITL (step time) and
# the reported acceptance length separately, not on raw tok/s alone.
set -uo pipefail

LABEL="${1:?usage: bench.sh <label> [--scenarios dec1,dec4,...]}"
shift
SCENARIOS="dec1,dec4,pre16k"
CONTAINER="qwen3.8-27b-nvfp4"
PORT="8888"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) SCENARIOS="dec1,dec4,pre16k,long128k"; shift ;;
    --scenarios) SCENARIOS="$2"; shift 2 ;;
    --container) CONTAINER="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

want() { [[ ",${SCENARIOS}," == *",$1,"* ]]; }

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST_RESULTS="${WORK_DIR}/.cache/huggingface/bench-results/${LABEL}"
CTR_RESULTS="/root/.cache/huggingface/bench-results/${LABEL}"
mkdir -p "${HOST_RESULTS}"

run_scenario() {
  local name="$1" seed="$2"; shift 2
  echo "=== scenario ${name} (seed ${seed}) $(date -Is) ==="
  docker exec "${CONTAINER}" vllm bench serve \
    --base-url "http://127.0.0.1:${PORT}" \
    --model qwen38-27b-unsloth-nvfp4 \
    --tokenizer unsloth/Qwen3.8-27B-NVFP4 \
    --trust-remote-code \
    --dataset-name random \
    --temperature 1.0 --top-p 0.95 --top-k 20 \
    --ignore-eos \
    --seed "${seed}" \
    --num-warmups 1 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result --result-dir "${CTR_RESULTS}" --result-filename "${name}.json" \
    --label "${LABEL}-${name}" \
    "$@" 2>&1 | tee "${HOST_RESULTS}/${name}.log" | tail -50
}

echo "### bench label=${LABEL} container=${CONTAINER} scenarios=${SCENARIOS}"

want dec1 && run_scenario dec1 11230 \
  --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 4 --max-concurrency 1

want dec4 && run_scenario dec4 11450 \
  --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 12 --max-concurrency 4

want pre16k && run_scenario pre16k 11677 \
  --random-input-len 16384 --random-output-len 16 \
  --num-prompts 8 --max-concurrency 4

want long128k && run_scenario long128k 11802 \
  --random-input-len 131072 --random-output-len 256 \
  --num-prompts 2 --max-concurrency 1

echo "### done; results in ${HOST_RESULTS}"
