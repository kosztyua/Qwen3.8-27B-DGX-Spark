#!/usr/bin/env bash
# Benchmark the running server (vLLM or SGLang — anything OpenAI-compatible on
# the port) with `vllm bench serve`. The bench client runs in its own
# throwaway container from the pinned vLLM image, so it works no matter which
# engine is serving. Results land in .cache/huggingface/bench-results/<label>/.
#
#   ./bench.sh <label> [--full] [--scenarios dec1,dec4,pre16k,long128k]
#                      [--port 8888] [--model NAME]
# The served model name is auto-detected from /v1/models unless --model is
# given (the two engines serve different names).
#
# Scenarios (random dataset, thinking-mode sampling, fixed seeds so prompts
# are identical across runs; restart the server between configs so the prefix
# cache is cold):
#   dec1     c=1, 1k-token prompts, 1k-token generations  (interactive decode)
#   dec4     c=4, same shape                              (batched decode)
#   pre16k   c=4, 16k-token prompts, 16-token generations (prefill/TTFT)
#   long128k c=1, 128k-token prompts, 256-token generations (long context;
#            needs CONTEXT_1M=1 or any config with max-model-len > 128k)
#
# Session scenarios (sess8/sess12/sess16/sess20) use the prefix_repetition
# dataset instead of random, because the random scenarios above measure the
# wrong regime for this box's production traffic. Measured production shape:
# ~32k median context, 89.6% prefix-cache hit rate, ~850-token generations,
# arriving as multi-turn agentic sessions rather than independent prompts.
# Each scenario gives every concurrent slot its own 28k prefix and runs four
# 4k-suffix requests against it, so intra-session prefix reuse and per-session
# KV residency both behave like the real thing. Vary only the concurrency
# across sess8..sess20 to get the batching curve; the server needs
# --max-num-seqs >= c AND a --cudagraph-capture-sizes ladder covering c*(1+k),
# or the wider batches silently fall back to eager attention.
#
# Absolute tok/s from these is still pessimistic vs real content (synthetic
# tokens are less predictable, so speculative acceptance runs lower). Compare
# configs against each other, not against the README's real-content numbers.
#
# Reading results: speculative-decode acceptance is content-dependent, so
# compare configs on ITL (step time) and the reported acceptance length
# separately, not on raw tok/s alone. The "Speculative Decoding" block is
# only printed when a vLLM server is being benchmarked.
set -uo pipefail

# Keep in sync with IMAGE in start.sh (the client just needs `vllm bench`).
CLIENT_IMAGE="vllm/vllm-openai@sha256:b5c860acda75d737a8e58cc99ba86ff13982695dceae194f906c2d7b54979358"

LABEL="${1:?usage: bench.sh <label> [--full] [--scenarios dec1,dec4,...] [--port N] [--model NAME]}"
shift
SCENARIOS="dec1,dec4,pre16k"
PORT="8888"
MODEL=""   # default: auto-detect from the server's /v1/models
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) SCENARIOS="dec1,dec4,pre16k,long128k"; shift ;;
    --scenarios) SCENARIOS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

want() { [[ ",${SCENARIOS}," == *",$1,"* ]]; }

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
HOST_RESULTS="${HF_HOME}/bench-results/${LABEL}"
CTR_RESULTS="/root/.cache/huggingface/bench-results/${LABEL}"
mkdir -p "${HOST_RESULTS}"

if ! curl -fsS -m 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "No server responding on http://127.0.0.1:${PORT} — start one with ./start.sh or ./start-sglang.sh"
  exit 1
fi

# Variants serve different model names; ask the server which it is.
if [[ -z "${MODEL}" ]]; then
  MODEL=$(curl -fsS -m 5 "http://127.0.0.1:${PORT}/v1/models" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
  [[ -n "${MODEL}" ]] || { echo "Could not detect the served model name; pass --model"; exit 1; }
fi
case "${MODEL}" in
  *radixark*) TOKENIZER="RadixArk/Qwen3.8-27B-NVFP4" ;;
  *)          TOKENIZER="unsloth/Qwen3.8-27B-NVFP4" ;;
esac

failures=0
run_scenario() {
  local name="$1" seed="$2"; shift 2
  echo "=== scenario ${name} (seed ${seed}) $(date -Is) ==="
  docker run --rm --network host \
    -e HF_HOME=/root/.cache/huggingface \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    --entrypoint vllm "${CLIENT_IMAGE}" bench serve \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "${MODEL}" \
    --tokenizer "${TOKENIZER}" \
    --trust-remote-code \
    --dataset-name "${DATASET:-random}" \
    --temperature 1.0 --top-p 0.95 --top-k 20 \
    --ignore-eos \
    --seed "${seed}" \
    --num-warmups 1 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result --result-dir "${CTR_RESULTS}" --result-filename "${name}.json" \
    --label "${LABEL}-${name}" \
    "$@" 2>&1 | tee "${HOST_RESULTS}/${name}.log" | tail -50
  if [[ "${PIPESTATUS[0]}" != "0" || ! -s "${HOST_RESULTS}/${name}.json" ]]; then
    echo "!!! scenario ${name} FAILED (see ${HOST_RESULTS}/${name}.log)"
    failures=$((failures + 1))
  fi
}

echo "### bench label=${LABEL} port=${PORT} model=${MODEL} scenarios=${SCENARIOS}"

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

# One 28k prefix per concurrent slot, four 4k-suffix requests against each.
# Total input 32,768 tokens; 1k generations. Seeds differ per concurrency so a
# rerun of the same scenario is reproducible, but note the prompt set is not
# identical across concurrencies by construction (num_prefixes == c).
sess() {
  local c="$1" seed="$2"
  DATASET=prefix_repetition run_scenario "sess${c}" "${seed}" \
    --prefix-repetition-prefix-len 28672 \
    --prefix-repetition-suffix-len 4096 \
    --prefix-repetition-output-len 1024 \
    --prefix-repetition-num-prefixes "${c}" \
    --num-prompts "$((c * 4))" --max-concurrency "${c}"
}

want sess8  && sess 8  11908
want sess12 && sess 12 11912
want sess16 && sess 16 11916
want sess20 && sess 20 11920

if [[ "${failures}" != "0" ]]; then
  echo "### done with ${failures} FAILED scenario(s); results in ${HOST_RESULTS}"
  exit 1
fi
echo "### done; results in ${HOST_RESULTS}"
