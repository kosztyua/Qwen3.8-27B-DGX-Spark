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
# The scenario approximates that but cannot match the hit rate: a 28,672-token
# prefix of a 32,768-token request caps reuse at 87.5%, and the achievable rate
# is 80.2% at SESS_REQS=12 (65.6% at 4), since the first turn of each session is
# always cold.
# Each scenario creates one 28k prefix per concurrent slot and runs SESS_REQS
# (default 12) 4k-suffix requests against each, so intra-session prefix reuse
# and per-session KV residency both behave like the real thing. Note the
# dataset shuffles all requests before dispatch, so turns of a given prefix do
# not run strictly in order or one-at-a-time -- reuse and residency hold, but
# it is not a literal one-session-per-slot simulation. Vary only the concurrency
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
# Pinned, not read from the environment: an inherited DATASET would silently
# redirect the four random scenarios (whose --random-* flags would then be
# ignored) while still writing results under the names dec1/dec4/pre16k.
# sess() shadows this per-call via a variable-assignment prefix.
DATASET=random
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) SCENARIOS="dec1,dec4,pre16k,long128k"; shift ;;
    --scenarios) SCENARIOS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

KNOWN_SCENARIOS=",dec1,dec4,pre16k,long128k,sess8,sess12,sess16,sess20,"
IFS=',' read -ra _sc <<<"${SCENARIOS}"
(( ${#_sc[@]} )) || { echo "no scenarios selected (known:${KNOWN_SCENARIOS//,/ })"; exit 1; }
for _s in "${_sc[@]}"; do
  [[ -n "${_s}" ]] || { echo "empty scenario name in --scenarios '${SCENARIOS}'"; exit 1; }
  [[ "${KNOWN_SCENARIOS}" == *",${_s},"* ]] \
    || { echo "unknown scenario '${_s}' (known:${KNOWN_SCENARIOS//,/ })"; exit 1; }
done
unset _sc _s

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
    --dataset-name "${DATASET}" \
    --temperature 1.0 --top-p 0.95 --top-k 20 \
    --ignore-eos \
    --seed "${seed}" \
    --num-warmups 1 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result --result-dir "${CTR_RESULTS}" --result-filename "${name}.json" \
    --label "${LABEL}-${name}" \
    "$@" 2>&1 | tee "${HOST_RESULTS}/${name}.log" | tail -50
  # `vllm bench serve` exits 0 and still writes a complete JSON when every
  # request failed, so the exit code alone is not a success signal.
  if [[ "${PIPESTATUS[0]}" != "0" || ! -s "${HOST_RESULTS}/${name}.json" ]] \
     || ! python3 -c 'import json,sys
d = json.load(open(sys.argv[1]))
if d.get("failed"):
    sys.stderr.write("completed=%s failed=%s\n" % (d.get("completed"), d.get("failed")))
    sys.exit(1)' "${HOST_RESULTS}/${name}.json"; then
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

# One 28k prefix per concurrent slot, SESS_REQS 4k-suffix requests against each.
# Total input 32,768 tokens per request; 1k generations. Seeds differ per concurrency so a
# rerun of the same scenario is reproducible, but note the prompt set is not
# identical across concurrencies by construction (num_prefixes == c).
# SESS_REQS controls turns per session. It sets how much of the run is prefill,
# which is the single biggest lever on whether a concurrency comparison is fair:
# every session pays one full 32k prefill on its first turn and only 4k per turn
# after, so a short session makes prefill dominate and penalises high
# concurrency. Solved from this repo's own artifacts (tune-B/sess12 vs
# tune-D/sess12, identical config at 4 vs 12 turns), prefill is ~18% of wall
# clock at SESS_REQS=12 and ~28% at 4. It cannot fall much below ~12-18% at any
# turn count, because every turn still prefills its own 4k suffix -- so this
# scenario always weights prefill more heavily than production does.
SESS_REQS="${SESS_REQS:-12}"
# SESS_PREFIXES pins the number of distinct prefixes (sessions). It defaults to
# the concurrency, which is the realistic shape -- one session per slot -- but
# that makes each sess<c> a DIFFERENT prompt set, so acceptance length (and
# therefore tok/s) is not comparable across concurrencies. Pinning it holds the
# content fixed.
#
# WARNING -- pinning SESS_PREFIXES and SESS_SEED makes two sess arms generate
# BYTE-IDENTICAL prompts, so running them in ONE invocation is invalid: the
# server keeps prefix caching on and never restarts between scenarios, so arm 2
# replays prompts arm 1 just wrote into the KV cache and pays almost no prefill.
# Measured on this box: 90.5% of arm 2's input tokens were cache hits, its
# warmup request took 41.9 s against arm 1's 75.5 s, and it reached the halfway
# mark in half the wall clock -- an ordering artifact that flatters whichever
# arm runs second. Run ONE scenario per invocation with a restart between:
#   SESS_PREFIXES=12 SESS_REQS=3 SESS_SEED=11908 ./bench.sh a-c8  --scenarios sess8
#   ./stop.sh && ./start.sh
#   SESS_PREFIXES=12 SESS_REQS=3 SESS_SEED=11908 ./bench.sh a-c12 --scenarios sess12
SESS_PREFIXES="${SESS_PREFIXES:-}"
# SESS_SEED overrides the per-scenario seed. Repeats of one config want the SAME
# seed (so only sampling varies); comparing two concurrencies wants the same seed
# on BOTH arms (so only concurrency varies). The built-in per-scenario seeds are
# deliberately different, which is right for the realistic one-session-per-slot
# shape and wrong for a controlled A/B.
SESS_SEED="${SESS_SEED:-}"
# Bounded for the same reason start.sh bounds MAX_SEQS: these feed arithmetic that
# sizes a benchmark run, and an unbounded value produces either a nonsense
# --num-prompts or a run that never ends. Digit count is capped in the pattern so
# a huge literal never reaches $(( )).
for _v in SESS_REQS SESS_PREFIXES; do
  _val="${!_v}"
  [[ -z "${_val}" ]] && continue
  if ! [[ "${_val}" =~ ^[0-9]{1,4}$ ]] || (( 10#${_val} < 1 )) || (( 10#${_val} > 1000 )); then
    echo "${_v} must be an integer between 1 and 1000, got '${_val}'"
    exit 1
  fi
  # Force base 10, as start.sh does for MAX_SEQS. Without this SESS_REQS=012 is
  # octal to bash (10 turns) but decimal to the client (12 prefixes), so the run
  # silently has a different prefill fraction than every log line claims -- and
  # 08/09 abort sess() mid-function, which skips run_scenario entirely and still
  # exits 0 with the success banner.
  printf -v "${_v}" '%d' "$(( 10#${_val} ))"
done
unset _v _val
sess() {
  local c="$1" seed="${SESS_SEED:-$2}"
  local prefixes="${SESS_PREFIXES:-$c}"
  local prompts="${SESS_PREFIXES:+$((SESS_PREFIXES * SESS_REQS))}"
  prompts="${prompts:-$((c * SESS_REQS))}"
  DATASET=prefix_repetition run_scenario "sess${c}" "${seed}" \
    --prefix-repetition-prefix-len 28672 \
    --prefix-repetition-suffix-len 4096 \
    --prefix-repetition-output-len 1024 \
    --prefix-repetition-num-prefixes "${prefixes}" \
    --num-prompts "${prompts}" --max-concurrency "${c}"
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
