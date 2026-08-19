#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B on vLLM (DGX Spark / GB10, aarch64)
#
# Serves one of two NVFP4 checkpoints of the same base model, selected by
# VARIANT (see the case block below). The default is the fastest measured
# config on this box: RadixArk/Qwen3.8-27B-NVFP4 with DSpark speculative
# decoding. An explicitly gated dflash2 variant is present for evaluation;
# it is not a production default and requires a separately supplied image.
#
# Notes that apply to every variant:
#   - The normal IMAGE is pinned by digest (v0.27.2rc1.dev110+gacb0f1dcd, the
#     2026-08-15 nightly); dflash2 requires its caller-supplied image to be
#     digest-pinned too. Reasons: (a) stable 0.27.1 cannot load this model family
#     ("MergedColumnParallelLinear has no attribute data"); (b) this nightly
#     contains the SM12x FlashInfer XQA decode path (PR #49718) and the fused
#     GDN MTP decode kernel (PR #51674), and an open revert (PR #51987) may
#     remove XQA from later nightlies. Re-evaluate the pin once these merge:
#     #52244 (GDN prefix-cache hits under MTP), #52000 (uniform-decode graph
#     dispatch), #52013 (dedicated MTP draft lm_head), #50862 (FlashInfer GDN
#     prefill on SM12x), #51954 (GDN decode gate copies).
#   - --cudagraph-capture-sizes must contain c*(1+k) for c=1..max-num-seqs or
#     decode batches run attention eagerly (PR #52000); each variant sets its
#     own ladder for its k.
#   - --kv-cache-dtype fp8 halves KV memory vs bf16. The unsloth checkpoint
#     ships calibrated KV scales; RadixArk's uses 1.0 defaults (measured
#     quality was indistinguishable at short/medium context, see README).
#   - Dense model (SwiGLU gate/up/down_proj, no experts) -> no --moe-backend.
#   - MTP-variant-only bug in this build: num_speculative_tokens=3 mis-drafts
#     in the 4-token decode path and single-stream collapses; k=5 is the
#     measured optimum for MTP.
#
# Chat template: thinking mode and preserve_thinking are ON by default
# (disable per request via chat_template_kwargs {"enable_thinking": false}).
# The model defaults to reasoning_effort "xhigh"; for shorter thinking traces
# add e.g. --default-chat-template-kwargs '{"reasoning_effort": "medium"}'
#
# Default sampling params (thinking mode, per model card):
#   temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
#   presence_penalty=0.0, repetition_penalty=1.0
# temp/top_p/top_k come from the checkpoint's generation_config.json (vLLM
# default --generation-config auto); min_p/presence_penalty/repetition_penalty
# are vLLM's own defaults. For instruct / non-thinking requests, clients should
# override per request: temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5

# Anchor all paths to the repo dir regardless of the caller's cwd (matches
# download.sh/bench.sh, and keeps .cache/ + log/pid files in one place).
cd "$(dirname "$0")"

# Speculative-decoding variant.
#   VARIANT=dspark (default): RadixArk modelopt checkpoint + the 1.4B DSpark
#     block drafter (k=7, probabilistic draft sampling — greedy drafting
#     measures ~23% slower). Fastest measured config: 38-43 tok/s
#     single-stream on real reasoning/code content, 122.8 tok/s aggregate at
#     8 streams. 262k context, 1,274,196-token KV pool (pinned 68 GiB).
#     DSPARK_TARGET=unsloth swaps in the unsloth checkpoint (works, ~10-30%
#     slower; see README).
#   VARIANT=mtp: unsloth checkpoint + its built-in MTP head (k=5). Best on
#     unpredictable single-stream content (27.5 tok/s random c=1) and the
#     only variant with the CONTEXT_1M option and vision validation.
#   VARIANT=dflash2: experimental z-lab 1.9B DFlash 2 drafter (k=7) over the
#     RadixArk target. Upstream H200 measurements are promising, but vLLM
#     support is still PR-only and has not been validated on GB10. It is
#     fail-closed behind DFLASH2_EXPERIMENTAL=1 and DFLASH2_IMAGE=<digest>.
# The dspark drafter config needs its architectures field patched from
# DSparkDraftModel (which vLLM's registry routes to a DeepSeek-V4 class) to
# Qwen3DSparkModel; this script maintains a patched local copy automatically.
VARIANT="${VARIANT:-dspark}"
DSPARK_DRAFT_ID="RadixArk/Qwen3.8-27B-DSpark"
DFLASH2_DRAFT_ID="z-lab/Qwen3.8-27B-DFlash2"
DRAFT_LOCAL_DIR_NAME="dspark-qwen38-local"
case "${VARIANT}" in
  mtp)
    MODEL_ID="unsloth/Qwen3.8-27B-NVFP4"
    SERVED_MODEL_NAME="qwen38-27b-unsloth-nvfp4"
    QUANT_ARGS=(--quantization compressed-tensors)
    SPEC_CONFIG='{"method": "mtp", "num_speculative_tokens": 5}'
    # c*(1+k) for c=1..8 at k=5 (see ladder note above)
    LADDER=(1 2 4 6 8 12 16 18 24 30 36 42 48)
    SPEC_WIDTH=6
    PROBE_MIN="0.55"
    # The concurrency/KV tuning below was measured on the dspark variant only.
    # MTP has no separate drafter and a ~2.1M-token pool, so pinning dspark's
    # 68 GiB here would shrink it for no measured reason. Keep MTP on the
    # previously shipped defaults until someone benchmarks it.
    DEFAULT_MAX_SEQS=8
    DEFAULT_KV_CACHE_MEMORY=""
    ;;
  dspark)
    DRAFT_ID="${DSPARK_DRAFT_ID}"
    # DSPARK_TARGET=radixark (default): fastest measured config overall.
    # DSPARK_TARGET=unsloth: same drafter over the unsloth checkpoint —
    # works, and still beats MTP on real content (39.1/34.6/30.9 tok/s),
    # but 10-30% behind the RadixArk target everywhere: the
    # compressed-tensors kernel path is ~12ms/step slower than modelopt,
    # and draft acceptance is lower against this target.
    case "${DSPARK_TARGET:-radixark}" in
      radixark)
        MODEL_ID="RadixArk/Qwen3.8-27B-NVFP4"
        SERVED_MODEL_NAME="qwen38-27b-radixark-nvfp4"
        # The checkpoint's hf_quant_config.json (modelopt) is auto-detected.
        QUANT_ARGS=()
        ;;
      unsloth)
        MODEL_ID="unsloth/Qwen3.8-27B-NVFP4"
        SERVED_MODEL_NAME="qwen38-27b-unsloth-nvfp4"
        QUANT_ARGS=(--quantization compressed-tensors)
        ;;
      *)
        echo "Unknown DSPARK_TARGET '${DSPARK_TARGET}' (expected 'radixark' or 'unsloth')"
        exit 1
        ;;
    esac
    SPEC_CONFIG='{"method": "dspark", "model": "/root/.cache/huggingface/'"${DRAFT_LOCAL_DIR_NAME}"'", "num_speculative_tokens": 7, "draft_sample_method": "probabilistic"}'
    # c*(1+k) for c=1..8 at k=7
    LADDER=(1 2 4 8 16 24 32 40 48 56 64)
    SPEC_WIDTH=8
    # DSpark position-0 acceptance is legitimately lower than MTP's
    PROBE_MIN="0.25"
    # Measured optimum for this variant (README "Concurrency and KV sizing").
    DEFAULT_MAX_SEQS=12
    DEFAULT_KV_CACHE_MEMORY=73014444032   # 68 GiB -> 1,274,196 tokens
    ;;
  dflash2)
    if [[ "${DFLASH2_EXPERIMENTAL:-0}" != "1" ]]; then
      cat <<'MSG'
VARIANT=dflash2 is preparation-only and not validated on DGX Spark. Its vLLM
support is still an open upstream PR, and a concurrency crash has been reported
on adjacent SM120 hardware. See DFLASH2_EVALUATION.md.

After building and pinning a compatible image, explicitly opt in with:

    DFLASH2_EXPERIMENTAL=1 DFLASH2_IMAGE=<image@sha256:...> VARIANT=dflash2 ./start.sh
MSG
      exit 1
    fi
    if [[ -z "${DFLASH2_IMAGE:-}" ]]; then
      echo "VARIANT=dflash2 requires DFLASH2_IMAGE pinned to an image containing vLLM PR #52816."
      echo "The repository's normal pinned vLLM image predates DFlash 2 support."
      exit 1
    fi
    if ! [[ "${DFLASH2_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
      echo "DFLASH2_IMAGE must end in @sha256:<64 lowercase hex characters>; tags are intentionally rejected."
      exit 1
    fi
    MODEL_ID="RadixArk/Qwen3.8-27B-NVFP4"
    SERVED_MODEL_NAME="qwen38-27b-radixark-nvfp4"
    DRAFT_ID="${DFLASH2_DRAFT_ID}"
    QUANT_ARGS=()
    SPEC_CONFIG='{"method": "dflash", "model": "'"${DRAFT_ID}"'", "num_speculative_tokens": 7}'
    # c*(1+k) for c=1..8 at k=7
    LADDER=(1 2 4 8 16 24 32 40 48 56 64)
    SPEC_WIDTH=8
    PROBE_MIN="0.50"
    # Conservative evaluation defaults. Let vLLM profile the KV pool around
    # the larger BF16 drafter, and prove single-stream stability before raising
    # concurrency (the upstream SM120 report failed at c=4).
    DEFAULT_MAX_SEQS=1
    DEFAULT_KV_CACHE_MEMORY=""
    ;;
  *)
    echo "Unknown VARIANT '${VARIANT}' (expected 'mtp', 'dspark', or 'dflash2')"
    exit 1
    ;;
esac

# ── tuning knobs ─────────────────────────────────────────────────────────────
#
# DSpark's MAX_SEQS and KV_CACHE_MEMORY default to its measured optimum, which
# is NOT what shipped before: the old behaviour is MAX_SEQS=8
# KV_CACHE_MEMORY= (empty). MTP retains its prior defaults; dFlash2 starts with
# conservative evaluation defaults. SSM_DTYPE and PROFILER default to off.
#
# MAX_SEQS=<n>            Concurrent sequence cap: dspark=12 (measured knee),
#                         mtp=8, dflash2=1 (conservative initial evaluation).
#                         The per-variant ladder above only covers c=1..8, so
#                         anything higher must extend it or the wider decode
#                         batches fall back to eager attention (PR #52000) —
#                         that extension is done automatically here.
#                         Measured +9.9% at 12 vs 8 (six runs, three per arm,
#                         identical prompts, restart between arms; 40.68 -> 44.73
#                         tok/s, arms non-overlapping, acceptance matched).
#                         Likely a floor for production, whose longer sessions
#                         spend less of the run prefilling. c=16 was not measured
#                         under the controlled method -- treat it as unknown.
# KV_CACHE_MEMORY=<bytes> Pin the KV pool to an exact size instead of deriving it
#                         from --gpu-memory-utilization. DSpark defaults to the
#                         measured 68 GiB; MTP and dflash2 derive it. Deterministic across
#                         restarts, where the utilization fraction is not: two
#                         starts of the identical config measured 1,413,515 and
#                         1,419,112 KV tokens, because 0.84 is a share of
#                         whatever happened to be free at boot. Default
#                         73014444032 (68 GiB) = 1,274,196 tokens. 12 sessions of
#                         ~32k live context is 393k tokens = 31% of the pool; the
#                         45% observed in benchmarks is that plus retained
#                         prefix-cache blocks, which is what the pool is for.
#                         Set KV_CACHE_MEMORY= (empty) to restore the old
#                         utilization-derived sizing.
# SSM_DTYPE=<dtype>       Gated-DeltaNet recurrent state dtype (float32 default).
#                         48 of 64 layers are GDN and their state is written once
#                         per draft position, so this is the largest non-weight
#                         term in the step. bfloat16 halves it — but it perturbs
#                         the sampled distribution, and under probabilistic
#                         rejection sampling acceptance = 1 - TV(p,q), so validate
#                         on acceptance length, not just output quality.
#                         Use bfloat16, never float16: vLLM's fused GDN decode
#                         path accepts only float32/bfloat16.
# PROFILER=1              Expose /start_profile and /stop_profile for a torch
#                         trace. Only the exact value 1 enables it; anything else
#                         (0, no, false, empty, unset) leaves it off. Development
#                         only — do not leave on.
#
# Defaults come from the variant block above, because the tuning was measured on
# dspark only: dspark defaults to the measured optimum (12 / 68 GiB), mtp keeps
# the previously shipped 8 / utilization-derived, and dflash2 deliberately
# starts at 1 / utilization-derived. MAX_SEQS=8 with KV_CACHE_MEMORY= (empty)
# restores the pre-tuning behaviour on dspark or mtp.
MAX_SEQS="${MAX_SEQS:-${DEFAULT_MAX_SEQS}}"
# Note the '-' rather than ':-': an explicitly empty KV_CACHE_MEMORY= means
# "fall back to deriving the pool from --gpu-memory-utilization", while unset
# means "use the variant default".
KV_CACHE_MEMORY="${KV_CACHE_MEMORY-${DEFAULT_KV_CACHE_MEMORY}}"
# Bounded because the ladder loop below allocates per iteration: MAX_SEQS=1e19
# once grew a bash array until memguard killed the engine (2026-08-17). The
# pattern rejects leading zeros, so the arithmetic below is unambiguous.
if ! [[ "${MAX_SEQS}" =~ ^[1-9][0-9]{0,2}$ ]] || (( MAX_SEQS > 256 )); then
  echo "MAX_SEQS must be an integer between 1 and 256, got '${MAX_SEQS}'"
  exit 1
fi
if (( MAX_SEQS > 8 )); then
  for (( _c = 9; _c <= MAX_SEQS; _c++ )); do
    LADDER+=( "$(( SPEC_WIDTH * _c ))" )
  done
fi

TUNING_ARGS=()
[[ -n "${KV_CACHE_MEMORY:-}" ]] && TUNING_ARGS+=(--kv-cache-memory "${KV_CACHE_MEMORY}")
[[ -n "${SSM_DTYPE:-}" ]]       && TUNING_ARGS+=(--mamba-ssm-cache-dtype "${SSM_DTYPE}")
if [[ "${PROFILER:-0}" == "1" ]]; then
  TUNING_ARGS+=(--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/root/.cache/huggingface/prof", "max_iterations": 5}')
fi


# Pinned: vllm/vllm-openai:nightly-aarch64 as of 2026-08-15 (see header notes).
# DFlash 2 support landed after this build and remains PR-only, so that variant
# must supply a separately built, digest-pinned image and can never silently
# fall back to the normal image.
DEFAULT_IMAGE="vllm/vllm-openai@sha256:b5c860acda75d737a8e58cc99ba86ff13982695dceae194f906c2d7b54979358"
if [[ "${VARIANT}" == "dflash2" ]]; then
  IMAGE="${DFLASH2_IMAGE}"
else
  IMAGE="${DEFAULT_IMAGE}"
fi
CONTAINER_NAME="qwen3.8-27b-nvfp4"
SGLANG_CONTAINER_NAME="qwen3.8-27b-sglang"
HOST="0.0.0.0"
PORT="8888"
PID_FILE=".vllm.pid"
LOG_FILE=".vllm.log"
WORK_DIR="$(pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
TRITON_CACHE_DIR="${WORK_DIR}/.cache/triton"
VLLM_CACHE_DIR="${WORK_DIR}/.cache/vllm"
FLASHINFER_CACHE_DIR="${WORK_DIR}/.cache/flashinfer"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

# Context: native 262,144 tokens by default (no RoPE modification, no quality
# caveat). CONTEXT_1M=1 ./start.sh extends to 1M via static YaRN (model card
# "Processing Ultra-Long Texts": factor 4.0 over the native 262144, applied
# through --hf-overrides). Note (per model card): static YaRN can slightly
# impact short-context quality — that is why it is opt-in. With FP8 KV cache a
# single 1M-token sequence needs ~32GB of KV; the ~2.26M-token pool admits ~8
# concurrent 262k sequences or 2 x 1M.
CONTEXT_ARGS=(--max-model-len 262144)
if [[ "${CONTEXT_1M:-0}" == "1" ]]; then
  if [[ "${VARIANT}" != "mtp" ]]; then
    echo "CONTEXT_1M=1 is only validated with VARIANT=mtp (the YaRN recipe is untested on the RadixArk checkpoint and external drafters)."
    exit 1
  fi
  CONTEXT_ARGS=(
    --max-model-len 1000000
    --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
  )
fi

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${VLLM_CACHE_DIR}" "${FLASHINFER_CACHE_DIR}"

# Pick up HF_TOKEN from ~/.bashrc (defined without `export` there) so the
# container gets authenticated Hub access (higher rate limits, faster downloads).
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.bashrc" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=["'"'"']\?\([A-Za-z0-9_-]\+\).*/\1/p' "${HOME}/.bashrc" | head -1)"
fi
export HF_TOKEN

# Both engines bind the same port and share the GPU: refuse to double-start.
if docker ps --format '{{.Names}}' | grep -qxF "${SGLANG_CONTAINER_NAME}"; then
  echo "The SGLang server (${SGLANG_CONTAINER_NAME}) is running — stop it first: ./stop.sh"
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qxF "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qxF "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    # This early exit used to be harmless, because the script had no tunables.
    # Now it can silently discard the entire override matrix: someone runs
    # `MAX_SEQS=16 ./start.sh`, sees success, and benchmarks the OLD config.
    # Compare what is actually serving against what we would have launched.
    if ! mapfile -t _live < <(docker inspect -f '{{range .Args}}{{println .}}{{end}}' "${CONTAINER_NAME}" 2>/dev/null) || (( ${#_live[@]} == 0 )); then
      echo "  (container vanished while inspecting it; re-run ./start.sh)"
      exit 1
    fi
    _live_seqs=""; _live_kv=""; _live_ssm=""
    for (( _i = 0; _i < ${#_live[@]}; _i++ )); do
      case "${_live[_i]}" in
        --max-num-seqs)          _live_seqs="${_live[_i+1]:-}" ;;
        --kv-cache-memory)       _live_kv="${_live[_i+1]:-}" ;;
        --mamba-ssm-cache-dtype) _live_ssm="${_live[_i+1]:-}" ;;
      esac
    done
    _drift=0
    [[ "${_live_seqs}" != "${MAX_SEQS}" ]] && { echo "  !! running --max-num-seqs ${_live_seqs:-<unset>}, this invocation wanted ${MAX_SEQS}"; _drift=1; }
    [[ "${_live_kv}" != "${KV_CACHE_MEMORY}" ]] && { echo "  !! running --kv-cache-memory ${_live_kv:-<unset>}, this invocation wanted ${KV_CACHE_MEMORY:-<unset>}"; _drift=1; }
    [[ "${_live_ssm}" != "${SSM_DTYPE:-}" ]] && { echo "  !! running --mamba-ssm-cache-dtype ${_live_ssm:-<unset>}, this invocation wanted ${SSM_DTYPE:-<unset>}"; _drift=1; }
    if (( _drift )); then
      echo "  The running server does NOT match the requested configuration."
      echo "  Nothing was applied. Restart to apply it:  ./stop.sh && ./start.sh"
      exit 1
    fi
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

# Memory pre-flight. vLLM reserves ~102 GiB of the GB10's unified memory;
# booting while the previous engine's memory is still being released can
# starve the host into a hard freeze (container --memory caps don't cover
# GPU/unified allocations). Wait until the memory is actually free.
echo "Waiting for >= 100 GiB of available memory before booting..."
for i in $(seq 1 24); do
  mem_avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "${mem_avail}" -ge 100 ]; then break; fi
  sleep 5
done
if [ "${mem_avail}" -lt 100 ]; then
  echo "Only ${mem_avail} GiB available after 2 minutes — refusing to boot to protect the host."
  echo "Stop whatever is using the memory (./stop.sh for the other engine) and retry."
  exit 1
fi
echo "OK: ${mem_avail} GiB available"

# External-drafter variants: make sure the target and drafter are cached. For
# DSpark only, also maintain the patched local copy (architectures field fix).
if [[ "${VARIANT}" == "dspark" || "${VARIANT}" == "dflash2" ]]; then
  for repo in "${MODEL_ID}" "${DRAFT_ID}"; do
    cache_dir="${HF_HOME}/hub/models--${repo//\//--}"
    if ! ls "${cache_dir}"/snapshots/*/*.safetensors >/dev/null 2>&1; then
      echo "${repo} not in cache — downloading"
      docker run --rm --network host \
        -e HF_HOME=/root/.cache/huggingface -e HF_TOKEN="${HF_TOKEN:-}" \
        -v "${HF_HOME}:/root/.cache/huggingface" \
        --entrypoint python3 "${IMAGE}" \
        -c "from huggingface_hub import snapshot_download; snapshot_download('${repo}')"
    fi
  done
  if [[ "${VARIANT}" == "dspark" ]]; then
    DRAFT_LOCAL="${HF_HOME}/${DRAFT_LOCAL_DIR_NAME}"
    if [[ ! -s "${DRAFT_LOCAL}/model.safetensors" ]]; then
      echo "Creating patched drafter copy at ${DRAFT_LOCAL} (architectures -> Qwen3DSparkModel)"
      snap="$(ls -d "${HF_HOME}"/hub/models--RadixArk--Qwen3.8-27B-DSpark/snapshots/*/ | head -1)"
      mkdir -p "${DRAFT_LOCAL}"
      cp -L "${snap}/model.safetensors" "${DRAFT_LOCAL}/"
      sed 's/"DSparkDraftModel"/"Qwen3DSparkModel"/' "${snap}/config.json" > "${DRAFT_LOCAL}/config.json"
    fi
  fi
fi

echo "Starting vLLM container for ${MODEL_ID} (NVFP4, ${VARIANT} speculative decoding)"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching vLLM container
EOF

# .cache/vllm holds the torch.compile artifact cache and .cache/flashinfer the
# FlashInfer autotune cache; mounting both cuts warm-restart engine init from
# ~3 min to ~30 s.
docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --privileged \
  --gpus all \
  -e VLLM_TARGET_DEVICE=cuda \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high \
  -e CUTE_DSL_ARCH=sm_121a \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${TRITON_CACHE_DIR}:/root/.triton" \
  -v "${VLLM_CACHE_DIR}:/root/.cache/vllm" \
  -v "${FLASHINFER_CACHE_DIR}:/root/.cache/flashinfer" \
  --entrypoint vllm \
  "${IMAGE}" \
  serve \
  "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  "${QUANT_ARGS[@]}" \
  --attention-backend triton_attn \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.84 \
  "${CONTEXT_ARGS[@]}" \
  "${TUNING_ARGS[@]}" \
  --max-num-seqs "${MAX_SEQS}" \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --skip-mm-profiling \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --speculative-config "${SPEC_CONFIG}" \
  --cudagraph-capture-sizes "${LADDER[@]}" \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

# Runtime memory guard: kills the container before unified-memory pressure
# can freeze the host (see memguard.sh). stop.sh cleans it up.
if [[ -f .memguard.pid ]] && kill -0 "$(cat .memguard.pid)" 2>/dev/null; then
  kill "$(cat .memguard.pid)" 2>/dev/null || true
fi
nohup ./memguard.sh "${CONTAINER_NAME}" >/dev/null 2>&1 &
echo $! > .memguard.pid
echo "Runtime memory guard active (memguard.sh, log: .memguard.log)"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Stream the vLLM startup log to the terminal AND record it in .vllm.log.
# $! tracks tee (pipeline tail); killing it on exit SIGPIPEs docker logs.
docker logs -f "${CONTAINER_NAME}" 2>&1 | tee -a "${LOG_FILE}" &
log_follow_pid=$!

wait_ready() {
  local heartbeat=0
  until curl -fsS -m 5 "${READY_URL}" >/dev/null 2>&1; do
    if ! docker ps --format '{{.Names}}' | grep -qxF "${CONTAINER_NAME}"; then
      echo "vLLM container exited before becoming ready"
      tail -n 200 "${LOG_FILE}" || true
      return 1
    fi
    # The log itself is streaming above; only a light heartbeat every ~30s.
    if (( heartbeat % 6 == 0 )); then
      echo "  still starting..."
    fi
    heartbeat=$((heartbeat + 1))
    sleep 5
  done
}

echo "Waiting for HTTP readiness at ${READY_URL}"
wait_ready || exit 1

echo "vLLM is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"

# ── speculative-decoding health probe ─────────────────────────────────────────────
# This vLLM build has a per-launch initialization lottery: some starts come up
# with a mis-drafting speculative-decode state (MTP position-0 acceptance drops
# from ~77% to ~44% and single-stream decode falls from ~27 to ~16 tok/s).
# A plain container restart fixes it. Probe acceptance and restart once if bad.
probe_acceptance() {
  local before after
  before=$(curl -fs -m 10 "http://127.0.0.1:${PORT}/metrics" | awk '
    /^vllm:spec_decode_num_accepted_tokens_per_pos_total.*position="0"/ {a=$NF}
    /^vllm:spec_decode_num_drafts_total/ {d=$NF} END {print a+0, d+0}')
  for i in 1 2 3; do
    curl -fs -m 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model": "'"${SERVED_MODEL_NAME}"'", "max_tokens": 150,
           "chat_template_kwargs": {"reasoning_effort": "low"},
           "messages": [{"role": "user", "content": "Briefly describe photosynthesis, then count from 1 to 20."}]}' \
      >/dev/null || true
  done
  after=$(curl -fs -m 10 "http://127.0.0.1:${PORT}/metrics" | awk '
    /^vllm:spec_decode_num_accepted_tokens_per_pos_total.*position="0"/ {a=$NF}
    /^vllm:spec_decode_num_drafts_total/ {d=$NF} END {print a+0, d+0}')
  # Fails closed: zero drafts (probe requests all failed, metrics unreadable,
  # or spec decode not running at all) counts as unhealthy.
  python3 - "${before:-0 0}" "${after:-0 0}" "${PROBE_MIN}" <<'PY'
import sys
a0, d0 = map(float, sys.argv[1].split())
a1, d1 = map(float, sys.argv[2].split())
drafts = d1 - d0
if drafts <= 0:
    print("n/a (no drafts observed)")
    sys.exit(1)
rate = (a1 - a0) / drafts
print(f"{rate:.2f}")
sys.exit(0 if rate >= float(sys.argv[3]) else 1)
PY
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "WARNING: python3 not found on the host — skipping the speculative-decode health probe."
  echo "If decode runs ~40% slower than expected (~16 instead of ~27 tok/s), restart with ./stop.sh && ./start.sh"
else
  echo "Probing speculative-decode health (position-0 acceptance)..."
  if rate=$(probe_acceptance); then
    echo "Spec-decode acceptance healthy (position-0 rate ${rate})"
  else
    echo "Spec-decode acceptance degraded (position-0 rate ${rate}) — restarting container once (known per-launch initialization lottery)"
    docker restart "${CONTAINER_NAME}" >/dev/null
    wait_ready || exit 1
    if rate=$(probe_acceptance); then
      echo "Spec-decode acceptance healthy after restart (position-0 rate ${rate})"
    else
      echo "WARNING: spec-decode acceptance still degraded (position-0 rate ${rate}); decode will run ~40% slower than optimal. Try ./stop.sh && ./start.sh"
    fi
  fi
fi

echo "vLLM is ready and responding; shell is now free."
