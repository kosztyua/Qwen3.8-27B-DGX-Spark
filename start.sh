#!/usr/bin/env bash
set -euo pipefail

# unsloth/Qwen3.8-27B-NVFP4 on vLLM (DGX Spark / GB10, aarch64)
#
# NVFP4-specific notes:
#   - Checkpoint is compressed-tensors format (dynamic NVFP4: mixed 4-bit
#     weights/activations with 8-bit groups for sensitive modules), so we pass
#     --quantization compressed-tensors like the Qwen3.6-27B-NVFP4 setup.
#   - IMAGE IS PINNED BY DIGEST (v0.27.2rc1.dev110+gacb0f1dcd, the 2026-08-15
#     nightly). Reasons: (a) stable 0.27.1 cannot load this checkpoint
#     ("MergedColumnParallelLinear has no attribute data"); (b) this nightly
#     contains the SM12x FlashInfer XQA decode path (PR #49718) and the fused
#     GDN MTP decode kernel (PR #51674), and an open revert (PR #51987) may
#     remove XQA from later nightlies. Re-evaluate the pin once these merge:
#     #52244 (GDN prefix-cache hits under MTP), #52000 (uniform-decode graph
#     dispatch), #52013 (dedicated MTP draft lm_head), #50862 (FlashInfer GDN
#     prefill on SM12x), #51954 (GDN decode gate copies).
#   - Speculative decoding: unsloth ships MTP weights (model_mtp.safetensors).
#     num_speculative_tokens=5 measured fastest single-stream on this box
#     (27.5 tok/s vs 23.8 at k=2; c=4 aggregate dips ~7%). AVOID k=3: this
#     build mis-drafts in the 4-token decode path (c=1, k=3) and single-stream
#     collapses to ~15 tok/s.
#   - --cudagraph-capture-sizes must contain c*(1+k) for c=1..max-num-seqs or
#     decode batches run attention eagerly (PR #52000): for k=5 that is
#     6/12/18/24 (all in the ladder below); a k=2 config would need 3/6/9/12,
#     i.e. ladder 1 2 3 4 6 8 9 12 16 24.
#   - The checkpoint carries a calibrated FP8 kv_cache_scheme (static
#     per-tensor scales, 2x KV memory savings). We pass --kv-cache-dtype fp8
#     explicitly (matches vLLM's official recipe; "auto" resolves to the same
#     today, the flag guards against a silent bf16 fallback on image bumps).
#   - Dense model (SwiGLU gate/up/down_proj, no experts) -> no --moe-backend.
#
# The repo ships its own chat_template.jinja; thinking mode and
# preserve_thinking are ON by default (disable per request via
# chat_template_kwargs {"enable_thinking": false}). The model defaults to
# reasoning_effort "xhigh"; for shorter thinking traces add e.g.
#   --default-chat-template-kwargs '{"reasoning_effort": "medium"}'
#
# Default sampling params (thinking mode, per model card):
#   temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
#   presence_penalty=0.0, repetition_penalty=1.0
# temp/top_p/top_k come from the repo's generation_config.json (vLLM default
# --generation-config auto); min_p/presence_penalty/repetition_penalty are
# vLLM's own defaults. For instruct / non-thinking requests, clients should
# override per request: temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5

# Anchor all paths to the repo dir regardless of the caller's cwd (matches
# download.sh/bench.sh, and keeps .cache/ + log/pid files in one place).
cd "$(dirname "$0")"

MODEL_ID="unsloth/Qwen3.8-27B-NVFP4"
SERVED_MODEL_NAME="qwen38-27b-unsloth-nvfp4"
# Pinned: vllm/vllm-openai:nightly-aarch64 as of 2026-08-15 (see header notes)
IMAGE="vllm/vllm-openai@sha256:b5c860acda75d737a8e58cc99ba86ff13982695dceae194f906c2d7b54979358"
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
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting vLLM container for ${MODEL_ID} (NVFP4, MTP speculative decoding)"
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
  --quantization compressed-tensors \
  --attention-backend triton_attn \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.84 \
  "${CONTEXT_ARGS[@]}" \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --skip-mm-profiling \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}' \
  --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

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

# ── MTP health probe ─────────────────────────────────────────────────────────
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
  python3 - "${before:-0 0}" "${after:-0 0}" <<'PY'
import sys
a0, d0 = map(float, sys.argv[1].split())
a1, d1 = map(float, sys.argv[2].split())
drafts = d1 - d0
if drafts <= 0:
    print("n/a (no drafts observed)")
    sys.exit(1)
rate = (a1 - a0) / drafts
print(f"{rate:.2f}")
sys.exit(0 if rate >= 0.55 else 1)
PY
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "WARNING: python3 not found on the host — skipping the MTP health probe."
  echo "If decode runs ~40% slower than expected (~16 instead of ~27 tok/s), restart with ./stop.sh && ./start.sh"
else
  echo "Probing speculative-decode health (MTP position-0 acceptance)..."
  if rate=$(probe_acceptance); then
    echo "MTP acceptance healthy (position-0 rate ${rate})"
  else
    echo "MTP acceptance degraded (position-0 rate ${rate}) — restarting container once (known per-launch initialization lottery)"
    docker restart "${CONTAINER_NAME}" >/dev/null
    wait_ready || exit 1
    if rate=$(probe_acceptance); then
      echo "MTP acceptance healthy after restart (position-0 rate ${rate})"
    else
      echo "WARNING: MTP acceptance still degraded (position-0 rate ${rate}); decode will run ~40% slower than optimal. Try ./stop.sh && ./start.sh"
    fi
  fi
fi

echo "vLLM is ready and responding; shell is now free."
