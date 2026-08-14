#!/usr/bin/env bash
set -euo pipefail

# unsloth/Qwen3.8-27B-NVFP4 on vLLM (DGX Spark / GB10, aarch64)
#
# NVFP4-specific notes:
#   - Checkpoint is compressed-tensors format (dynamic NVFP4: mixed 4-bit
#     weights/activations with 8-bit groups for sensitive modules), so we pass
#     --quantization compressed-tensors like the Qwen3.6-27B-NVFP4 setup.
#   - Requires vLLM >= 0.25.0 (nightly-aarch64 is 0.26.1rc1+), flashinfer and
#     nvidia-cutlass-dsl are bundled in the image (CUTE_DSL_ARCH=sm_121a below).
#   - Speculative decoding: no DFlash drafter exists for Qwen3.8, but unsloth
#     ships MTP weights (model_mtp.safetensors) and documents MTP spec decode:
#       --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
#     (faster decode, somewhat lower peak throughput, per unsloth docs).
#   - The checkpoint carries a calibrated FP8 kv_cache_scheme (static
#     per-tensor scales, 2x KV memory savings). We do NOT pass
#     --kv-cache-dtype, so vLLM "auto" applies the checkpoint's FP8 KV cache.
#     Pass --kv-cache-dtype bfloat16 to override back to bf16.
#   - Dense model (SwiGLU gate/up/down_proj, no experts) -> no --moe-backend.
#
# The repo ships its own chat_template.jinja; thinking mode and
# preserve_thinking are ON by default (disable per request via
# chat_template_kwargs, see model card).
#
# Default sampling params (thinking mode, per model card):
#   temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
#   presence_penalty=0.0, repetition_penalty=1.0
# temp/top_p/top_k come from the repo's generation_config.json (vLLM default
# --generation-config auto); min_p/presence_penalty/repetition_penalty are
# vLLM's own defaults. For instruct / non-thinking requests, clients should
# override per request: temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5

MODEL_ID="unsloth/Qwen3.8-27B-NVFP4"
SERVED_MODEL_NAME="qwen38-27b-unsloth-nvfp4"
IMAGE="vllm/vllm-openai:nightly-aarch64"
CONTAINER_NAME="qwen3.8-27b-nvfp4"
HOST="0.0.0.0"
PORT="8888"
PID_FILE=".vllm.pid"
LOG_FILE=".vllm.log"
WORK_DIR="$(pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
TRITON_CACHE_DIR="${WORK_DIR}/.cache/triton"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

# Context is extended to 1M tokens via YaRN (model card "Processing Ultra-Long
# Texts"): factor 4.0 over the native 262144, applied through --hf-overrides.
# Note (per model card): static YaRN can slightly impact short-context
# quality. With FP8 KV cache a single 1M-token sequence needs ~32GB of KV;
# concurrent long sequences are admitted as KV space allows.

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}"

# Pick up HF_TOKEN from ~/.bashrc (defined without `export` there) so the
# container gets authenticated Hub access (higher rate limits, faster downloads).
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.bashrc" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=["'"'"']\?\([A-Za-z0-9_-]\+\).*/\1/p' "${HOME}/.bashrc" | head -1)"
fi
export HF_TOKEN

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
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
  --gpu-memory-utilization 0.84 \
  --max-model-len 1000000 \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --skip-mm-profiling \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' \
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

echo "Waiting for HTTP readiness at ${READY_URL}"
heartbeat=0
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "vLLM container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  # The log itself is streaming above; only a light heartbeat every ~30s.
  if (( heartbeat % 6 == 0 )); then
    echo "  still starting..."
  fi
  heartbeat=$((heartbeat + 1))
  sleep 5
done

echo "vLLM is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"

echo "vLLM is ready and responding; shell is now free."
