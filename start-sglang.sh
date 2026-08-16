#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B on SGLang with DSpark block-speculative decoding (DGX Spark /
# GB10, aarch64). Alternative engine to start.sh — same port, but a
# DIFFERENT checkpoint and served-model name (see below).
#
# Default target: RadixArk/Qwen3.8-27B-NVFP4 (modelopt format), the only
# checkpoint SGLang serves correctly today. Validated on this box 2026-08-16:
# 30+ battery runs with zero repetition loops and zero non-terminating
# responses (greedy included), correct answers throughout, ~34-45 tok/s
# single-stream on real content.
#
# SGLANG_TARGET=unsloth serves unsloth/Qwen3.8-27B-NVFP4 instead, but that
# path is BROKEN (2026-08-15): thinking-mode output collapses into hard
# repetition loops regardless of sampling or verification mode. Suspected
# cause: SGLang's partial support for unsloth's compressed-tensors mixed
# NVFP4/FP8 scheme (logs "Falling back to UnquantizedLinearMethod", bf16 KV
# instead of the calibrated FP8 scheme). It therefore additionally requires
# SGLANG_EXPERIMENTAL=1. Full story in the README's "SGLang status" section.
#
# Known limitations vs start.sh (details in README):
#   - Context is the native 262,144 tokens; no validated YaRN-1M recipe.
#   - The `reasoning_effort` template kwarg is not passed through, so
#     requests think at the model's default (xhigh) effort;
#     `chat_template_kwargs: {"enable_thinking": false}` works.
#
# Flags follow the community GB10 setup (hasso5703/dgx-spark-qwen38). The
# hard --memory caps reduce, but do not eliminate, the GB10 unified-memory
# freeze risk (one hard freeze observed on this box during a cold SGLang
# boot); the pre-flight below refuses to start unless the memory that the
# other engine held has actually been released.

# Anchor all paths to the repo dir regardless of the caller's cwd (matches
# download.sh/bench.sh, and keeps .cache/ + log/pid files in one place).
cd "$(dirname "$0")"

DRAFT_ID="RadixArk/Qwen3.8-27B-DSpark"
case "${SGLANG_TARGET:-radixark}" in
  radixark)
    MODEL_ID="RadixArk/Qwen3.8-27B-NVFP4"
    SERVED_MODEL_NAME="qwen38-27b-radixark-nvfp4"
    ;;
  unsloth)
    if [[ "${SGLANG_EXPERIMENTAL:-0}" != "1" ]]; then
      cat <<'MSG'
SGLANG_TARGET=unsloth is EXPERIMENTAL AND BROKEN: thinking-mode output
degrades into repetition loops (even at the recommended sampling settings).
See the "SGLang status" section of the README for the full story.

To retest it anyway (e.g. after an SGLang update):

    SGLANG_TARGET=unsloth SGLANG_EXPERIMENTAL=1 ./start-sglang.sh

Or just run ./start-sglang.sh for the working RadixArk checkpoint.
MSG
      exit 1
    fi
    MODEL_ID="unsloth/Qwen3.8-27B-NVFP4"
    SERVED_MODEL_NAME="qwen38-27b-unsloth-nvfp4"
    ;;
  *)
    echo "Unknown SGLANG_TARGET '${SGLANG_TARGET}' (expected 'radixark' or 'unsloth')"
    exit 1
    ;;
esac
# Pinned: lmsysorg/sglang:qwen38-27b as of 2026-08-15 (validated on this box)
IMAGE="lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1"
CONTAINER_NAME="qwen3.8-27b-sglang"
VLLM_CONTAINER_NAME="qwen3.8-27b-nvfp4"
HOST="0.0.0.0"
PORT="8888"
PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
WORK_DIR="$(pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
SGLANG_CACHE_DIR="${WORK_DIR}/.cache/sglang"
READY_URL="http://127.0.0.1:${PORT}/health"

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH"; exit 1; }

mkdir -p "${HF_HOME}" "${SGLANG_CACHE_DIR}"

# Pick up HF_TOKEN from ~/.bashrc (defined without `export` there).
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.bashrc" ]]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=["'"'"']\?\([A-Za-z0-9_-]\+\).*/\1/p' "${HOME}/.bashrc" | head -1)"
fi
export HF_TOKEN

# Both engines bind the same port and share the GPU: refuse to double-start.
if docker ps --format '{{.Names}}' | grep -qxF "${VLLM_CONTAINER_NAME}"; then
  echo "The vLLM server (${VLLM_CONTAINER_NAME}) is running — stop it first: ./stop.sh"
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

# Memory pre-flight. SGLang startup (weights + torch.compile + CUDA-graph
# capture) needs most of the GB10's unified memory, and the container
# --memory caps do not fully account GPU allocations — booting while the
# other engine's memory is still held can hard-freeze the whole machine
# (observed once on this box). Wait for the memory to be actually released.
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

# Fetch the target checkpoint and the DSpark drafter (~2.7 GB) if missing,
# so this script works standalone (./download.sh --sglang also gets both).
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

echo "Starting SGLang container for ${MODEL_ID} (NVFP4, DSpark speculative decoding)"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"
echo "First boot spends ~5-9 min in torch.compile; later boots reuse the cache."

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container
EOF

# --memory/--memory-swap hard caps: a runaway allocation on the GB10's unified
# memory can otherwise freeze the whole box. TORCHINDUCTOR_CACHE_DIR persists
# torch.compile artifacts across restarts.
docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus all \
  --memory 100g --memory-swap 100g --shm-size 16g \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${SGLANG_CACHE_DIR}:/cache" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tp-size 1 \
  --mem-fraction-static 0.50 \
  --attention-backend flashinfer \
  --chunked-prefill-size 8192 \
  --disable-prefill-cuda-graph \
  --cuda-graph-max-bs 4 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "${DRAFT_ID}" \
  --speculative-dspark-block-size 7 \
  --speculative-draft-model-quantization unquant \
  --mamba-scheduler-strategy extra_buffer \
  --enable-torch-compile --torch-compile-max-bs 4 \
  --num-continuous-decode-steps 2 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --host "${HOST}" --port "${PORT}" \
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

docker logs -f "${CONTAINER_NAME}" 2>&1 | tee -a "${LOG_FILE}" &
log_follow_pid=$!

echo "Waiting for HTTP readiness at ${READY_URL}"
heartbeat=0
until curl -fsS -m 2 "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qxF "${CONTAINER_NAME}"; then
    echo "SGLang container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  if (( heartbeat % 6 == 0 )); then
    echo "  still starting..."
  fi
  heartbeat=$((heartbeat + 1))
  sleep 5
done

if command -v python3 >/dev/null 2>&1; then
  echo "SGLang is ready — running a generation smoke test..."
  smoke="$(curl -fs -m 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "'"${SERVED_MODEL_NAME}"'", "max_tokens": 30, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": false},
         "messages": [{"role": "user", "content": "Reply with exactly: READY"}]}' \
    | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())
except Exception as e:
    print(f"SMOKE-PARSE-ERROR: {e}")' || true)"
  echo "Smoke test reply: ${smoke}"
else
  echo "python3 not found on the host — skipping the generation smoke test."
fi

echo "SGLang is ready and responding; shell is now free."
echo "OpenAI base URL: http://${HOST}:${PORT}/v1 (model name: ${SERVED_MODEL_NAME})"
