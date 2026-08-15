#!/usr/bin/env bash
set -euo pipefail

# unsloth/Qwen3.8-27B-NVFP4 on SGLang with DSpark block-speculative decoding
# (DGX Spark / GB10, aarch64). Alternative engine to start.sh — same model,
# same port, same served-model name, so clients don't care which one runs.
#
# ############################ EXPERIMENTAL ################################
# STATUS 2026-08-15: BROKEN with this checkpoint. Thinking-mode output
# degrades into hard repetition loops on ordinary prompts, even with the
# model card's recommended sampling (temp 1.0 / top_p 0.95 / top_k 20), and
# both with and without strict rejection-sampling verification. Suspected
# cause: SGLang's partial support for unsloth's compressed-tensors mixed
# NVFP4/FP8 scheme (startup logs "Falling back to UnquantizedLinearMethod"
# and allocates a bf16 KV cache instead of the checkpoint's calibrated FP8
# scheme). The community-validated SGLang path uses the RadixArk modelopt
# checkpoint instead (see hasso5703/dgx-spark-qwen38). During further
# isolation testing the machine hard-froze from unified-memory pressure
# despite the --memory caps below. Full details in the README's "SGLang
# status" section. Requires SGLANG_EXPERIMENTAL=1 to run.
# ##########################################################################
#
# What made this path interesting: it measured ~38-49 tok/s single-stream on
# thinking-heavy content vs ~28-31 for vLLM, and 84.2 tok/s aggregate at 4
# streams vs 65.9 — but the thinking-heavy numbers are contaminated by the
# repetition loops (loops draft near-perfectly), so treat them as invalid
# until the output quality is fixed.
#
# Known limitations vs start.sh (details in README):
#   - Context is the native 262,144 tokens; no validated YaRN-1M recipe.
#   - SGLang does not recognize the `reasoning_effort` template kwarg of
#     unsloth's chat template, so requests think at the model's default
#     (xhigh) effort; `chat_template_kwargs: {"enable_thinking": false}`
#     works to disable thinking entirely.
#
# Flags follow the community GB10 setup (hasso5703/dgx-spark-qwen38); the
# hard --memory caps reduce (but on this box did not eliminate) the risk of
# a unified-memory hard freeze.

# Anchor all paths to the repo dir regardless of the caller's cwd (matches
# download.sh/bench.sh, and keeps .cache/ + log/pid files in one place).
cd "$(dirname "$0")"

if [[ "${SGLANG_EXPERIMENTAL:-0}" != "1" ]]; then
  cat <<'MSG'
start-sglang.sh is currently EXPERIMENTAL AND BROKEN with this checkpoint:
thinking-mode output degrades into repetition loops (even at the recommended
sampling settings), and a test run hard-froze this machine from unified-memory
pressure. See the "SGLang status" section of the README for the full story.

If you want to run it anyway (e.g. to retest after an SGLang update):

    SGLANG_EXPERIMENTAL=1 ./start-sglang.sh

For working SGLang serving of this model today, use the RadixArk modelopt
checkpoint via https://github.com/hasso5703/dgx-spark-qwen38 instead.
MSG
  exit 1
fi

MODEL_ID="unsloth/Qwen3.8-27B-NVFP4"
DRAFT_ID="RadixArk/Qwen3.8-27B-DSpark"
SERVED_MODEL_NAME="qwen38-27b-unsloth-nvfp4"
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

# The DSpark drafter (~2.7 GB) is fetched by ./download.sh --sglang; fetch it
# here too if missing so this script works standalone.
if ! ls "${HF_HOME}"/hub/models--RadixArk--Qwen3.8-27B-DSpark/snapshots/*/model.safetensors >/dev/null 2>&1; then
  echo "DSpark drafter not in cache — downloading ${DRAFT_ID} (~2.7 GB)"
  docker run --rm --network host \
    -e HF_HOME=/root/.cache/huggingface -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    --entrypoint python3 "${IMAGE}" \
    -c "from huggingface_hub import snapshot_download; snapshot_download('${DRAFT_ID}')"
fi

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
