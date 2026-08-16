#!/usr/bin/env bash
set -euo pipefail

# Stop whichever engine container is running (vLLM from start.sh, SGLang from
# start-sglang.sh). Stopped containers are left in place for `docker logs`
# post-mortem; the next start script removes them.

# Anchor to the repo dir so the pid/log files are found from any cwd.
cd "$(dirname "$0")"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

stopped=0
stop_one() {
  local name="$1" pid_file="$2" log_file="$3"
  if ! docker ps -a --format '{{.Names}}' | grep -qxF "${name}"; then
    rm -f "${pid_file}"
    return
  fi
  if docker ps --format '{{.Names}}' | grep -qxF "${name}"; then
    echo "Stopping container ${name}..."
    docker stop "${name}" >/dev/null
    echo "[$(date -Is)] container stopped" >> "${log_file}"
    echo "Stopped ${name}."
    stopped=1
  fi
  rm -f "${pid_file}"
}

stop_one "qwen3.8-27b-nvfp4" ".vllm.pid" ".vllm.log"
stop_one "qwen3.8-27b-sglang" ".sglang.pid" ".sglang.log"

# Stop the runtime memory guard (it also exits by itself once its container
# is gone, but clean up proactively).
if [[ -f .memguard.pid ]]; then
  kill "$(cat .memguard.pid)" 2>/dev/null || true
  rm -f .memguard.pid
fi

if [[ "${stopped}" == "0" ]]; then
  echo "No engine container is running; nothing to stop"
fi
