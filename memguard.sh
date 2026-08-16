#!/usr/bin/env bash
# Runtime memory guard for the GB10's unified memory. Started automatically
# by start.sh / start-sglang.sh; stopped by stop.sh.
#
# Why: a memory spiral on the GB10 starves the host until the whole machine
# hard-freezes (observed on this box). Container --memory caps do not help,
# because GPU/unified allocations are not accounted to the container cgroup.
# This guard watches MemAvailable and force-removes the engine container
# before the host goes under. Losing the server beats losing the machine.
#
#   ./memguard.sh <container-name>
#
# Kills the container when MemAvailable stays below MIN_AVAILABLE_GIB for two
# consecutive samples (2s apart). Exits by itself once the container is gone.
set -u
cd "$(dirname "$0")"

CONTAINER="${1:?usage: memguard.sh <container-name>}"
MIN_AVAILABLE_GIB="${MIN_AVAILABLE_GIB:-4}"
LOG_FILE=".memguard.log"

log() { echo "[$(date -Is)] $*" >> "${LOG_FILE}"; }

log "memguard started for ${CONTAINER} (threshold ${MIN_AVAILABLE_GIB} GiB)"
strikes=0
while docker ps --format '{{.Names}}' 2>/dev/null | grep -qxF "${CONTAINER}"; do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "${avail}" -lt "${MIN_AVAILABLE_GIB}" ]; then
    strikes=$((strikes + 1))
    log "low memory: ${avail} GiB available (strike ${strikes}/2)"
    if [ "${strikes}" -ge 2 ]; then
      log "KILLING ${CONTAINER} to protect the host (${avail} GiB available)"
      docker rm -f "${CONTAINER}" >/dev/null 2>&1
      log "container removed; restart it with the matching start script"
      exit 0
    fi
  else
    strikes=0
  fi
  sleep 2
done
log "memguard exiting: ${CONTAINER} is no longer running"
