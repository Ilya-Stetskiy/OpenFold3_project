#!/usr/bin/env bash
set -euo pipefail

SESSION="openfold3_cpu_heavy"
LOG_DIR="logs/run_profiles"
LOG_FILE="${LOG_DIR}/cpu_heavy_latest.log"
mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists."
  echo "tmux attach -t ${SESSION}"
  echo "tail -f ${LOG_FILE}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "python3 scripts/run/run_cpu_heavy_adaptive.py --dataset current_internal --dry-run --no-progress-bar 2>&1 | tee ${LOG_FILE}"
echo "Started ${SESSION}"
echo "tmux attach -t ${SESSION}"
echo "tail -f ${LOG_FILE}"
