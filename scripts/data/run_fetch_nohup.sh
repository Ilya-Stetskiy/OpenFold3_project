#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
LOG_DIR="logs/data_fetching"
LOG_FILE="${LOG_DIR}/fetch_nohup.log"

mkdir -p "${LOG_DIR}"

nohup "${PYTHON_BIN}" scripts/data/fetch_dataset_sources.py --dataset all --workers 8 --retries 5 --no-progress-bar > "${LOG_FILE}" 2>&1 &
PID="$!"

echo "Started fetch in background."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "tail -f ${LOG_FILE}"
