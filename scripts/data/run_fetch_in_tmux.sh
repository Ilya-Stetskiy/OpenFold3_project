#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
SESSION_NAME="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import yaml

path = Path("configs/downloads.yaml")
data = yaml.safe_load(path.read_text()) if path.exists() else {}
print(((data or {}).get("background") or {}).get("tmux_session_name", "ddg_fetch"))
PY
)"
LOG_DIR="logs/data_fetching"
LOG_FILE="${LOG_DIR}/fetch_latest.log"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists."
  echo "tmux attach -t ${SESSION_NAME}"
  echo "tail -f ${LOG_FILE}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "${PYTHON_BIN} scripts/data/fetch_dataset_sources.py --dataset all --workers 8 --retries 5 --no-progress-bar 2>&1 | tee ${LOG_FILE}"

echo "Started tmux session: ${SESSION_NAME}"
echo "tmux attach -t ${SESSION_NAME}"
echo "tail -f ${LOG_FILE}"
