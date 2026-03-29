#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/helpers:${PYTHONPATH:-}"

python3 "${PROJECT_ROOT}/server_smoke/run_server_end_to_end.py" "$@"
