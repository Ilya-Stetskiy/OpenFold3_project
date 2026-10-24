#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 [--dry-run] <bundle_path> <user@host> <remote_dir>"
  exit 2
fi

BUNDLE_PATH="$1"
REMOTE_HOST="$2"
REMOTE_DIR="$3"

if [[ ! -e "${BUNDLE_PATH}" ]]; then
  echo "Bundle path does not exist: ${BUNDLE_PATH}"
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run:"
  echo "rsync -av ${BUNDLE_PATH} ${REMOTE_HOST}:${REMOTE_DIR}/"
  exit 0
fi

rsync -av "${BUNDLE_PATH}" "${REMOTE_HOST}:${REMOTE_DIR}/"
