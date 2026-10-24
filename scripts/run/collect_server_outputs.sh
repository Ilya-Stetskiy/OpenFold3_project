#!/usr/bin/env bash
set -euo pipefail

FORCE=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 [--dry-run] [--force] <user@host:remote_project> <local_output_dir>"
  exit 2
fi

REMOTE_PROJECT="$1"
LOCAL_DIR="$2"
mkdir -p "${LOCAL_DIR}"

if [[ "${FORCE}" != "1" && -n "$(find "${LOCAL_DIR}" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]]; then
  echo "Local output dir is not empty; use --force to collect into it: ${LOCAL_DIR}"
  exit 1
fi

SOURCES=(
  "results"
  "reports"
  "logs"
  "data/run_state"
  "data/processed/current_internal"
)

for source in "${SOURCES[@]}"; do
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "rsync -av ${REMOTE_PROJECT}/${source}/ ${LOCAL_DIR}/${source}/"
  else
    mkdir -p "${LOCAL_DIR}/${source}"
    rsync -av "${REMOTE_PROJECT}/${source}/" "${LOCAL_DIR}/${source}/"
  fi
done
