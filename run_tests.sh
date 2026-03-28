#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_TAG="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)-$(python3 -c 'import sys; print(f"py{sys.version_info[0]}{sys.version_info[1]}")')"
DEPS_DIR="${PROJECT_ROOT}/.test_deps/${PLATFORM_TAG}"
HELPERS_DIR="${PROJECT_ROOT}/helpers"
TESTS_DIR="${PROJECT_ROOT}/tests"
REQUIREMENTS_PATH="${PROJECT_ROOT}/requirements-test.txt"

mkdir -p "${DEPS_DIR}"

if python3 - <<'PY'
import importlib.util
mods = ["pytest", "pandas", "coverage"]
raise SystemExit(0 if all(importlib.util.find_spec(m) is not None for m in mods) else 1)
PY
then
  PYTHONPATH="${HELPERS_DIR}" python3 -m pytest "${TESTS_DIR}" --cov=of_notebook_lib --cov-report=term-missing -q
  exit $?
fi

PYTHONPATH="${DEPS_DIR}:${HELPERS_DIR}" python3 -m pytest "${TESTS_DIR}" --cov=of_notebook_lib --cov-report=term-missing -q
