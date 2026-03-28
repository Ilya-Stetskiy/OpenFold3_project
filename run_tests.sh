#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_TAG="$(python3 -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")')"
PLATFORM_TAG="$(python3 -c 'import platform; print(platform.system().lower())')"
DEPS_DIR="${PROJECT_ROOT}/.test_deps/${PLATFORM_TAG}_${PY_TAG}"
HELPERS_DIR="${PROJECT_ROOT}/helpers"
TESTS_DIR="${PROJECT_ROOT}/tests"
REQUIREMENTS_PATH="${PROJECT_ROOT}/requirements-test.txt"

mkdir -p "${DEPS_DIR}"

if ! python3 -c "import coverage, pandas, pytest" >/dev/null 2>&1 && [[ ! -d "${DEPS_DIR}/pytest" || ! -d "${DEPS_DIR}/pandas" || ! -d "${DEPS_DIR}/coverage" ]]; then
  python3 -m pip install --target "${DEPS_DIR}" -r "${REQUIREMENTS_PATH}"
fi

PYTHONPATH="${DEPS_DIR}:${HELPERS_DIR}" python3 -m pytest "${TESTS_DIR}" --cov=of_notebook_lib --cov-report=term-missing -q
