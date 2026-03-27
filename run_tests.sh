#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_DIR="${PROJECT_ROOT}/.test_deps"
HELPERS_DIR="${PROJECT_ROOT}/helpers"
TESTS_DIR="${PROJECT_ROOT}/tests"
REQUIREMENTS_PATH="${PROJECT_ROOT}/requirements-test.txt"

mkdir -p "${DEPS_DIR}"

if [[ ! -d "${DEPS_DIR}/pytest" || ! -d "${DEPS_DIR}/pandas" || ! -d "${DEPS_DIR}/pytest_cov" ]]; then
  python3 -m pip install --target "${DEPS_DIR}" -r "${REQUIREMENTS_PATH}"
fi

PYTHONPATH="${DEPS_DIR}:${HELPERS_DIR}" python3 -m pytest "${TESTS_DIR}" --cov=of_notebook_lib --cov-report=term-missing -q
