#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${1:-$REPO_ROOT/runtime_smoke/nightly_suite}"

cd "$REPO_ROOT"

python scripts/dev/run_nightly_test_suite.py \
  --output-root "$OUTPUT_ROOT" \
  --include-wt
