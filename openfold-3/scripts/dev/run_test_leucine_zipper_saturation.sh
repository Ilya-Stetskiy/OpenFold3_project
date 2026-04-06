#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${1:-$REPO_ROOT/runtime_smoke/test_leucine_zipper_saturation}"
RUNNER_YAML="${RUNNER_YAML:-$REPO_ROOT/examples/example_runner_yamls/low_mem.yml}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-2.0}"
NUM_DIFFUSION_SAMPLES="${NUM_DIFFUSION_SAMPLES:-4}"
NUM_MODEL_SEEDS="${NUM_MODEL_SEEDS:-1}"
MAX_MUTATIONS="${MAX_MUTATIONS:-30}"
NUM_CPU_WORKERS="${NUM_CPU_WORKERS:-4}"
MAX_INFLIGHT_QUERIES="${MAX_INFLIGHT_QUERIES:-2}"

mkdir -p "$OUTPUT_ROOT"

cd "$REPO_ROOT"

python scripts/dev/run_overnight_screening.py \
  --base-query-json "$REPO_ROOT/examples/example_inference_inputs/query_test_leucine_zipper.json" \
  --query-id test_leucine_zipper \
  --mutations-csv "$REPO_ROOT/examples/example_screening_jobs/test_leucine_zipper_chainA_saturation.csv" \
  --output-root "$OUTPUT_ROOT" \
  --runner-yaml "$RUNNER_YAML" \
  --num-diffusion-samples "$NUM_DIFFUSION_SAMPLES" \
  --num-model-seeds "$NUM_MODEL_SEEDS" \
  --max-mutations "$MAX_MUTATIONS" \
  --num-cpu-workers "$NUM_CPU_WORKERS" \
  --max-inflight-queries "$MAX_INFLIGHT_QUERIES" \
  --min-free-disk-gb "$MIN_FREE_DISK_GB" \
  --include-wt
