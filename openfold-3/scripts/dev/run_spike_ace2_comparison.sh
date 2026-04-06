#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${1:-$REPO_ROOT/runtime_smoke/spike_ace2_compare}"
RUNNER_YAML="${RUNNER_YAML:-$REPO_ROOT/examples/example_runner_yamls/low_mem.yml}"
NUM_DIFFUSION_SAMPLES="${NUM_DIFFUSION_SAMPLES:-1}"
NUM_MODEL_SEEDS="${NUM_MODEL_SEEDS:-1}"

mkdir -p "$OUTPUT_ROOT"

cd "$REPO_ROOT"

python scripts/dev/run_single_protein_comparison.py \
  --query-json "$REPO_ROOT/examples/example_inference_inputs/query_spike_ace2_full.json" \
  --query-id spike_ace2_full \
  --runner-yaml "$RUNNER_YAML" \
  --output-root "$OUTPUT_ROOT" \
  --num-diffusion-samples "$NUM_DIFFUSION_SAMPLES" \
  --num-model-seeds "$NUM_MODEL_SEEDS"
