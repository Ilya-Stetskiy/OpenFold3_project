#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUERY_JSON="${PROJECT_ROOT}/server_smoke/query_ubiquitin.json"
DEFAULT_OUTPUT_DIR="${PROJECT_ROOT}/server_smoke/_runtime_output"

RUNNER="${OPENFOLD_RUNNER:-run_openfold}"
OUTPUT_DIR="${1:-${DEFAULT_OUTPUT_DIR}}"

if ! command -v "${RUNNER}" >/dev/null 2>&1; then
  echo "ERROR: OpenFold runner not found: ${RUNNER}" >&2
  echo "Set OPENFOLD_RUNNER=/absolute/path/to/run_openfold or add it to PATH." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

echo "Running OpenFold smoke test"
echo "runner     : ${RUNNER}"
echo "query_json : ${QUERY_JSON}"
echo "output_dir : ${OUTPUT_DIR}"

"${RUNNER}" predict \
  --query_json="${QUERY_JSON}" \
  --output_dir="${OUTPUT_DIR}" \
  --use_templates=false \
  --use_msa_server=true \
  --num_diffusion_samples=1 \
  --num_model_seeds=1

AGG_COUNT="$(find "${OUTPUT_DIR}" -type f -name '*_confidences_aggregated.json' | wc -l | tr -d ' ')"
MODEL_COUNT="$(find "${OUTPUT_DIR}" -type f \( -name '*_model.cif' -o -name '*_model.pdb' \) | wc -l | tr -d ' ')"

if [[ "${AGG_COUNT}" -lt 1 ]]; then
  echo "ERROR: no *_confidences_aggregated.json files were produced" >&2
  exit 3
fi

if [[ "${MODEL_COUNT}" -lt 1 ]]; then
  echo "ERROR: no model files (*.cif or *.pdb) were produced" >&2
  exit 4
fi

echo "Smoke test passed"
echo "aggregated_json_files=${AGG_COUNT}"
echo "model_files=${MODEL_COUNT}"
echo "output_dir=${OUTPUT_DIR}"
