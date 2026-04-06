#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash ./check_server_gpu.sh [--quick] [--with-ddg-harness] [--output-root DIR]

Runs the consolidated GPU/server verification suite for OpenFold3_project.

Options:
  --quick             Skip the longest checks (`test_kernels.py` and compare-batch smoke).
  --with-ddg-harness  Also run the ddG harness on the smoke-test output structure.
  --output-root DIR   Store logs and runtime outputs under DIR.
  -h, --help          Show this help.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENFOLD_SRC_DIR="${REPO_ROOT}/openfold-3"
NOTEBOOKS_DIR="${REPO_ROOT}/openfold_notebooks"
PYTHON_BIN="${PYTHON_BIN:-python3}"
QUICK=0
WITH_DDG_HARNESS=0
OUTPUT_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      shift
      ;;
    --with-ddg-harness)
      WITH_DDG_HARNESS=1
      shift
      ;;
    --output-root)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --output-root requires a value" >&2
        exit 2
      fi
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

timestamp="$(date +%Y%m%d_%H%M%S)"
CHECK_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/.runtime/server_checks/${timestamp}}"
LOG_DIR="${CHECK_ROOT}/logs"
BIN_DIR="${CHECK_ROOT}/bin"
SMOKE_OUTPUT_DIR="${CHECK_ROOT}/server_smoke"
mkdir -p "${LOG_DIR}" "${BIN_DIR}" "${SMOKE_OUTPUT_DIR}"

if [[ ! -d "${OPENFOLD_SRC_DIR}" ]]; then
  echo "ERROR: expected openfold-3 source tree at ${OPENFOLD_SRC_DIR}" >&2
  exit 2
fi

if [[ ! -d "${NOTEBOOKS_DIR}" ]]; then
  echo "ERROR: expected openfold_notebooks source tree at ${NOTEBOOKS_DIR}" >&2
  exit 2
fi

ACTIVE_PREFIX="$("${PYTHON_BIN}" - <<'PY'
import sys
from pathlib import Path

print(Path(sys.executable).resolve().parents[1])
PY
)"

export OPENFOLD_PROJECT_DIR="${REPO_ROOT}"
export OPENFOLD_REPO_DIR="${OPENFOLD_SRC_DIR}"
export OPENFOLD_PREFIX="${OPENFOLD_PREFIX:-${ACTIVE_PREFIX}}"
export OPENFOLD_RESULTS_DIR="${CHECK_ROOT}/results"
export OPENFOLD_MSA_CACHE_DIR="${CHECK_ROOT}/msa_cache/colabfold_msas"
export OPENFOLD_TRITON_CACHE_DIR="${CHECK_ROOT}/triton_cache"
export OPENFOLD_FIXED_MSA_TMP_DIR="${CHECK_ROOT}/of3_colabfold_msas"
mkdir -p \
  "${OPENFOLD_RESULTS_DIR}" \
  "${OPENFOLD_MSA_CACHE_DIR}" \
  "${OPENFOLD_TRITON_CACHE_DIR}" \
  "${OPENFOLD_FIXED_MSA_TMP_DIR}"

RUNNER_WRAPPER="${BIN_DIR}/run_openfold"
cat > "${RUNNER_WRAPPER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${OPENFOLD_SRC_DIR}\${PYTHONPATH:+:\${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m openfold3.run_openfold "\$@"
EOF
chmod +x "${RUNNER_WRAPPER}"
export PATH="${BIN_DIR}:${PATH}"
export OPENFOLD_RUNNER="${RUNNER_WRAPPER}"

run_step() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}.log"
  echo
  echo "==> ${name}"
  "$@" 2>&1 | tee "${log_path}"
}

run_step preflight_python "${PYTHON_BIN}" - <<'PY'
import importlib.util

required = [
    "biotite",
    "numpy",
    "pandas",
    "pytest",
    "pytest_cov",
    "requests",
    "torch",
]
missing = [module for module in required if importlib.util.find_spec(module) is None]
if missing:
    raise SystemExit(
        "Missing required Python modules for server check: " + ", ".join(missing)
    )
print("Python environment looks ready")
PY

run_step repo_state git -C "${REPO_ROOT}" log --oneline -1 --decorate
run_step gpu_probe nvidia-smi

run_step torch_cuda "${PYTHON_BIN}" - <<'PY'
import torch

print("cuda_available =", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
print("device_count   =", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(f"device_{index}      = {torch.cuda.get_device_name(index)}")
PY

ccd_args=()
if ! "${PYTHON_BIN}" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("boto3") is not None else 1)
PY
then
  ccd_args+=(--skip-ccd-update)
fi

run_step openfold_mutation_tests env \
  PYTHONPATH="${OPENFOLD_SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m pytest \
  "${OPENFOLD_SRC_DIR}/openfold3/tests/test_mutation_runner.py" \
  "${OPENFOLD_SRC_DIR}/openfold3/tests/test_mutation_runner_cpu_benchmark_and_comparison.py" \
  "${OPENFOLD_SRC_DIR}/openfold3/tests/test_mutation_runner_cpu_integration.py" \
  -q \
  "${ccd_args[@]}"

run_step openfold_semantic_cuda env \
  PYTHONPATH="${OPENFOLD_SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m pytest \
  "${OPENFOLD_SRC_DIR}/semantic_tests/test_core_model_semantic_numerical.py" \
  -q -rs

if [[ "${QUICK}" -eq 0 ]]; then
  run_step openfold_kernel_tests env \
    PYTHONPATH="${OPENFOLD_SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m pytest \
    "${OPENFOLD_SRC_DIR}/openfold3/tests/test_kernels.py" \
    -q -rs
else
  echo
  echo "==> openfold_kernel_tests"
  echo "Skipped because --quick was requested" | tee "${LOG_DIR}/openfold_kernel_tests.log"
fi

run_step notebook_tests bash "${NOTEBOOKS_DIR}/run_tests.sh"
run_step server_smoke bash "${NOTEBOOKS_DIR}/run_server_smoke.sh" "${SMOKE_OUTPUT_DIR}"
run_step server_end_to_end bash "${NOTEBOOKS_DIR}/run_server_end_to_end.sh" \
  --experiment-name "server_check_e2e" \
  --amino-acids AG \
  --subprocess-batch-size 1 \
  --dispatch-partial-batches \
  --batch-gather-timeout-seconds 5

if [[ "${QUICK}" -eq 0 ]]; then
  run_step compare_mutation_batch bash "${NOTEBOOKS_DIR}/run_compare_mutation_batch.sh" \
    --experiment-name "server_check_compare" \
    --amino-acids AG \
    --exclude-wt \
    --subprocess-batch-size 1 \
    --dispatch-partial-batches \
    --batch-gather-timeout-seconds 5
else
  echo
  echo "==> compare_mutation_batch"
  echo "Skipped because --quick was requested" | tee "${LOG_DIR}/compare_mutation_batch.log"
fi

if [[ "${WITH_DDG_HARNESS}" -eq 1 ]]; then
  smoke_model_path="$(find "${SMOKE_OUTPUT_DIR}" -type f \( -name '*_model.cif' -o -name '*_model.pdb' \) | sort | head -n 1)"
  smoke_confidence_path="$(find "${SMOKE_OUTPUT_DIR}" -type f -name '*_confidences_aggregated.json' | sort | head -n 1)"
  if [[ -z "${smoke_model_path}" || -z "${smoke_confidence_path}" ]]; then
    echo "ERROR: could not locate smoke-test structure/confidence outputs for ddG harness" >&2
    exit 1
  fi
  ddg_report_path="${CHECK_ROOT}/ddg/report.json"
  run_step ddg_harness env \
    PYTHONPATH="${OPENFOLD_SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" "${OPENFOLD_SRC_DIR}/scripts/dev/run_ddg_benchmark_harness.py" \
    --case-id "ubiquitin_smoke_ddg" \
    --structure "${smoke_model_path}" \
    --confidence-json "${smoke_confidence_path}" \
    --mutation "A:Q2A" \
    --notes "Generated by check_server_gpu.sh" \
    --output "${ddg_report_path}"
  run_step ddg_harness_validate env DDG_REPORT_PATH="${ddg_report_path}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

report_path = Path(os.environ["DDG_REPORT_PATH"])
payload = json.loads(report_path.read_text(encoding="utf-8"))
results = payload.get("results", [])
if not results:
    raise SystemExit(f"No ddG results were written to {report_path}")
ok_methods = [result["method"] for result in results if result.get("status") == "ok"]
print(f"ok_methods={ok_methods}")
if not ok_methods:
    raise SystemExit(f"ddG harness produced no successful methods: {report_path}")
PY
fi

summary_path="${CHECK_ROOT}/summary.txt"
cat > "${summary_path}" <<EOF
repo_root=${REPO_ROOT}
check_root=${CHECK_ROOT}
logs_dir=${LOG_DIR}
results_dir=${OPENFOLD_RESULTS_DIR}
runner=${OPENFOLD_RUNNER}
quick=${QUICK}
with_ddg_harness=${WITH_DDG_HARNESS}
timestamp=${timestamp}
EOF

echo
echo "Server GPU check completed successfully"
echo "summary_path=${summary_path}"
echo "check_root=${CHECK_ROOT}"
