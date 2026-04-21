#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

BOOTSTRAP_DIR="${PROJECT_ROOT}/.runtime_codex/bootstrap"
VENV_DIR="${PROJECT_ROOT}/.runtime_codex/venvs/saambe3d"
SAAMBE_ROOT="${PROJECT_ROOT}/third_party/SAAMBE-3D"

PYTHON_BIN="${PYTHON_BIN:-python3}"

log() {
  printf '%s\n' "$*"
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    printf 'Missing %s: %s\n' "${label}" "${path}" >&2
    exit 1
  fi
}

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'Command not found: %s\n' "${command_name}" >&2
    exit 1
  fi
}

bootstrap_virtualenv() {
  mkdir -p "${BOOTSTRAP_DIR}"
  "${PYTHON_BIN}" -m pip install --break-system-packages --target "${BOOTSTRAP_DIR}" virtualenv
}

create_env() {
  PYTHONPATH="${BOOTSTRAP_DIR}" "${PYTHON_BIN}" -m virtualenv "${VENV_DIR}"
}

install_deps() {
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${VENV_DIR}/bin/python" -m pip install \
    "numpy==2.4.4" \
    "scipy==1.17.1" \
    "scikit-learn==1.5.2" \
    "xgboost==2.1.1" \
    "biopython==1.87" \
    "prody==2.6.1" \
    "pyparsing==3.1.1" \
    "requests" \
    "matplotlib"
}

write_env_file() {
  local env_file="${PROJECT_ROOT}/.runtime_codex/saambe3d_env.sh"
  mkdir -p "$(dirname "${env_file}")"
  cat > "${env_file}" <<EOF
#!/usr/bin/env bash
export SAAMBE_3D_PYTHON="${VENV_DIR}/bin/python"
export SAAMBE_3D_SCRIPT="${SAAMBE_ROOT}/saambe-3d.py"
EOF
  chmod +x "${env_file}"
  log "Wrote env file: ${env_file}"
}

print_usage() {
  cat <<EOF

SAAMBE-3D setup complete.

Use it in the current shell:
source "${PROJECT_ROOT}/.runtime_codex/saambe3d_env.sh"

Smoke example:
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'openfold-3')
from openfold3.benchmark.mutation_analysis import run_mutation_analysis
from openfold3.benchmark.models import MutationInput

result = run_mutation_analysis(
    structure_path=Path("PATH_TO_STRUCTURE"),
    mutation=MutationInput("A", "L", 1, "A"),
    methods=["saambe_3d"],
    output_dir=Path(".runtime_codex/manual_saambe_run"),
)
print(result.result_json_path)
print(result.report["flat_results"])
PY
EOF
}

main() {
  require_path "${SAAMBE_ROOT}/saambe-3d.py" "SAAMBE-3D script"
  require_command "${PYTHON_BIN}"
  bootstrap_virtualenv
  create_env
  install_deps
  write_env_file
  print_usage
}

main "$@"
