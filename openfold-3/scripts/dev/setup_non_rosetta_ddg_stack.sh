#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

CONDA_BIN_DEFAULT="${HOME}/miniconda3/bin/conda"
CONDA_BIN="${CONDA_BIN:-${CONDA_BIN_DEFAULT}}"
CREATE_ENVS="${CREATE_ENVS:-1}"
CPU_ONLY_TORCH_INDEX_URL="${CPU_ONLY_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

FOLDX_BINARY_DEFAULT="${PROJECT_ROOT}/tools/bin/foldx"
SAAMBE_ENV_NAME="${SAAMBE_ENV_NAME:-py311_saambe3d}"
HELIXON_ENV_NAME="${HELIXON_ENV_NAME:-ddg-predict}"
PROMPT_ENV_NAME="${PROMPT_ENV_NAME:-prompt-ddg}"

SAAMBE_ROOT="${PROJECT_ROOT}/third_party/SAAMBE-3D"
HELIXON_ROOT="${PROJECT_ROOT}/third_party/binding-ddg-predictor"
PROMPT_ROOT="${PROJECT_ROOT}/third_party/Prompt-DDG"

SAAMBE_PYTHON_DEFAULT="${HOME}/miniconda3/envs/${SAAMBE_ENV_NAME}/bin/python"
HELIXON_PYTHON_DEFAULT="${HOME}/miniconda3/envs/${HELIXON_ENV_NAME}/bin/python"
PROMPT_PYTHON_DEFAULT="${HOME}/miniconda3/envs/${PROMPT_ENV_NAME}/bin/python"

mkdir -p "${PROJECT_ROOT}/.deps/helixon" "${PROJECT_ROOT}/.deps/promptddg"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    return 1
  fi
}

maybe_create_conda_envs() {
  if [[ "${CREATE_ENVS}" != "1" ]]; then
    echo "Skipping conda env creation because CREATE_ENVS=${CREATE_ENVS}"
    return 0
  fi

  if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "conda not found at ${CONDA_BIN}" >&2
    echo "Set CONDA_BIN=/path/to/conda or CREATE_ENVS=0 if envs already exist." >&2
    return 1
  fi

  echo "Creating/updating SAAMBE-3D env: ${SAAMBE_ENV_NAME}"
  "${CONDA_BIN}" env update -n "${SAAMBE_ENV_NAME}" -f "${SAAMBE_ROOT}/environment.yml" --prune || \
    "${CONDA_BIN}" env create -n "${SAAMBE_ENV_NAME}" -f "${SAAMBE_ROOT}/environment.yml"

  echo "Creating/updating HeliXon env: ${HELIXON_ENV_NAME}"
  "${CONDA_BIN}" create -y -n "${HELIXON_ENV_NAME}" python=3.8 || true
  "${CONDA_BIN}" run -n "${HELIXON_ENV_NAME}" python -m pip install --upgrade pip
  "${CONDA_BIN}" run -n "${HELIXON_ENV_NAME}" python -m pip install \
    "torch==1.10.2" "biopython==1.79" "easydict" "numpy<2"

  echo "Creating/updating Prompt-DDG env: ${PROMPT_ENV_NAME}"
  "${CONDA_BIN}" create -y -n "${PROMPT_ENV_NAME}" python=3.8 || true
  "${CONDA_BIN}" run -n "${PROMPT_ENV_NAME}" python -m pip install --upgrade pip
  "${CONDA_BIN}" run -n "${PROMPT_ENV_NAME}" python -m pip install \
    --index-url "${CPU_ONLY_TORCH_INDEX_URL}" \
    "torch==1.10.2" "pandas==1.2.4" "numpy==1.20.1" "tqdm==4.59.0" \
    "biopython==1.79" "scikit-learn==0.24.1" "easydict==1.9"
}

write_env_file() {
  local env_file="${PROJECT_ROOT}/.runtime_codex/non_rosetta_ddg_env.sh"
  mkdir -p "$(dirname "${env_file}")"

  local foldx_binary="${FOLDX_BINARY:-${FOLDX_BINARY_DEFAULT}}"
  local saambe_python="${SAAMBE_3D_PYTHON:-${SAAMBE_PYTHON_DEFAULT}}"
  local helixon_python="${HELIXON_BINDING_DDG_PYTHON:-${HELIXON_PYTHON_DEFAULT}}"
  local prompt_python="${PROMPT_DDG_PYTHON:-${PROMPT_PYTHON_DEFAULT}}"

  cat > "${env_file}" <<EOF
#!/usr/bin/env bash
export FOLDX_BINARY="${foldx_binary}"
export SAAMBE_3D_PYTHON="${saambe_python}"
export SAAMBE_3D_SCRIPT="${SAAMBE_ROOT}/saambe-3d.py"
export SAAMBE_3D_PYTHONPATH="${PROJECT_ROOT}/.deps/saambe3d"

export HELIXON_BINDING_DDG_PYTHON="${helixon_python}"
export HELIXON_BINDING_DDG_SCRIPT="${HELIXON_ROOT}/scripts/predict.py"
export HELIXON_BINDING_DDG_MODEL="${HELIXON_ROOT}/data/model.pt"
export HELIXON_BINDING_DDG_PYTHONPATH="${PROJECT_ROOT}/.deps/helixon"

export PROMPT_DDG_PYTHON="${prompt_python}"
export PROMPT_DDG_INFER_SCRIPT="${REPO_ROOT}/scripts/dev/prompt_ddg_infer.py"
export PROMPT_DDG_CHECKPOINT="${PROMPT_ROOT}/trained_models/ddg_model.ckpt"
EOF

  chmod +x "${env_file}"
  echo "Wrote env file: ${env_file}"
}

print_status() {
  local foldx_binary="${FOLDX_BINARY:-${FOLDX_BINARY_DEFAULT}}"
  local saambe_script="${SAAMBE_ROOT}/saambe-3d.py"
  local helixon_script="${HELIXON_ROOT}/scripts/predict.py"
  local helixon_model="${HELIXON_ROOT}/data/model.pt"
  local prompt_wrapper="${REPO_ROOT}/scripts/dev/prompt_ddg_infer.py"
  local prompt_ckpt="${PROMPT_ROOT}/trained_models/ddg_model.ckpt"

  echo
  echo "Status summary"
  echo "  FoldX binary: ${foldx_binary}"
  echo "  SAAMBE-3D script: ${saambe_script}"
  echo "  HeliXon script: ${helixon_script}"
  echo "  HeliXon model: ${helixon_model}"
  echo "  Prompt-DDG wrapper: ${prompt_wrapper}"
  echo "  Prompt-DDG checkpoint: ${prompt_ckpt}"
  echo

  local missing=0
  require_path "${foldx_binary}" "FoldX binary" || missing=1
  require_path "${saambe_script}" "SAAMBE-3D script" || missing=1
  require_path "${helixon_script}" "HeliXon predictor script" || missing=1
  require_path "${prompt_wrapper}" "Prompt-DDG wrapper" || missing=1

  if [[ ! -f "${helixon_model}" ]]; then
    echo "Missing HeliXon model weights: ${helixon_model}" >&2
    echo "Place the predictor checkpoint there or set HELIXON_BINDING_DDG_MODEL manually." >&2
    missing=1
  fi
  if [[ ! -f "${prompt_ckpt}" ]]; then
    echo "Missing Prompt-DDG checkpoint: ${prompt_ckpt}" >&2
    echo "Download ddg_model.ckpt into ${PROMPT_ROOT}/trained_models/ or set PROMPT_DDG_CHECKPOINT manually." >&2
    missing=1
  fi

  if [[ "${missing}" -ne 0 ]]; then
    echo
    echo "Non-Rosetta stack is not fully ready yet." >&2
    return 1
  fi

  echo "Non-Rosetta stack looks ready."
}

print_how_to_use() {
  local env_file="${PROJECT_ROOT}/.runtime_codex/non_rosetta_ddg_env.sh"
  cat <<EOF

Next steps on the server:
1. source "${env_file}"
2. run your analysis process from "${PROJECT_ROOT}"

Example smoke call:
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'openfold-3')
from openfold3.benchmark.mutation_analysis import run_mutation_analysis
from openfold3.benchmark.models import MutationInput

result = run_mutation_analysis(
    structure_path=Path('PATH_TO_INPUT_STRUCTURE'),
    mutation=MutationInput('A', 'L', 1, 'A'),
    methods=['foldx', 'saambe_3d', 'helixon_binding_ddg', 'prompt_ddg'],
    output_dir=Path('.runtime_codex/manual_multiscale_run'),
)
print(result.result_json_path)
print(result.report['flat_results'])
PY
EOF
}

main() {
  maybe_create_conda_envs
  write_env_file
  print_status
  print_how_to_use
}

main "$@"
