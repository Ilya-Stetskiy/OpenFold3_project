from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


@dataclass(slots=True)
class RuntimeConfig:
    project_dir: Path = _path_from_env("OPENFOLD_PROJECT_DIR", "/home/jovyan/OpenFold")
    openfold_repo_dir: Path = _path_from_env(
        "OPENFOLD_REPO_DIR",
        str(Path(__file__).resolve().parents[3] / "openfold-3"),
    )
    openfold_prefix: Path = _path_from_env(
        "OPENFOLD_PREFIX", "/home/jovyan/.mlspace/envs/openfold310"
    )
    results_dir: Path = _path_from_env(
        "OPENFOLD_RESULTS_DIR", "/home/jovyan/OpenFold/results_refactored"
    )
    msa_cache_dir: Path = _path_from_env(
        "OPENFOLD_MSA_CACHE_DIR", "/home/jovyan/OpenFold/msa_cache/colabfold_msas"
    )
    triton_cache_dir: Path = _path_from_env(
        "OPENFOLD_TRITON_CACHE_DIR", "/tmp/triton_cache"
    )
    fixed_msa_tmp_dir: Path = _path_from_env(
        "OPENFOLD_FIXED_MSA_TMP_DIR", "/tmp/of3_colabfold_msas"
    )
    use_fused_attention: bool = False
    use_deepspeed: bool = False

    @property
    def openfold_runner(self) -> Path:
        return self.openfold_prefix / "bin" / "run_openfold"

    @property
    def openfold_python(self) -> Path:
        candidate = self.openfold_prefix / "bin" / "python"
        if candidate.exists():
            return candidate
        return Path(sys.executable)

    def build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_HOME"] = str(self.openfold_prefix)
        env["PATH"] = f"{self.openfold_prefix / 'bin'}:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = (
            f"{self.openfold_prefix / 'lib'}:{self.openfold_prefix / 'lib64'}:"
            f"{env.get('LD_LIBRARY_PATH', '')}"
        )
        env["TRITON_CACHE_DIR"] = str(self.triton_cache_dir)
        env["PYTHONUNBUFFERED"] = "1"
        env["OPENFOLD_USE_FUSED_ATTENTION"] = "1" if self.use_fused_attention else "0"
        env["OPENFOLD_USE_DEEPSPEED"] = "1" if self.use_deepspeed else "0"
        return env
