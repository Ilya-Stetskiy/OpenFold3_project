from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def _default_project_dir() -> Path:
    markers = ("openfold-3", "openfold_notebooks")
    for start in (Path.cwd(), Path(__file__).resolve()):
        for candidate in (start, *start.parents):
            if all((candidate / marker).exists() for marker in markers):
                return candidate
    return Path.cwd()


def _normalize_path(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return (Path.cwd() / expanded).resolve()


@dataclass(slots=True)
class RuntimeConfig:
    project_dir: Path = field(
        default_factory=lambda: _path_from_env(
            "OPENFOLD_PROJECT_DIR",
            str(_default_project_dir()),
        )
    )
    openfold_repo_dir: Path | None = None
    openfold_prefix: Path | None = None
    results_dir: Path | None = None
    msa_cache_dir: Path | None = None
    triton_cache_dir: Path | None = None
    fixed_msa_tmp_dir: Path | None = None
    msa_tmp_mode: str = os.environ.get("OPENFOLD_MSA_TMP_MODE", "symlink")
    use_fused_attention: bool = False
    use_deepspeed: bool = False

    def __post_init__(self) -> None:
        self.project_dir = _normalize_path(self.project_dir)
        self.openfold_repo_dir = _normalize_path(
            self.openfold_repo_dir
            or _path_from_env(
                "OPENFOLD_REPO_DIR",
                str(self.project_dir / "openfold-3"),
            )
        )
        self.openfold_prefix = _normalize_path(
            self.openfold_prefix
            or _path_from_env(
                "OPENFOLD_PREFIX",
                str(self.project_dir / ".venv"),
            )
        )
        self.results_dir = _normalize_path(
            self.results_dir
            or _path_from_env(
                "OPENFOLD_RESULTS_DIR",
                str(self.project_dir / "results"),
            )
        )
        self.msa_cache_dir = _normalize_path(
            self.msa_cache_dir
            or _path_from_env(
                "OPENFOLD_MSA_CACHE_DIR",
                str(self.project_dir / "msa_cache" / "colabfold_msas"),
            )
        )
        self.triton_cache_dir = _normalize_path(
            self.triton_cache_dir
            or _path_from_env(
                "OPENFOLD_TRITON_CACHE_DIR",
                str(self.project_dir / ".runtime" / "triton_cache"),
            )
        )
        self.fixed_msa_tmp_dir = _normalize_path(
            self.fixed_msa_tmp_dir
            or _path_from_env(
                "OPENFOLD_FIXED_MSA_TMP_DIR",
                str(self.project_dir / ".runtime" / "of3_colabfold_msas"),
            )
        )

    @property
    def openfold_runner(self) -> Path:
        prefix_runner = self.openfold_prefix / "bin" / "run_openfold"
        if prefix_runner.exists():
            return prefix_runner
        active_env_runner = Path(sys.executable).resolve().parent / "run_openfold"
        if active_env_runner.exists():
            return active_env_runner
        path_runner = shutil.which("run_openfold")
        if path_runner is not None:
            return Path(path_runner).resolve()
        return prefix_runner

    @property
    def openfold_python(self) -> Path:
        candidate = self.openfold_prefix / "bin" / "python"
        if candidate.exists():
            return candidate
        return Path(sys.executable)

    def build_env(self) -> dict[str, str]:
        self.triton_cache_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        prefix_bin = self.openfold_prefix / "bin"
        if prefix_bin.exists():
            env["PATH"] = f"{prefix_bin}:{env.get('PATH', '')}"

        prefix_libs = [
            str(lib_dir)
            for lib_dir in (self.openfold_prefix / "lib", self.openfold_prefix / "lib64")
            if lib_dir.exists()
        ]
        if prefix_libs:
            env["LD_LIBRARY_PATH"] = (
                f"{':'.join(prefix_libs)}:{env.get('LD_LIBRARY_PATH', '')}"
            )
            env["CUDA_HOME"] = str(self.openfold_prefix)
        env["TRITON_CACHE_DIR"] = str(self.triton_cache_dir)
        env["PYTHONUNBUFFERED"] = "1"
        env["OPENFOLD_USE_FUSED_ATTENTION"] = "1" if self.use_fused_attention else "0"
        env["OPENFOLD_USE_DEEPSPEED"] = "1" if self.use_deepspeed else "0"
        return env
