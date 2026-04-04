from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = WORKSPACE_ROOT / "openfold3_length_benchmark"
OPENFOLD_NOTEBOOK_HELPERS = WORKSPACE_ROOT / "helpers"
OPENFOLD_REPO_DIR = WORKSPACE_ROOT.parent / "openfold-3"


def ensure_project_paths() -> None:
    for path in (OPENFOLD_NOTEBOOK_HELPERS, OPENFOLD_REPO_DIR):
        if not path.exists():
            continue
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


ensure_project_paths()

from of_notebook_lib.config import RuntimeConfig  # noqa: E402
from of_notebook_lib.runner import RunResult, run_prediction  # noqa: E402


def clone_runtime(runtime: RuntimeConfig, **overrides) -> RuntimeConfig:
    return replace(runtime, **overrides)


def benchmark_cli_path() -> Path:
    return OPENFOLD_REPO_DIR / "scripts" / "dev" / "benchmark_rmsd_vs_pdb.py"


def default_runs_root() -> Path:
    return PACKAGE_ROOT / "runs"


def default_mmcif_cache_dir() -> Path:
    return PACKAGE_ROOT / "cache" / "mmcif"
