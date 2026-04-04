from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = WORKSPACE_ROOT / "openfold3_length_benchmark"
OPENFOLD_NOTEBOOK_HELPERS = WORKSPACE_ROOT / "helpers"


def _looks_like_openfold_repo(path: Path) -> bool:
    return (
        (path / "openfold3").is_dir()
        and (
            (path / "scripts" / "dev" / "benchmark_rmsd_vs_pdb.py").exists()
            or (path / "pyproject.toml").exists()
        )
    )


def resolve_openfold_repo_dir(base_dir: str | Path | None = None) -> Path:
    anchor = Path(base_dir).expanduser().resolve() if base_dir is not None else WORKSPACE_ROOT
    candidates = [
        anchor / "openfold-3",
        anchor.parent / "openfold-3",
        anchor,
        anchor.parent,
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _looks_like_openfold_repo(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate an openfold-3 checkout. "
        f"Tried: {', '.join(str(path) for path in seen)}"
    )


def default_openfold_repo_dir() -> Path:
    try:
        return resolve_openfold_repo_dir()
    except FileNotFoundError:
        return WORKSPACE_ROOT


OPENFOLD_REPO_DIR = default_openfold_repo_dir()


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


def benchmark_cli_path(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir).expanduser().resolve() / "scripts" / "dev" / "benchmark_rmsd_vs_pdb.py"
    return resolve_openfold_repo_dir() / "scripts" / "dev" / "benchmark_rmsd_vs_pdb.py"


def default_runs_root() -> Path:
    return PACKAGE_ROOT / "runs"


def default_mmcif_cache_dir() -> Path:
    return PACKAGE_ROOT / "cache" / "mmcif"
