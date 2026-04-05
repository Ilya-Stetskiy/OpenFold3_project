from __future__ import annotations

from pathlib import Path

from openfold3_length_benchmark.composition import (
    collect_entry_compositions,
    compositions_to_dataframe,
    extract_entry_composition,
    parse_pdb_ids,
    preview_entries,
)
from openfold3_length_benchmark.interop import (
    RuntimeConfig,
    clone_runtime,
    resolve_openfold_repo_dir,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = WORKSPACE_ROOT / "openfold3_runtime_benchmark"


def default_runs_root() -> Path:
    return PACKAGE_ROOT / "runs"


__all__ = [
    "PACKAGE_ROOT",
    "RuntimeConfig",
    "WORKSPACE_ROOT",
    "clone_runtime",
    "collect_entry_compositions",
    "compositions_to_dataframe",
    "default_runs_root",
    "extract_entry_composition",
    "parse_pdb_ids",
    "preview_entries",
    "resolve_openfold_repo_dir",
]
