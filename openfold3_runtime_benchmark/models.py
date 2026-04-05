from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class RuntimeBenchmarkRunResult:
    run_name: str
    run_root: Path
    preview_df: pd.DataFrame
    case_results_df: pd.DataFrame
    failures_df: pd.DataFrame
    summary: dict[str, object]
    summary_path: Path
    manifest_path: Path
    case_results_csv_path: Path
    case_results_json_path: Path
    events_path: Path
    samples_path: Path
    plot_paths: dict[str, Path]
