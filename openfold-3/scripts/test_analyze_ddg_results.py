from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _load_analyze_results() -> Any:
    script_path = Path(__file__).with_name("analyze_ddg_results.py")
    spec = importlib.util.spec_from_file_location("analyze_ddg_results", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze_results


def test_analyze_results_writes_outputs_when_no_rows_are_ok(tmp_path):
    input_csv = tmp_path / "results.csv"
    output_dir = tmp_path / "analysis"
    pd.DataFrame(
        [
            {
                "status": "failed",
                "method": "esm2",
                "structure_source": "experimental",
                "unit": "kcal/mol",
                "ddg": None,
                "experimental_ddg": 1.2,
                "protein_id": "p1",
                "mutation": "A1V",
            }
        ]
    ).to_csv(input_csv, index=False)

    analyze_results = _load_analyze_results()
    analyze_results(input_csv, output_dir)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["rows"] == 1
    assert summary["ok_rows"] == 0
    assert summary["failed_rows"] == 1
    assert (output_dir / "failures.csv").exists()
    assert list(pd.read_csv(output_dir / "metrics_by_method.csv").columns) == [
        "method",
        "n",
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "sign_accuracy",
        "mean_prediction",
        "mean_experimental",
    ]
