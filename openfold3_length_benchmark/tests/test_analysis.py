from __future__ import annotations

from pathlib import Path

import pandas as pd

from openfold3_length_benchmark.analysis import (
    build_binned_summary,
    select_model_row,
    select_oracle_row,
    summarize_results,
)
from openfold3_length_benchmark.plots import write_binned_svg, write_scatter_svg


def test_select_model_row_prefers_highest_ranking_score() -> None:
    rmsd_df = pd.DataFrame(
        [
            {
                "sample": "sample_low_rmsd",
                "rmsd_after_superposition": 1.2,
                "sample_ranking_score": 0.4,
                "avg_plddt": 90.0,
            },
            {
                "sample": "sample_best_ranked",
                "rmsd_after_superposition": 2.5,
                "sample_ranking_score": 0.9,
                "avg_plddt": 75.0,
            },
        ]
    )

    selected = select_model_row(rmsd_df)
    oracle = select_oracle_row(rmsd_df)

    assert selected is not None
    assert oracle is not None
    assert selected["sample"] == "sample_best_ranked"
    assert oracle["sample"] == "sample_low_rmsd"


def test_summary_and_svg_generation(tmp_path: Path) -> None:
    results_df = pd.DataFrame(
        [
            {
                "pdb_id": "2CRB",
                "status": "ok",
                "total_protein_length": 97,
                "model_selected_rmsd": 1.5,
                "oracle_best_rmsd": 1.2,
                "sample_ranking_score": 0.92,
                "avg_plddt": 88.0,
            },
            {
                "pdb_id": "5KC1",
                "status": "ok",
                "total_protein_length": 2712,
                "model_selected_rmsd": 4.0,
                "oracle_best_rmsd": 3.5,
                "sample_ranking_score": 0.81,
                "avg_plddt": 71.0,
            },
            {
                "pdb_id": "1PSM",
                "status": "ok",
                "total_protein_length": 38,
                "model_selected_rmsd": 0.9,
                "oracle_best_rmsd": 0.9,
                "sample_ranking_score": 0.95,
                "avg_plddt": 91.0,
            },
            {
                "pdb_id": "4ZEY",
                "status": "failed",
                "total_protein_length": 84,
                "model_selected_rmsd": None,
                "oracle_best_rmsd": None,
                "sample_ranking_score": None,
                "avg_plddt": None,
            },
        ]
    )
    preview_df = results_df[["pdb_id", "status"]].copy()
    failures_df = results_df[results_df["status"] == "failed"].assign(
        failure_reason="Synthetic failure"
    )

    summary = summarize_results(results_df, preview_df, failures_df)
    binned_df = build_binned_summary(results_df)
    scatter_path = write_scatter_svg(
        results_df,
        output_path=tmp_path / "scatter.svg",
        regression=summary.get("linear_regression"),
    )
    binned_path = write_binned_svg(
        binned_df,
        output_path=tmp_path / "binned.svg",
    )

    assert summary["n_successful"] == 3
    assert summary["n_failed"] == 1
    assert scatter_path.exists()
    assert binned_path.exists()
    assert "<svg" in scatter_path.read_text(encoding="utf-8")
    assert "<svg" in binned_path.read_text(encoding="utf-8")
