from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            "| " + " | ".join("" if row[column] is None else str(row[column]) for column in columns) + " |"
        )
    return "\n".join(lines)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def rmsd_rows_to_dataframe(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        aggregated = row.get("aggregated_confidence") or {}
        coverage = row.get("coverage") or {}
        records.append(
            {
                "query": row.get("query"),
                "seed": row.get("seed"),
                "sample": row.get("sample"),
                "pred_path": row.get("pred_path"),
                "ref_path": row.get("ref_path"),
                "rmsd_before_superposition": row.get("rmsd_before_superposition"),
                "rmsd_after_superposition": row.get("rmsd_after_superposition"),
                "matched_atom_count": coverage.get("matched_atom_count"),
                "pred_filtered_atom_count": coverage.get("pred_filtered_atom_count"),
                "ref_filtered_atom_count": coverage.get("ref_filtered_atom_count"),
                "avg_plddt": aggregated.get("avg_plddt"),
                "ptm": aggregated.get("ptm"),
                "iptm": aggregated.get("iptm"),
                "sample_ranking_score": aggregated.get("sample_ranking_score"),
            }
        )
    return pd.DataFrame.from_records(records)


def select_model_row(rmsd_df: pd.DataFrame) -> dict[str, Any] | None:
    if rmsd_df.empty:
        return None

    ordered = rmsd_df.copy()
    if ordered["sample_ranking_score"].notna().any():
        ordered = ordered.assign(
            _ranking_missing=ordered["sample_ranking_score"].isna().astype(int),
            _plddt_missing=ordered["avg_plddt"].isna().astype(int),
        ).sort_values(
            by=[
                "_ranking_missing",
                "sample_ranking_score",
                "_plddt_missing",
                "avg_plddt",
                "rmsd_after_superposition",
            ],
            ascending=[True, False, True, False, True],
        )
    else:
        ordered = ordered.sort_values(
            by=["rmsd_after_superposition", "avg_plddt"],
            ascending=[True, False],
        )
    return ordered.iloc[0].to_dict()


def select_oracle_row(rmsd_df: pd.DataFrame) -> dict[str, Any] | None:
    if rmsd_df.empty:
        return None
    ordered = rmsd_df.sort_values(
        by=["rmsd_after_superposition", "sample_ranking_score"],
        ascending=[True, False],
    )
    return ordered.iloc[0].to_dict()


def build_binned_summary(
    results_df: pd.DataFrame,
    *,
    max_bins: int = 4,
) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(
            columns=[
                "length_bin",
                "case_count",
                "min_length",
                "max_length",
                "mean_model_selected_rmsd",
                "median_model_selected_rmsd",
            ]
        )

    valid = results_df[
        (results_df["status"] == "ok")
        & results_df["total_protein_length"].notna()
        & results_df["model_selected_rmsd"].notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "length_bin",
                "case_count",
                "min_length",
                "max_length",
                "mean_model_selected_rmsd",
                "median_model_selected_rmsd",
            ]
        )

    unique_lengths = int(valid["total_protein_length"].nunique())
    bin_count = min(max_bins, len(valid), unique_lengths)
    if bin_count <= 0:
        return pd.DataFrame()

    if bin_count == 1:
        valid["_length_bin"] = "all"
    else:
        valid["_length_bin"] = pd.qcut(
            valid["total_protein_length"],
            q=bin_count,
            duplicates="drop",
        ).astype(str)

    summary = (
        valid.groupby("_length_bin", dropna=False)
        .agg(
            case_count=("pdb_id", "count"),
            min_length=("total_protein_length", "min"),
            max_length=("total_protein_length", "max"),
            mean_model_selected_rmsd=("model_selected_rmsd", "mean"),
            median_model_selected_rmsd=("model_selected_rmsd", "median"),
        )
        .reset_index()
        .rename(columns={"_length_bin": "length_bin"})
    )
    return summary


def _series_corr(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    method: str,
) -> float | None:
    if frame.empty or len(frame) < 2:
        return None
    subset = frame[[left, right]].dropna()
    if len(subset) < 2:
        return None
    value = subset[left].corr(subset[right], method=method)
    if pd.isna(value):
        return None
    return float(value)


def summarize_results(
    results_df: pd.DataFrame,
    preview_df: pd.DataFrame,
    failures_df: pd.DataFrame,
) -> dict[str, Any]:
    successful = results_df[
        (results_df["status"] == "ok")
        & results_df["total_protein_length"].notna()
        & results_df["model_selected_rmsd"].notna()
    ].copy()

    pearson = _series_corr(
        successful,
        "total_protein_length",
        "model_selected_rmsd",
        method="pearson",
    )
    spearman = _series_corr(
        successful,
        "total_protein_length",
        "model_selected_rmsd",
        method="spearman",
    )

    regression: dict[str, float] | None = None
    if len(successful) >= 2:
        slope, intercept = np.polyfit(
            successful["total_protein_length"].astype(float),
            successful["model_selected_rmsd"].astype(float),
            1,
        )
        regression = {
            "slope": float(slope),
            "intercept": float(intercept),
        }

    binned_summary = build_binned_summary(successful).to_dict(orient="records")
    return {
        "n_requested": int(len(preview_df)),
        "n_successful": int((results_df["status"] == "ok").sum()) if not results_df.empty else 0,
        "n_failed": int((results_df["status"] == "failed").sum()) if not results_df.empty else 0,
        "evaluated_pdb_ids": successful["pdb_id"].tolist(),
        "pearson_length_vs_rmsd": pearson,
        "spearman_length_vs_rmsd": spearman,
        "linear_regression": regression,
        "mean_model_selected_rmsd": (
            None
            if successful.empty
            else float(successful["model_selected_rmsd"].mean())
        ),
        "median_model_selected_rmsd": (
            None
            if successful.empty
            else float(successful["model_selected_rmsd"].median())
        ),
        "failed_pdb_ids": failures_df["pdb_id"].tolist() if not failures_df.empty else [],
        "binned_summary": binned_summary,
    }


def write_summary_markdown(
    path: str | Path,
    *,
    summary: dict[str, Any],
    results_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    plot_paths: dict[str, Path],
) -> None:
    path = Path(path)
    lines = [
        "# OpenFold3 Length Benchmark",
        "",
        f"- Requested entries: {summary['n_requested']}",
        f"- Successful entries: {summary['n_successful']}",
        f"- Failed entries: {summary['n_failed']}",
        f"- Pearson(length, RMSD): {summary['pearson_length_vs_rmsd']}",
        f"- Spearman(length, RMSD): {summary['spearman_length_vs_rmsd']}",
        "",
        "## Plot files",
    ]

    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: {plot_path}")

    lines.extend(
        [
            "",
            "## Successful rows",
            "",
        ]
    )

    if results_df.empty:
        lines.append("No rows were produced.")
    else:
        success_view = results_df[
            [
                "pdb_id",
                "total_protein_length",
                "model_selected_rmsd",
                "oracle_best_rmsd",
                "sample_ranking_score",
                "avg_plddt",
                "status",
            ]
        ]
        lines.append(_markdown_table(success_view))

    if not failures_df.empty:
        lines.extend(
            [
                "",
                "## Failures",
                "",
                _markdown_table(failures_df[["pdb_id", "failure_reason"]]),
            ]
        )

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
