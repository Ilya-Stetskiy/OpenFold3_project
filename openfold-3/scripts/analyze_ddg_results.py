from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _metric_row(group: pd.DataFrame, *, score_column: str = "ddg") -> dict[str, Any]:
    n = len(group)
    score = group[score_column]
    exp = group["experimental_ddg"]
    return {
        "n": n,
        "pearson": score.corr(exp, method="pearson") if n >= 2 else math.nan,
        "spearman": score.corr(exp, method="spearman") if n >= 2 else math.nan,
        "rmse": math.sqrt(((score - exp) ** 2).mean()) if n else math.nan,
        "mae": (score - exp).abs().mean() if n else math.nan,
        "sign_accuracy": ((score >= 0) == (exp >= 0)).mean() if n else math.nan,
        "mean_prediction": score.mean() if n else math.nan,
        "mean_experimental": exp.mean() if n else math.nan,
    }


def _metrics_by_group(ok: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in ok.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        row.update(_metric_row(group))
        rows.append(row)
    metric_columns = list(_metric_row(pd.DataFrame(columns=["ddg", "experimental_ddg"])).keys())
    if not rows:
        return pd.DataFrame(columns=[*group_columns, *metric_columns])
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _paired_structure_comparison(ok: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in sorted(ok["method"].dropna().unique()):
        method_rows = ok[ok["method"] == method]
        pivot = method_rows.pivot_table(
            index=["protein_id", "mutation"],
            columns="structure_source",
            values="ddg",
            aggfunc="first",
        )
        if not {"experimental", "openfold"}.issubset(pivot.columns):
            continue
        paired = pivot.dropna(subset=["experimental", "openfold"])
        if paired.empty:
            continue
        diff = paired["openfold"] - paired["experimental"]
        rows.append(
            {
                "method": method,
                "n": len(paired),
                "experimental_openfold_pearson": paired["experimental"].corr(paired["openfold"]),
                "mean_openfold_minus_experimental": diff.mean(),
                "mae_openfold_minus_experimental": diff.abs().mean(),
                "median_abs_openfold_minus_experimental": diff.abs().median(),
                "sign_agreement": ((paired["experimental"] >= 0) == (paired["openfold"] >= 0)).mean(),
            }
        )
    return pd.DataFrame(rows)


def analyze_results(input_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_csv)
    frame = _numeric(frame, ["ddg", "ddg_raw", "ddg_std", "experimental_ddg", "n_runs_requested", "n_runs_valid"])

    status_counts = frame["status"].value_counts(dropna=False).to_dict()
    method_counts = frame["method"].value_counts(dropna=False).to_dict()
    source_counts = frame["structure_source"].value_counts(dropna=False).to_dict()

    ok = frame[
        (frame["status"] == "ok")
        & frame["ddg"].notna()
        & frame["experimental_ddg"].notna()
    ].copy()
    failed = frame[frame["status"] != "ok"].copy()
    top_errors = ok.copy()
    top_errors["abs_error"] = (top_errors["ddg"] - top_errors["experimental_ddg"]).abs()

    metrics_by_method = _metrics_by_group(ok, ["method"])
    metrics_by_method_source = _metrics_by_group(ok, ["method", "structure_source", "unit"])
    paired = _paired_structure_comparison(ok)

    esm2 = ok[ok["method"] == "esm2"].copy()
    esm2_sign_check = {}
    if not esm2.empty:
        inverted = esm2.copy()
        inverted["ddg_inverted"] = -inverted["ddg"]
        esm2_sign_check = {
            "as_is": _metric_row(esm2),
            "inverted": _metric_row(inverted, score_column="ddg_inverted"),
        }

    summary = {
        "input_csv": str(input_csv),
        "rows": int(len(frame)),
        "unique_cases": int(frame["protein_id"].nunique()) if "protein_id" in frame.columns else None,
        "ok_rows": int((frame["status"] == "ok").sum()),
        "failed_rows": int((frame["status"] != "ok").sum()),
        "status_counts": status_counts,
        "method_counts": method_counts,
        "structure_source_counts": source_counts,
        "esm2_sign_check": esm2_sign_check,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    metrics_by_method.to_csv(output_dir / "metrics_by_method.csv", index=False)
    metrics_by_method_source.to_csv(output_dir / "metrics_by_method_source.csv", index=False)
    paired.to_csv(output_dir / "paired_structure_comparison.csv", index=False)
    failed.to_csv(output_dir / "failures.csv", index=False)
    top_errors.sort_values("abs_error", ascending=False).head(100).to_csv(output_dir / "top_errors.csv", index=False)

    print(f"Rows: {len(frame)}")
    print(f"Unique cases: {summary['unique_cases']}")
    print(f"OK rows: {summary['ok_rows']}")
    print(f"Failed rows: {summary['failed_rows']}")
    print("\nMetrics by method/source:")
    print(metrics_by_method_source.to_string(index=False))
    print(f"\nWrote analysis artifacts to: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze ddG benchmark results.csv.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze_results(args.input_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
