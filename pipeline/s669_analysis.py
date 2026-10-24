"""Analysis helpers for S669 benchmark predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .external_benchmarks import S669Config, load_s669

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_SUFFIXES = ("dir", "inv")
EXCLUDED_PREDICTION_PREFIXES = {"Experimental_DDG"}
S669_ANALYSIS_COLUMNS = [
    "benchmark",
    "direction",
    "protein_id",
    "pdb_id",
    "chain",
    "position",
    "pdb_residue_id",
    "mutation_id",
    "method",
    "representation",
    "ddg",
    "experimental_ddg",
]
METRIC_COLUMNS = [
    "method",
    "representation",
    "n",
    "pearson",
    "spearman",
    "rmse",
    "mae",
    "bias",
    "sign_accuracy",
    "mean_pred",
    "mean_exp",
]


@dataclass(frozen=True, slots=True)
class S669AnalysisConfig:
    """Configuration for S669 published-prediction analysis."""

    data_dir: Path = REPO_ROOT / "data"
    output_dir: Path = REPO_ROOT / "analysis" / "s669_benchmark_analysis"
    s669_config: S669Config | None = None
    force_preprocess: bool = False

    def resolved(self) -> "S669AnalysisConfig":
        data_dir = Path(self.data_dir).expanduser().resolve()
        output_dir = Path(self.output_dir).expanduser().resolve()
        s669_config = self.s669_config or S669Config(data_dir=data_dir)
        if self.force_preprocess:
            s669_config = replace(s669_config, force_preprocess=True)
        return replace(self, data_dir=data_dir, output_dir=output_dir, s669_config=s669_config)


def analyze_s669_published_predictions(
    config: S669AnalysisConfig | dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Compare S669 bundled method predictions against experimental ddG."""

    cfg = _coerce_analysis_config(config)
    assert cfg.s669_config is not None
    processed = load_s669(cfg.s669_config)
    raw_path = _raw_s669_csv_path(cfg.s669_config.resolved())
    raw = pd.read_csv(raw_path)

    long_predictions = build_s669_prediction_table(processed, raw)
    metrics = metric_table(long_predictions)
    coverage = coverage_table(long_predictions)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = cfg.output_dir / "s669_published_predictions_long.csv"
    metrics_path = cfg.output_dir / "s669_published_prediction_metrics.csv"
    coverage_path = cfg.output_dir / "s669_published_prediction_coverage.csv"
    summary_path = cfg.output_dir / "s669_analysis_summary.json"

    long_predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    summary = {
        "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
        "raw_s669_csv": str(raw_path),
        "processed_s669_rows": int(len(processed)),
        "prediction_rows": int(len(long_predictions)),
        "methods": sorted(long_predictions["method"].unique().tolist()),
        "outputs": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "coverage": str(coverage_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "coverage": coverage_path,
        "summary": summary_path,
    }


def build_s669_prediction_table(processed: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per imported S669 case and published predictor."""

    prediction_columns = _prediction_columns(raw)
    rows: list[dict[str, Any]] = []
    raw_by_key = {
        (str(row["Protein"]), str(row["Seq_Mut"])): row
        for row in raw.to_dict(orient="records")
    }

    for case in processed.to_dict(orient="records"):
        base_protein = str(case["protein_id"]).removesuffix("_direct").removesuffix("_inverse")
        raw_row = raw_by_key.get((base_protein, _direct_mutation_id(case)))
        if raw_row is None:
            continue
        suffix = "dir" if case["direction"] == "direct" else "inv"
        for column in prediction_columns[suffix]:
            value = raw_row.get(column)
            if pd.isna(value):
                continue
            rows.append(
                {
                    "benchmark": case["benchmark"],
                    "direction": case["direction"],
                    "protein_id": case["protein_id"],
                    "pdb_id": case["pdb_id"],
                    "chain": case["chain"],
                    "position": int(case["position"]),
                    "pdb_residue_id": str(case["pdb_residue_id"]),
                    "mutation_id": case["mutation_id"],
                    "method": column[: -(len(suffix) + 1)],
                    "representation": _representation_for_method(column[: -(len(suffix) + 1)]),
                    "ddg": float(value),
                    "experimental_ddg": float(case["experimental_ddg"]),
                }
            )

    frame = pd.DataFrame(rows, columns=S669_ANALYSIS_COLUMNS)
    return frame.sort_values(
        ["method", "direction", "protein_id", "position", "mutation_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics by method for S669 published predictions."""

    rows: list[dict[str, Any]] = []
    for (method, representation), group in predictions.groupby(["method", "representation"], sort=True):
        rows.append({"method": method, "representation": representation, **_metric_row(group)})
    return pd.DataFrame(rows, columns=METRIC_COLUMNS).sort_values(
        ["spearman", "pearson", "sign_accuracy"],
        ascending=[False, False, False],
        kind="mergesort",
    )


def coverage_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return S669 prediction coverage by method and direction."""

    return (
        predictions.groupby(["method", "representation", "direction"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["method", "direction"], kind="mergesort")
        .reset_index(drop=True)
    )


def _metric_row(group: pd.DataFrame) -> dict[str, float | int]:
    clean = group[["ddg", "experimental_ddg"]].dropna()
    n = int(len(clean))
    if n == 0:
        return {
            "n": 0,
            "pearson": float("nan"),
            "spearman": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "bias": float("nan"),
            "sign_accuracy": float("nan"),
            "mean_pred": float("nan"),
            "mean_exp": float("nan"),
        }
    pred = clean["ddg"]
    exp = clean["experimental_ddg"]
    diff = pred - exp
    return {
        "n": n,
        "pearson": float(pred.corr(exp, method="pearson")) if n > 1 else float("nan"),
        "spearman": float(pred.corr(exp, method="spearman")) if n > 1 else float("nan"),
        "rmse": float((diff.pow(2).mean()) ** 0.5),
        "mae": float(diff.abs().mean()),
        "bias": float(diff.mean()),
        "sign_accuracy": float((pred.ge(0) == exp.ge(0)).mean()),
        "mean_pred": float(pred.mean()),
        "mean_exp": float(exp.mean()),
    }


def _prediction_columns(raw: pd.DataFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"dir": [], "inv": []}
    for column in raw.columns:
        for suffix in PREDICTION_SUFFIXES:
            marker = f"_{suffix}"
            if not column.endswith(marker):
                continue
            prefix = column[: -len(marker)]
            if prefix in EXCLUDED_PREDICTION_PREFIXES:
                continue
            result[suffix].append(column)
    return result


def _direct_mutation_id(case: dict[str, Any]) -> str:
    mutation_id = str(case["mutation_id"])
    if case["direction"] == "direct":
        return mutation_id
    return f"{mutation_id[-1]}{mutation_id[1:-1]}{mutation_id[0]}"


def _representation_for_method(method: str) -> str:
    if "Seq" in method or method in {"SAAFEC-Seq"}:
        return "sequence"
    return "structure"


def _raw_s669_csv_path(config: S669Config) -> Path:
    if config.source_csv is not None:
        return config.source_csv
    assert config.raw_dir is not None
    return config.raw_dir / "S669" / "S669.csv"


def _coerce_analysis_config(
    config: S669AnalysisConfig | dict[str, Any] | None,
) -> S669AnalysisConfig:
    if config is None:
        return S669AnalysisConfig().resolved()
    if isinstance(config, S669AnalysisConfig):
        return config.resolved()
    if isinstance(config, dict):
        return S669AnalysisConfig(**config).resolved()
    raise TypeError(f"Unsupported S669 analysis config type: {type(config)!r}")
