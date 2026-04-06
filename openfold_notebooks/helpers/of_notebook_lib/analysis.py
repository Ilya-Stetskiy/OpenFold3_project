from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class SampleRecord:
    sample_name: str
    query_name: str
    seed_name: str | None
    agg_path: Path
    conf_path: Path | None
    model_path: Path | None
    sample_ranking_score: float | None
    iptm: float | None
    ptm: float | None
    avg_plddt: float | None
    gpde: float | None
    has_clash: float | None
    raw_json: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_names(agg_path: Path) -> tuple[str, str | None]:
    parent = agg_path.parent
    if parent.name.startswith("seed_") and parent.parent.name:
        return parent.parent.name, parent.name
    return parent.name, None


def collect_samples(output_dir: Path) -> list[SampleRecord]:
    samples: list[SampleRecord] = []

    for agg_path in sorted(output_dir.rglob("*_confidences_aggregated.json")):
        raw = _load_json(agg_path)
        sample_name = agg_path.name.replace("_confidences_aggregated.json", "")
        query_name, seed_name = _infer_names(agg_path)
        conf_path = agg_path.with_name(f"{sample_name}_confidences.json")
        model_cif = agg_path.with_name(f"{sample_name}_model.cif")
        model_pdb = agg_path.with_name(f"{sample_name}_model.pdb")
        model_path = model_cif if model_cif.exists() else model_pdb if model_pdb.exists() else None

        samples.append(
            SampleRecord(
                sample_name=sample_name,
                query_name=query_name,
                seed_name=seed_name,
                agg_path=agg_path,
                conf_path=conf_path if conf_path.exists() else None,
                model_path=model_path,
                sample_ranking_score=_safe_float(raw.get("sample_ranking_score")),
                iptm=_safe_float(raw.get("iptm")),
                ptm=_safe_float(raw.get("ptm")),
                avg_plddt=_safe_float(raw.get("avg_plddt")),
                gpde=_safe_float(raw.get("gpde")),
                has_clash=_safe_float(raw.get("has_clash")),
                raw_json=raw,
            )
        )

    return samples


def samples_to_dataframe(samples: list[SampleRecord]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        row = asdict(sample)
        row["agg_path"] = str(sample.agg_path)
        row["conf_path"] = str(sample.conf_path) if sample.conf_path else None
        row["model_path"] = str(sample.model_path) if sample.model_path else None
        row["mutation_label"] = (
            sample.query_name.split("__", 1)[1] if "__" in sample.query_name else sample.query_name
        )
        rows.append(row)

    return pd.DataFrame(rows)


def best_samples_by_metric(samples: list[SampleRecord]) -> dict[str, SampleRecord | None]:
    metrics = {
        "sample_ranking_score": True,
        "iptm": True,
        "ptm": True,
        "avg_plddt": True,
        "gpde": False,
    }
    sample_df = samples_to_dataframe(samples)
    if sample_df.empty:
        return {name: None for name in metrics}

    clash_free = sample_df[sample_df["has_clash"].isin([None, 0.0]) | sample_df["has_clash"].isna()]
    if clash_free.empty:
        clash_free = sample_df

    winners: dict[str, SampleRecord | None] = {}
    for metric, higher_is_better in metrics.items():
        metric_df = clash_free[clash_free[metric].notna()]
        if metric_df.empty:
            winners[metric] = None
            continue
        ordered = metric_df.sort_values(metric, ascending=not higher_is_better)
        best_name = ordered.iloc[0]["sample_name"]
        winners[metric] = next(sample for sample in samples if sample.sample_name == best_name)

    return winners


def summarize_mutation_batch(df_samples: pd.DataFrame) -> pd.DataFrame:
    if df_samples.empty:
        return df_samples.copy()

    agg_map = {
        "sample_ranking_score": "max",
        "iptm": "max",
        "ptm": "max",
        "avg_plddt": "max",
        "gpde": "min",
        "has_clash": "min",
    }

    summary = (
        df_samples.groupby(["query_name", "mutation_label"], dropna=False)
        .agg(agg_map)
        .reset_index()
    )
    summary["is_wt"] = summary["mutation_label"].eq("WT")
    return summary


def rank_mutations(df_summary: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if df_summary.empty:
        return df_summary.copy()

    ranked = df_summary.sort_values(
        by=[
            "is_wt",
            "has_clash",
            "sample_ranking_score",
            "iptm",
            "ptm",
            "avg_plddt",
            "gpde",
        ],
        ascending=[False, True, False, False, False, False, True],
    )
    return ranked.head(top_n).reset_index(drop=True)


def _copy_if_exists(src: Path | None, dst_dir: Path) -> None:
    if src and src.exists():
        shutil.copy2(src, dst_dir / src.name)


def write_best_samples_report(
    report_path: Path,
    samples: list[SampleRecord],
    winners: dict[str, SampleRecord | None],
) -> None:
    lines = [f"Total samples found: {len(samples)}", ""]

    for metric, sample in winners.items():
        lines.append(f"Best by {metric}:")
        if sample is None:
            lines.append("  sample: NA")
        else:
            lines.append(f"  sample: {sample.sample_name}")
            lines.append(f"  query_name: {sample.query_name}")
            lines.append(f"  seed_name: {sample.seed_name or 'NA'}")
            lines.append(f"  sample_ranking_score: {sample.sample_ranking_score}")
            lines.append(f"  iptm: {sample.iptm}")
            lines.append(f"  ptm: {sample.ptm}")
            lines.append(f"  avg_plddt: {sample.avg_plddt}")
            lines.append(f"  gpde: {sample.gpde}")
            lines.append(f"  has_clash: {sample.has_clash}")
            lines.append(f"  agg_path: {sample.agg_path}")
            lines.append(f"  conf_path: {sample.conf_path}")
            lines.append(f"  model_path: {sample.model_path}")
        lines.append("")

    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def copy_best_artifacts(summary_dir: Path, winners: dict[str, SampleRecord | None]) -> None:
    for metric, sample in winners.items():
        if sample is None:
            continue
        metric_dir = summary_dir / f"best_by_{metric}"
        metric_dir.mkdir(parents=True, exist_ok=True)
        _copy_if_exists(sample.agg_path, metric_dir)
        _copy_if_exists(sample.conf_path, metric_dir)
        _copy_if_exists(sample.model_path, metric_dir)
