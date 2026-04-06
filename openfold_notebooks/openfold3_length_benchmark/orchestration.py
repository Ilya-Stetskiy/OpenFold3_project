from __future__ import annotations

import json
import hashlib
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import pandas as pd

from .analysis import (
    load_jsonl,
    rmsd_rows_to_dataframe,
    select_model_row,
    select_oracle_row,
    summarize_results,
    write_summary_markdown,
)
from .benchmarking import run_rmsd_benchmark
from .composition import collect_entry_compositions, parse_pdb_ids, preview_entries
from .interop import (
    RuntimeConfig,
    clone_runtime,
    default_runs_root,
    run_prediction,
)
from .models import BenchmarkRunResult, EntryComposition
from .plots import write_binned_svg, write_scatter_svg


RESULT_COLUMNS = [
    "pdb_id",
    "status",
    "failure_reason",
    "chain_group",
    "total_protein_length",
    "chain_count",
    "chain_lengths",
    "matched_atom_count",
    "matched_atom_count_ca",
    "matched_atom_count_backbone",
    "model_selected_rmsd",
    "model_selected_rmsd_ca",
    "model_selected_rmsd_backbone",
    "oracle_best_rmsd",
    "oracle_best_rmsd_ca",
    "oracle_best_rmsd_backbone",
    "avg_plddt",
    "sample_ranking_score",
    "reference_path",
    "submitted_query_path",
    "openfold_query_path",
    "predict_run_dir",
    "predict_summary_dir",
    "rmsd_output_dir",
    "model_selected_sample",
    "oracle_sample",
    "oracle_sample_ca",
    "oracle_sample_backbone",
]

def _slug_timestamp(prefix: str) -> str:
    safe_prefix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in prefix).strip("_")
    return f"{safe_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _copy_reference(source_path: Path, refs_dir: Path, pdb_id: str) -> Path:
    refs_dir.mkdir(parents=True, exist_ok=True)
    destination = refs_dir / f"{pdb_id}.cif"
    if destination.exists():
        return destination
    shutil.copy2(source_path, destination)
    return destination


def _build_query_payload(composition: EntryComposition) -> dict[str, Any]:
    return {
        "queries": {
            composition.pdb_id: {
                "chains": composition.molecules,
            }
        }
    }


def _chain_group(chain_count: int) -> str:
    if chain_count == 1:
        return "single_chain"
    if chain_count == 2:
        return "double_chain"
    return "other_chain"


def _prediction_cache_key(
    *,
    query_payload: dict[str, Any],
    use_msa_server: bool,
    num_diffusion_samples: int,
    num_model_seeds: int,
    runner_yaml: str | Path | None,
) -> str:
    payload = {
        "query_payload": query_payload,
        "use_msa_server": use_msa_server,
        "num_diffusion_samples": num_diffusion_samples,
        "num_model_seeds": num_model_seeds,
        "runner_yaml": None if runner_yaml is None else str(Path(runner_yaml)),
        "use_templates": False,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _prediction_cache_root(
    runs_root: Path,
    prediction_cache_root: str | Path | None = None,
) -> Path:
    return (
        Path(prediction_cache_root).expanduser().resolve()
        if prediction_cache_root is not None
        else runs_root.parent / "prediction_cache"
    )


def _prediction_cache_dir(
    *,
    runs_root: Path,
    composition: EntryComposition,
    cache_key: str,
    prediction_cache_root: str | Path | None = None,
) -> Path:
    return (
        _prediction_cache_root(runs_root, prediction_cache_root)
        / _chain_group(composition.chain_count)
        / composition.pdb_id
        / cache_key
    )


def _is_prediction_cache_complete(cache_dir: Path) -> bool:
    output_dir = cache_dir / "output"
    summary_dir = cache_dir / "summary"
    return output_dir.exists() and summary_dir.exists() and any(output_dir.rglob("*_model.cif")) and any(output_dir.rglob("*_confidences_aggregated.json"))


def _cached_prediction_result(cache_dir: Path):
    return SimpleNamespace(
        query_path=cache_dir / "query.json",
        run_dir=cache_dir,
        summary_dir=cache_dir / "summary",
        output_dir=cache_dir / "output",
    )


def _store_prediction_cache(prediction_result, cache_dir: Path, metadata: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_cache_dir = cache_dir / "output"
    summary_cache_dir = cache_dir / "summary"
    if not output_cache_dir.exists():
        shutil.copytree(prediction_result.output_dir, output_cache_dir)
    if not summary_cache_dir.exists():
        shutil.copytree(prediction_result.summary_dir, summary_cache_dir)
    if not (cache_dir / "query.json").exists():
        shutil.copy2(prediction_result.query_path, cache_dir / "query.json")
    _write_json(cache_dir / "cache_metadata.json", metadata)


def _run_rmsd_benchmark(
    *,
    runtime: RuntimeConfig,
    pred_root: Path,
    ref_dir: Path,
    output_dir: Path,
    atom_set: str,
) -> Path:
    started = time.perf_counter()
    output_dir = run_rmsd_benchmark(
        pred_root=pred_root,
        ref_dir=ref_dir,
        output_dir=output_dir,
        atom_set=atom_set,
    )
    _write_json(
        output_dir / "benchmark_runtime.json",
        {
            "elapsed_seconds": time.perf_counter() - started,
            "atom_set": atom_set,
            "pred_root": str(pred_root),
            "ref_dir": str(ref_dir),
        },
    )
    return output_dir


def _format_chain_lengths(composition: EntryComposition) -> str:
    return ",".join(
        f"{chain_id}:{composition.chain_lengths[chain_id]}"
        for chain_id in sorted(composition.chain_lengths)
    )


def _failure_row(
    composition: EntryComposition,
    *,
    failure_reason: str,
    reference_path: Path | None = None,
    submitted_query_path: Path | None = None,
    openfold_query_path: Path | None = None,
    predict_run_dir: Path | None = None,
    predict_summary_dir: Path | None = None,
    rmsd_output_dir: Path | None = None,
) -> dict[str, Any]:
    return {
        "pdb_id": composition.pdb_id,
        "status": "failed",
        "failure_reason": failure_reason,
        "chain_group": _chain_group(composition.chain_count),
        "total_protein_length": composition.total_protein_length or None,
        "chain_count": composition.chain_count,
        "chain_lengths": _format_chain_lengths(composition),
        "matched_atom_count": None,
        "matched_atom_count_ca": None,
        "matched_atom_count_backbone": None,
        "model_selected_rmsd": None,
        "model_selected_rmsd_ca": None,
        "model_selected_rmsd_backbone": None,
        "oracle_best_rmsd": None,
        "oracle_best_rmsd_ca": None,
        "oracle_best_rmsd_backbone": None,
        "avg_plddt": None,
        "sample_ranking_score": None,
        "reference_path": None if reference_path is None else str(reference_path),
        "submitted_query_path": None if submitted_query_path is None else str(submitted_query_path),
        "openfold_query_path": None if openfold_query_path is None else str(openfold_query_path),
        "predict_run_dir": None if predict_run_dir is None else str(predict_run_dir),
        "predict_summary_dir": None if predict_summary_dir is None else str(predict_summary_dir),
        "rmsd_output_dir": None if rmsd_output_dir is None else str(rmsd_output_dir),
        "model_selected_sample": None,
        "oracle_sample": None,
        "oracle_sample_ca": None,
        "oracle_sample_backbone": None,
    }


def _normalize_atom_sets(atom_set: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(atom_set, str):
        requested = [atom_set]
    else:
        requested = list(atom_set)
    normalized: list[str] = []
    for value in requested:
        name = str(value).strip().lower()
        if not name:
            continue
        if name not in {"ca", "backbone", "all"}:
            raise ValueError(f"Unsupported atom_set: {value!r}")
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise ValueError("At least one atom_set must be provided")
    if "ca" in normalized:
        normalized = ["ca"] + [
            name for name in normalized if name != "ca"
        ]
    return tuple(normalized)


def _select_selected_sample_name(rmsd_df: pd.DataFrame) -> str | None:
    if rmsd_df.empty:
        return None
    ordered = rmsd_df.copy()
    ordered = ordered.assign(
        _ranking_missing=ordered["sample_ranking_score"].isna().astype(int),
        _plddt_missing=ordered["avg_plddt"].isna().astype(int),
    ).sort_values(
        by=[
            "_ranking_missing",
            "sample_ranking_score",
            "_plddt_missing",
            "avg_plddt",
            "sample",
        ],
        ascending=[True, False, True, False, True],
    )
    if ordered.empty:
        return None
    return str(ordered.iloc[0]["sample"])


def _metric_value(metric_rows: dict[str, dict[str, Any]], atom_set: str, key: str) -> Any:
    row = metric_rows.get(atom_set) or {}
    return row.get(key)


def _success_row(
    composition: EntryComposition,
    *,
    primary_atom_set: str,
    reference_path: Path,
    submitted_query_path: Path,
    prediction_result,
    rmsd_output_dir: str | Path,
    selected_sample_name: str,
    selected_rows_by_metric: dict[str, dict[str, Any]],
    oracle_rows_by_metric: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "pdb_id": composition.pdb_id,
        "status": "ok",
        "failure_reason": None,
        "chain_group": _chain_group(composition.chain_count),
        "total_protein_length": composition.total_protein_length,
        "chain_count": composition.chain_count,
        "chain_lengths": _format_chain_lengths(composition),
        "matched_atom_count": _metric_value(selected_rows_by_metric, primary_atom_set, "matched_atom_count"),
        "matched_atom_count_ca": _metric_value(selected_rows_by_metric, "ca", "matched_atom_count"),
        "matched_atom_count_backbone": _metric_value(selected_rows_by_metric, "backbone", "matched_atom_count"),
        "model_selected_rmsd": _metric_value(selected_rows_by_metric, primary_atom_set, "rmsd_after_superposition"),
        "model_selected_rmsd_ca": _metric_value(selected_rows_by_metric, "ca", "rmsd_after_superposition"),
        "model_selected_rmsd_backbone": _metric_value(selected_rows_by_metric, "backbone", "rmsd_after_superposition"),
        "oracle_best_rmsd": _metric_value(oracle_rows_by_metric, primary_atom_set, "rmsd_after_superposition"),
        "oracle_best_rmsd_ca": _metric_value(oracle_rows_by_metric, "ca", "rmsd_after_superposition"),
        "oracle_best_rmsd_backbone": _metric_value(oracle_rows_by_metric, "backbone", "rmsd_after_superposition"),
        "avg_plddt": _metric_value(selected_rows_by_metric, primary_atom_set, "avg_plddt"),
        "sample_ranking_score": _metric_value(selected_rows_by_metric, primary_atom_set, "sample_ranking_score"),
        "reference_path": str(reference_path),
        "submitted_query_path": str(submitted_query_path),
        "openfold_query_path": str(prediction_result.query_path),
        "predict_run_dir": str(prediction_result.run_dir),
        "predict_summary_dir": str(prediction_result.summary_dir),
        "rmsd_output_dir": str(rmsd_output_dir),
        "model_selected_sample": selected_sample_name,
        "oracle_sample": _metric_value(oracle_rows_by_metric, primary_atom_set, "sample"),
        "oracle_sample_ca": _metric_value(oracle_rows_by_metric, "ca", "sample"),
        "oracle_sample_backbone": _metric_value(oracle_rows_by_metric, "backbone", "sample"),
    }


def _results_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(
        by=["status", "pdb_id"],
        ascending=[False, True],
    )


def _sample_points_dataframe(sample_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not sample_rows:
        return pd.DataFrame(
            columns=[
                "pdb_id",
                "chain_group",
                "total_protein_length",
                "sample",
                "seed",
                "rmsd_after_superposition",
                "rmsd_ca",
                "rmsd_backbone",
                "sample_ranking_score",
                "avg_plddt",
                "is_selected_model",
                "is_oracle_best",
                "is_oracle_best_ca",
                "is_oracle_best_backbone",
            ]
        )
    return pd.DataFrame(sample_rows).sort_values(
        by=["pdb_id", "sample_ranking_score", "rmsd_after_superposition"],
        ascending=[True, False, True],
    )


def run_length_benchmark(
    runtime: RuntimeConfig,
    pdb_ids: str | Iterable[str],
    *,
    atom_set: str | Iterable[str] = ("ca", "backbone"),
    use_msa_server: bool = True,
    num_diffusion_samples: int = 1,
    num_model_seeds: int = 1,
    runner_yaml: str | Path | None = None,
    output_root: str | Path | None = None,
    prediction_cache_root: str | Path | None = None,
    max_entries: int | None = None,
    cache_dir: str | Path | None = None,
) -> BenchmarkRunResult:
    atom_sets = _normalize_atom_sets(atom_set)
    primary_atom_set = atom_sets[0]
    requested_ids = parse_pdb_ids(pdb_ids, max_entries=max_entries)
    compositions = collect_entry_compositions(
        requested_ids,
        cache_dir=cache_dir,
        max_entries=max_entries,
    )
    preview_df = preview_entries(
        requested_ids,
        cache_dir=cache_dir,
        max_entries=max_entries,
    )

    runs_root = Path(output_root or default_runs_root()).expanduser().resolve()
    run_name = _slug_timestamp("length_benchmark")
    run_root = runs_root / run_name
    refs_dir = run_root / "refs"
    queries_dir = run_root / "queries"
    openfold_outputs_dir = run_root / "openfold_outputs"
    rmsd_root = run_root / "rmsd"
    plots_dir = run_root / "plots"

    for path in (refs_dir, queries_dir, openfold_outputs_dir, rmsd_root, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    preview_df.to_csv(run_root / "preview.csv", index=False)

    batch_runtime = clone_runtime(runtime, results_dir=openfold_outputs_dir)

    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for composition in compositions:
        reference_path: Path | None = None
        submitted_query_path: Path | None = None
        prediction_result = None
        rmsd_output_dir: Path | None = None

        if composition.source_path is not None:
            reference_path = _copy_reference(composition.source_path, refs_dir, composition.pdb_id)

        if composition.status != "ok":
            rows.append(
                _failure_row(
                    composition,
                    failure_reason=composition.issue or "Failed during composition parsing",
                    reference_path=reference_path,
                )
            )
            continue

        query_payload = _build_query_payload(composition)
        submitted_query_path = _write_json(queries_dir / f"{composition.pdb_id}.json", query_payload)

        try:
            cache_key = _prediction_cache_key(
                query_payload=query_payload,
                use_msa_server=use_msa_server,
                num_diffusion_samples=num_diffusion_samples,
                num_model_seeds=num_model_seeds,
                runner_yaml=runner_yaml,
            )
            cache_dir = _prediction_cache_dir(
                runs_root=runs_root,
                composition=composition,
                cache_key=cache_key,
                prediction_cache_root=prediction_cache_root,
            )

            if _is_prediction_cache_complete(cache_dir):
                prediction_result = _cached_prediction_result(cache_dir)
            else:
                prediction_result = run_prediction(
                    batch_runtime,
                    query_payload,
                    experiment_name=composition.pdb_id,
                    use_templates=False,
                    use_msa_server=use_msa_server,
                    num_diffusion_samples=num_diffusion_samples,
                    num_model_seeds=num_model_seeds,
                    runner_yaml=runner_yaml,
                )
                _store_prediction_cache(
                    prediction_result,
                    cache_dir,
                    metadata={
                        "pdb_id": composition.pdb_id,
                        "cache_key": cache_key,
                        "chain_group": _chain_group(composition.chain_count),
                        "num_diffusion_samples": num_diffusion_samples,
                        "num_model_seeds": num_model_seeds,
                        "use_msa_server": use_msa_server,
                        "runner_yaml": None if runner_yaml is None else str(runner_yaml),
                    },
                )

            rmsd_output_dirs: dict[str, Path] = {}
            rmsd_frames: dict[str, pd.DataFrame] = {}
            for metric_name in atom_sets:
                metric_output_dir = _run_rmsd_benchmark(
                    runtime=batch_runtime,
                    pred_root=prediction_result.output_dir,
                    ref_dir=refs_dir,
                    output_dir=rmsd_root / composition.pdb_id / metric_name,
                    atom_set=metric_name,
                )
                rmsd_output_dirs[metric_name] = metric_output_dir
                rmsd_rows = load_jsonl(metric_output_dir / "rmsd_rows.jsonl")
                rmsd_frames[metric_name] = rmsd_rows_to_dataframe(rmsd_rows)

            primary_df = rmsd_frames[atom_sets[0]]
            selected_sample_name = _select_selected_sample_name(primary_df)
            if selected_sample_name is None:
                raise ValueError(f"No RMSD rows were produced for {composition.pdb_id}")

            selected_rows_by_metric: dict[str, dict[str, Any]] = {}
            oracle_rows_by_metric: dict[str, dict[str, Any]] = {}
            merged_sample_rows: dict[str, dict[str, Any]] = {}
            for metric_name, rmsd_df in rmsd_frames.items():
                oracle_row = select_oracle_row(rmsd_df)
                if oracle_row is None:
                    raise ValueError(
                        f"No RMSD rows were produced for {composition.pdb_id} / {metric_name}"
                    )
                oracle_rows_by_metric[metric_name] = oracle_row

                selected_matches = rmsd_df[rmsd_df["sample"] == selected_sample_name]
                if selected_matches.empty:
                    raise ValueError(
                        f"Selected sample {selected_sample_name} is missing for {composition.pdb_id} / {metric_name}"
                    )
                selected_rows_by_metric[metric_name] = selected_matches.iloc[0].to_dict()

                for sample_row in rmsd_df.to_dict(orient="records"):
                    sample_name = str(sample_row.get("sample"))
                    merged = merged_sample_rows.setdefault(
                        sample_name,
                        {
                            "pdb_id": composition.pdb_id,
                            "chain_group": _chain_group(composition.chain_count),
                            "total_protein_length": composition.total_protein_length,
                            "sample": sample_name,
                            "seed": sample_row.get("seed"),
                            "sample_ranking_score": sample_row.get("sample_ranking_score"),
                            "avg_plddt": sample_row.get("avg_plddt"),
                        },
                    )
                    merged[f"rmsd_{metric_name}"] = sample_row.get("rmsd_after_superposition")
                    merged[f"matched_atom_count_{metric_name}"] = sample_row.get("matched_atom_count")

            for sample_name, merged in merged_sample_rows.items():
                merged["rmsd_after_superposition"] = merged.get(f"rmsd_{primary_atom_set}")
                merged["is_selected_model"] = sample_name == selected_sample_name
                merged["is_oracle_best"] = sample_name == _metric_value(
                    oracle_rows_by_metric, primary_atom_set, "sample"
                )
                merged["is_oracle_best_ca"] = sample_name == _metric_value(
                    oracle_rows_by_metric, "ca", "sample"
                )
                merged["is_oracle_best_backbone"] = sample_name == _metric_value(
                    oracle_rows_by_metric, "backbone", "sample"
                )
                sample_rows.append(merged)

            rows.append(
                _success_row(
                    composition,
                    primary_atom_set=primary_atom_set,
                    reference_path=reference_path or composition.source_path,  # type: ignore[arg-type]
                    submitted_query_path=submitted_query_path,
                    prediction_result=prediction_result,
                    rmsd_output_dir=rmsd_root / composition.pdb_id,
                    selected_sample_name=selected_sample_name,
                    selected_rows_by_metric=selected_rows_by_metric,
                    oracle_rows_by_metric=oracle_rows_by_metric,
                )
            )
        except Exception as exc:
            rows.append(
                _failure_row(
                    composition,
                    failure_reason=str(exc),
                    reference_path=reference_path,
                    submitted_query_path=submitted_query_path,
                    openfold_query_path=None if prediction_result is None else prediction_result.query_path,
                    predict_run_dir=None if prediction_result is None else prediction_result.run_dir,
                    predict_summary_dir=None if prediction_result is None else prediction_result.summary_dir,
                    rmsd_output_dir=rmsd_output_dir,
                )
            )

    results_df = _results_dataframe(rows)
    sample_points_df = _sample_points_dataframe(sample_rows)
    failures_df = results_df[results_df["status"] == "failed"].copy()
    failures_df.to_csv(run_root / "failures.csv", index=False)
    results_df.to_csv(run_root / "results.csv", index=False)
    sample_points_df.to_csv(run_root / "sample_points.csv", index=False)

    graph_results_root = run_root / "graph_results"
    single_dir = graph_results_root / "single_chain"
    double_dir = graph_results_root / "double_chain"
    for category_dir in (single_dir, double_dir):
        category_dir.mkdir(parents=True, exist_ok=True)

    results_df[results_df["chain_group"] == "single_chain"].to_csv(single_dir / "results.csv", index=False)
    results_df[results_df["chain_group"] == "double_chain"].to_csv(double_dir / "results.csv", index=False)
    sample_points_df[sample_points_df["chain_group"] == "single_chain"].to_csv(single_dir / "sample_points.csv", index=False)
    sample_points_df[sample_points_df["chain_group"] == "double_chain"].to_csv(double_dir / "sample_points.csv", index=False)

    summary = summarize_results(results_df, preview_df, failures_df)
    plot_paths = {
        "scatter_svg": write_scatter_svg(
            results_df,
            output_path=plots_dir / "length_vs_rmsd.svg",
            regression=summary.get("linear_regression"),  # type: ignore[arg-type]
            sample_points_df=sample_points_df,
        ),
        "binned_svg": write_binned_svg(
            pd.DataFrame(summary.get("binned_summary") or []),
            output_path=plots_dir / "binned_length_vs_rmsd.svg",
        ),
    }

    summary_path = run_root / "summary.md"
    write_summary_markdown(
        summary_path,
        summary=summary,
        results_df=results_df,
        failures_df=failures_df,
        plot_paths=plot_paths,
    )

    results_json_path = _write_json(
        run_root / "results.json",
        {
            "run_name": run_name,
            "run_root": str(run_root),
            "summary": summary,
            "records": results_df.to_dict(orient="records"),
            "sample_points": sample_points_df.to_dict(orient="records"),
            "preview": preview_df.to_dict(orient="records"),
        },
    )

    return BenchmarkRunResult(
        run_name=run_name,
        run_root=run_root,
        preview_df=preview_df,
        results_df=results_df,
        sample_points_df=sample_points_df,
        failures_df=failures_df,
        summary=summary,
        summary_path=summary_path,
        results_csv_path=run_root / "results.csv",
        sample_points_csv_path=run_root / "sample_points.csv",
        results_json_path=results_json_path,
        failures_csv_path=run_root / "failures.csv",
        plot_paths=plot_paths,
    )
