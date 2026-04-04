from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
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
    "total_protein_length",
    "chain_count",
    "chain_lengths",
    "matched_atom_count",
    "model_selected_rmsd",
    "oracle_best_rmsd",
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
        "total_protein_length": composition.total_protein_length or None,
        "chain_count": composition.chain_count,
        "chain_lengths": _format_chain_lengths(composition),
        "matched_atom_count": None,
        "model_selected_rmsd": None,
        "oracle_best_rmsd": None,
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
    }


def _success_row(
    composition: EntryComposition,
    *,
    reference_path: Path,
    submitted_query_path: Path,
    prediction_result,
    rmsd_output_dir: Path,
    selected_row: dict[str, Any],
    oracle_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pdb_id": composition.pdb_id,
        "status": "ok",
        "failure_reason": None,
        "total_protein_length": composition.total_protein_length,
        "chain_count": composition.chain_count,
        "chain_lengths": _format_chain_lengths(composition),
        "matched_atom_count": selected_row.get("matched_atom_count"),
        "model_selected_rmsd": selected_row.get("rmsd_after_superposition"),
        "oracle_best_rmsd": oracle_row.get("rmsd_after_superposition"),
        "avg_plddt": selected_row.get("avg_plddt"),
        "sample_ranking_score": selected_row.get("sample_ranking_score"),
        "reference_path": str(reference_path),
        "submitted_query_path": str(submitted_query_path),
        "openfold_query_path": str(prediction_result.query_path),
        "predict_run_dir": str(prediction_result.run_dir),
        "predict_summary_dir": str(prediction_result.summary_dir),
        "rmsd_output_dir": str(rmsd_output_dir),
        "model_selected_sample": selected_row.get("sample"),
        "oracle_sample": oracle_row.get("sample"),
    }


def _results_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(
        by=["status", "pdb_id"],
        ascending=[False, True],
    )


def run_length_benchmark(
    runtime: RuntimeConfig,
    pdb_ids: str | Iterable[str],
    *,
    atom_set: str = "ca",
    use_msa_server: bool = True,
    num_diffusion_samples: int = 1,
    num_model_seeds: int = 1,
    runner_yaml: str | Path | None = None,
    output_root: str | Path | None = None,
    max_entries: int | None = None,
    cache_dir: str | Path | None = None,
) -> BenchmarkRunResult:
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

            rmsd_output_dir = _run_rmsd_benchmark(
                runtime=batch_runtime,
                pred_root=prediction_result.output_dir,
                ref_dir=refs_dir,
                output_dir=rmsd_root / composition.pdb_id,
                atom_set=atom_set,
            )

            rmsd_rows = load_jsonl(rmsd_output_dir / "rmsd_rows.jsonl")
            rmsd_df = rmsd_rows_to_dataframe(rmsd_rows)
            selected_row = select_model_row(rmsd_df)
            oracle_row = select_oracle_row(rmsd_df)
            if selected_row is None or oracle_row is None:
                raise ValueError(f"No RMSD rows were produced for {composition.pdb_id}")

            rows.append(
                _success_row(
                    composition,
                    reference_path=reference_path or composition.source_path,  # type: ignore[arg-type]
                    submitted_query_path=submitted_query_path,
                    prediction_result=prediction_result,
                    rmsd_output_dir=rmsd_output_dir,
                    selected_row=selected_row,
                    oracle_row=oracle_row,
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
    failures_df = results_df[results_df["status"] == "failed"].copy()
    failures_df.to_csv(run_root / "failures.csv", index=False)
    results_df.to_csv(run_root / "results.csv", index=False)

    summary = summarize_results(results_df, preview_df, failures_df)
    plot_paths = {
        "scatter_svg": write_scatter_svg(
            results_df,
            output_path=plots_dir / "length_vs_rmsd.svg",
            regression=summary.get("linear_regression"),  # type: ignore[arg-type]
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
            "preview": preview_df.to_dict(orient="records"),
        },
    )

    return BenchmarkRunResult(
        run_name=run_name,
        run_root=run_root,
        preview_df=preview_df,
        results_df=results_df,
        failures_df=failures_df,
        summary=summary,
        summary_path=summary_path,
        results_csv_path=run_root / "results.csv",
        results_json_path=results_json_path,
        failures_csv_path=run_root / "failures.csv",
        plot_paths=plot_paths,
    )
