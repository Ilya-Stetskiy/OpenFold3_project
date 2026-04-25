from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - server environments normally provide tqdm
    class _TqdmFallback:
        def __init__(self, iterable, **_kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, **_kwargs) -> None:
            return None

    def tqdm(iterable, **kwargs):
        return _TqdmFallback(iterable, **kwargs)

from openfold3.benchmark.structure_source import extract_protein_sequence
from openfold3.experiment_pipeline.structure.backends.openfold_backend import _select_output_cif
from openfold3.experiment_pipeline.structure.manifest import manifest_entry_from_result, save_manifest
from openfold3.experiment_pipeline.structure.models import BackendResult, MutationCase

from .adapters import DDGAdapter, ESM2Adapter, FoldXAdapter, RosettaDDGAdapter
from .adapters.base import AdapterResult
from .data.canonical import CanonicalMutationRecord, load_canonical_records, parse_record, write_canonical_records
from .runners.pipeline import PipelineConfig, compute_metrics, ensure_layout


RESULT_FIELDS = [
    "protein_id",
    "mutation",
    "representation",
    "structure_source",
    "method",
    "unit",
    "ddg",
    "ddg_raw",
    "ddg_std",
    "n_runs_requested",
    "n_runs_valid",
    "status",
    "error_message",
    "experimental_ddg",
    "raw_output_path",
]


@dataclass(frozen=True, slots=True)
class ServerRunConfig:
    output_root: Path
    methods: tuple[str, ...]
    structure_sources: tuple[str, ...]
    shard_index: int = 0
    shard_count: int = 1
    seed: int = 17
    resume: bool = True
    fail_fast: bool = False
    parallel_jobs: int = 1


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() in {"", "nan", "None"}


def _selected_records(records: Sequence[CanonicalMutationRecord], shard_index: int, shard_count: int) -> list[CanonicalMutationRecord]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    return [record for index, record in enumerate(records) if index % shard_count == shard_index]


def _case_key(record: CanonicalMutationRecord, method: str, structure_source: str) -> str:
    safe_parts = [
        record.protein_id.replace("/", "_"),
        record.mutation.replace("/", "_"),
        structure_source.replace("/", "_"),
        method.replace("/", "_"),
    ]
    return "__".join(safe_parts)


def _structure_sources_for_adapter(adapter: DDGAdapter, config: ServerRunConfig) -> tuple[str, ...]:
    return ("sequence_only",) if adapter.sequence_based else config.structure_sources


def _structure_paths_by_case(structure_results_csv: Path | None, column: str, status_column: str) -> dict[str, str]:
    if structure_results_csv is None:
        return {}
    if not structure_results_csv.exists():
        raise FileNotFoundError(f"Structure results file does not exist: {structure_results_csv}")
    frame = pd.read_csv(structure_results_csv)
    required = {"case_id", status_column, column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Structure results are missing columns: {', '.join(sorted(missing))}")
    mapping: dict[str, str] = {}
    for row in frame.to_dict(orient="records"):
        status = str(row.get(status_column, "")).strip()
        if status not in {"success", "ok"}:
            continue
        path = str(row.get(column, "")).strip()
        if path and Path(path).exists():
            mapping[str(row["case_id"])] = path
    return mapping


def _openfold_paths_by_case(structure_results_csv: Path | None) -> dict[str, str]:
    return _structure_paths_by_case(structure_results_csv, "openfold3_path", "openfold3_status")


def _sequence_from_processed_record(record: dict[str, Any], pdb_path: Path) -> tuple[str, str]:
    for column in ("sequence", "pdb_sequence"):
        value = record.get(column)
        if not _is_missing(value):
            sequence = str(value).strip().upper()
            if sequence:
                return sequence, column
    return extract_protein_sequence(pdb_path, str(record["chain"])), "pdb_path"


def _reject_reason_counts(rejected: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rejected:
        reason = str(row.get("reject_reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def build_structure_dataset(
    processed_csv: Path,
    output_csv: Path,
    rejected_csv: Path,
    *,
    max_records: int | None = None,
    require_position_residue_id_match: bool = False,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(
        processed_csv,
        dtype={"pdb_residue_id": str, "chain": str, "pdb_id": str, "protein_id": str},
    )
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        reason = ""
        try:
            position = int(record["position"])
            wt = str(record["wt_residue"]).upper()
            pdb_residue_id = str(record["pdb_residue_id"]).strip()
            if require_position_residue_id_match and pdb_residue_id != str(position):
                raise ValueError("position_does_not_match_pdb_residue_id")
            pdb_path = Path(str(record["pdb_path"])).expanduser().resolve()
            if not pdb_path.exists():
                raise FileNotFoundError(str(pdb_path))
            sequence, sequence_source = _sequence_from_processed_record(record, pdb_path)
            if position < 1 or position > len(sequence):
                raise ValueError("position_outside_sequence")
            if sequence[position - 1] != wt:
                raise ValueError(f"sequence_wt_mismatch:{sequence_source}")
            row = dict(record)
            row["sequence"] = sequence
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            rejected.append({**record, "reject_reason": reason})
        if max_records is not None and len(rows) >= max_records:
            break

    output_columns = list(frame.columns)
    if "sequence" not in output_columns:
        output_columns.append("sequence")
    rejected_columns = output_columns + ["reject_reason"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=output_columns).to_csv(output_csv, index=False)
    rejected_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rejected, columns=rejected_columns).to_csv(rejected_csv, index=False)
    return rows


def build_canonical_dataset(
    structure_csv: Path,
    output_json: Path,
    rejected_csv: Path,
    *,
    openfold_results_csv: Path | None = None,
    max_records: int | None = None,
) -> list[CanonicalMutationRecord]:
    frame = pd.read_csv(
        structure_csv,
        dtype={"pdb_residue_id": str, "chain": str, "pdb_id": str, "protein_id": str},
    )
    openfold_paths = _openfold_paths_by_case(openfold_results_csv)
    records: list[CanonicalMutationRecord] = []
    rejected: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        try:
            protein_id = str(row["protein_id"])
            mutation_id = str(row["mutation_id"])
            case_id = f"{protein_id}__{mutation_id}"
            payload = {
                "protein_id": f"{protein_id}_{row['pdb_id']}_{row['chain']}_{mutation_id}",
                "sequence": str(row["sequence"]).strip().upper(),
                "mutation": mutation_id,
                "position": int(row["position"]),
                "wt": str(row["wt_residue"]).upper(),
                "mut": str(row["mut_residue"]).upper(),
                "structure_paths": {
                    "experimental": str(Path(str(row["pdb_path"])).expanduser().resolve()),
                    "openfold": openfold_paths.get(case_id),
                    "foldx": None,
                },
                "experimental_ddg": None if _is_missing(row.get("experimental_ddg")) else float(row["experimental_ddg"]),
            }
            records.append(parse_record(payload))
        except Exception as exc:  # noqa: BLE001
            rejected.append({**row, "reject_reason": f"{type(exc).__name__}: {exc}"})
        if max_records is not None and len(records) >= max_records:
            break

    write_canonical_records(output_json, records)
    rejected_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rejected).to_csv(rejected_csv, index=False)
    return records


def split_csv(input_csv: Path, output_dir: Path, shard_count: int, *, prefix: str = "shard") -> list[Path]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    frame = pd.read_csv(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for shard_index in range(shard_count):
        shard = frame.iloc[[index for index in range(len(frame)) if index % shard_count == shard_index]]
        path = output_dir / f"{prefix}_{shard_index:04d}.csv"
        shard.to_csv(path, index=False)
        paths.append(path)
    return paths


def merge_structure_results(input_roots: list[Path], output_csv: Path) -> Path:
    rows_by_case: dict[str, dict[str, Any]] = {}
    for root in input_roots:
        results_path = root / "results.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing structure results: {results_path}")
        frame = pd.read_csv(results_path)
        for row in frame.to_dict(orient="records"):
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                continue
            merged = rows_by_case.setdefault(
                case_id,
                {
                    "case_id": case_id,
                    "openfold3_status": "",
                    "foldx_status": "",
                    "openfold3_path": "",
                    "foldx_path": "",
                },
            )
            for key in ("openfold3_status", "foldx_status", "openfold3_path", "foldx_path"):
                value = row.get(key, "")
                if not _is_missing(value):
                    merged[key] = value
    rows = sorted(rows_by_case.values(), key=lambda row: str(row.get("case_id", "")))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


def _openfold_query_payload(case: MutationCase, sequence: str) -> dict[str, object]:
    return {
        "chains": [
            {
                "molecule_type": "protein",
                "chain_ids": [case.chain],
                "sequence": sequence,
            }
        ]
    }


def _write_openfold_shard_results(output_root: Path, rows: list[dict[str, Any]]) -> Path:
    results_path = output_root / "results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        sorted(rows, key=lambda row: str(row.get("case_id", ""))),
        columns=["case_id", "openfold3_status", "foldx_status", "openfold3_path", "foldx_path"],
    ).to_csv(results_path, index=False)
    return results_path


def run_openfold_shard(
    structure_csv: Path,
    output_root: Path,
    *,
    openfold_python: str,
    openfold_runner_yaml: Path | None = None,
    openfold_ckpt: Path | None = None,
) -> Path:
    frame = pd.read_csv(
        structure_csv,
        dtype={"pdb_residue_id": str, "chain": str, "pdb_id": str, "protein_id": str},
    )
    cases = [MutationCase.from_row(row) for row in frame.to_dict(orient="records")]
    sequences = {case.case_id: str(row["sequence"]).strip().upper() for case, row in zip(cases, frame.to_dict(orient="records"), strict=True)}

    input_dir = output_root / "openfold3_batch" / "input"
    batch_output_dir = output_root / "openfold3_batch" / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    if batch_output_dir.exists():
        shutil.rmtree(batch_output_dir)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    query_json_path = input_dir / "query.json"
    query_json_path.write_text(
        json.dumps(
            {"queries": {case.case_id: _openfold_query_payload(case, sequences[case.case_id]) for case in cases}},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    command = [
        openfold_python,
        "-m",
        "openfold3.run_openfold",
        "predict",
        "--query-json",
        str(query_json_path),
        "--output-dir",
        str(batch_output_dir),
    ]
    if openfold_runner_yaml is not None:
        command.extend(["--runner-yaml", str(openfold_runner_yaml)])
    if openfold_ckpt is not None:
        command.extend(["--inference-ckpt-path", str(openfold_ckpt)])

    completed = subprocess.run(command, cwd=output_root, capture_output=True, text=True, check=False)
    log_path = output_root / "openfold3_batch" / "openfold_run.json"
    _write_json(
        log_path,
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "query_json_path": str(query_json_path),
            "batch_output_dir": str(batch_output_dir),
        },
    )

    rows: list[dict[str, Any]] = []
    if completed.returncode != 0:
        message = (
            f"OpenFold3 batch predict failed with return code {completed.returncode}: "
            f"{completed.stderr[-500:] or completed.stdout[-500:]}"
        )
        for case in cases:
            case_dir = output_root / "cases" / case.case_id
            backend_dir = case_dir / "openfold3"
            backend_dir.mkdir(parents=True, exist_ok=True)
            result = BackendResult("openfold3", "failed", backend_dir, (), message)
            save_manifest(case_dir / "manifest.json", {"openfold3": manifest_entry_from_result(result, "batch")})
            rows.append(
                {
                    "case_id": case.case_id,
                    "openfold3_status": "failed",
                    "foldx_status": "",
                    "openfold3_path": "",
                    "foldx_path": "",
                }
            )
        return _write_openfold_shard_results(output_root, rows)

    for case in tqdm(cases, desc="OpenFold batch outputs", unit="case", dynamic_ncols=True):
        case_dir = output_root / "cases" / case.case_id
        backend_dir = case_dir / "openfold3"
        output_dir = backend_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            query_output_dir = batch_output_dir / case.case_id
            candidates = [path for path in query_output_dir.rglob("*.cif") if path.is_file()]
            source_cif = _select_output_cif(candidates, batch_output_dir, query_id=case.case_id)
            model_cif_path = output_dir / "model.cif"
            shutil.copy2(source_cif, model_cif_path)
            result = BackendResult("openfold3", "ok", backend_dir, (model_cif_path,), "OpenFold3 batch predict completed")
            status = "success"
            structure_path = str(model_cif_path)
        except Exception as exc:  # noqa: BLE001
            result = BackendResult(
                "openfold3",
                "failed",
                backend_dir,
                (),
                f"OpenFold3 batch output selection failed: {type(exc).__name__}: {exc}",
            )
            status = "failed"
            structure_path = ""
        save_manifest(case_dir / "manifest.json", {"openfold3": manifest_entry_from_result(result, "batch")})
        rows.append(
            {
                "case_id": case.case_id,
                "openfold3_status": status,
                "foldx_status": "",
                "openfold3_path": structure_path,
                "foldx_path": "",
            }
        )

    return _write_openfold_shard_results(output_root, rows)


def build_adapters(methods: Iterable[str], *, foldx_runs: int, rosetta_runs: int, rosetta_top_k: int) -> list[DDGAdapter]:
    adapters: dict[str, DDGAdapter] = {
        "foldx": FoldXAdapter(number_of_runs=foldx_runs),
        "rosetta": RosettaDDGAdapter(number_of_runs=rosetta_runs, top_k=rosetta_top_k),
        "esm2": ESM2Adapter(),
    }
    requested = tuple(methods)
    missing = [method for method in requested if method not in adapters]
    if missing:
        raise ValueError(f"Unsupported methods requested: {', '.join(missing)}")
    return [adapters[method] for method in requested]


def _row_from_result(record: CanonicalMutationRecord, adapter: DDGAdapter, result: AdapterResult) -> dict[str, Any]:
    return {
        "protein_id": record.protein_id,
        "mutation": record.mutation,
        "representation": result.representation,
        "structure_source": result.structure_source,
        "method": adapter.method_name,
        "unit": result.units,
        "ddg": _safe_float(result.ddg),
        "ddg_raw": _safe_float(result.raw_value),
        "ddg_std": _safe_float(result.ddg_std),
        "n_runs_requested": result.n_runs_requested,
        "n_runs_valid": result.n_runs_valid,
        "status": result.status,
        "error_message": result.error_message,
        "experimental_ddg": record.experimental_ddg,
        "raw_output_path": str(result.raw_output_path),
    }


def _failed_row(
    record: CanonicalMutationRecord,
    adapter: DDGAdapter,
    structure_source: str,
    method_dir: Path,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "protein_id": record.protein_id,
        "mutation": record.mutation,
        "representation": adapter.representation(),
        "structure_source": structure_source,
        "method": adapter.method_name,
        "unit": None,
        "ddg": float("nan"),
        "ddg_raw": float("nan"),
        "ddg_std": float("nan"),
        "n_runs_requested": None,
        "n_runs_valid": None,
        "status": "failed",
        "error_message": f"{type(exc).__name__}: {exc}",
        "experimental_ddg": record.experimental_ddg,
        "raw_output_path": str(method_dir / "raw_output.json"),
    }


def _collect_rows(rows_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(rows_dir.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _write_server_outputs(output_root: Path, records: list[CanonicalMutationRecord], rows: list[dict[str, Any]], config: ServerRunConfig) -> None:
    layout = ensure_layout(output_root)
    _write_csv(layout["outputs"] / "results.csv", rows, RESULT_FIELDS)
    write_canonical_records(layout["data"] / "canonical_records.json", records)
    metrics = compute_metrics(rows)
    _write_json(
        layout["outputs"] / "metrics.json",
        {
            "skipped": metrics is None,
            "metrics": metrics,
            "successful_predictions": sum(row.get("status") == "ok" for row in rows),
            "successful_kcal_predictions": sum(
                row.get("status") == "ok" and row.get("unit") == "kcal/mol" for row in rows
            ),
        },
    )
    _write_json(
        layout["config"] / "server_run_config.json",
        {
            "output_root": str(config.output_root),
            "methods": list(config.methods),
            "structure_sources": list(config.structure_sources),
            "shard_index": config.shard_index,
            "shard_count": config.shard_count,
            "seed": config.seed,
            "resume": config.resume,
            "fail_fast": config.fail_fast,
            "parallel_jobs": config.parallel_jobs,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    )


DDGJob = tuple[CanonicalMutationRecord, DDGAdapter, str, Path, Path]


def _run_ddg_job(job: DDGJob) -> tuple[Path, dict[str, Any]]:
    record, adapter, structure_source, method_dir, row_path = job
    try:
        result = adapter.predict(record, structure_source, method_dir)
        row = _row_from_result(record, adapter, result)
    except Exception as exc:  # noqa: BLE001
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        row = _failed_row(record, adapter, structure_source, method_dir, exc)
    return row_path, row


def _can_run_ddg_job_in_parallel(job: DDGJob) -> bool:
    _record, adapter, _structure_source, _method_dir, _row_path = job
    return not adapter.sequence_based


def _persist_ddg_row(
    row_path: Path,
    row: dict[str, Any],
    *,
    log_path: Path,
    output_root: Path,
    records: list[CanonicalMutationRecord],
    rows_dir: Path,
    config: ServerRunConfig,
) -> None:
    _write_json(row_path, row)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    _write_server_outputs(output_root, records, _collect_rows(rows_dir), config)


def run_ddg_shard(records: list[CanonicalMutationRecord], adapters: list[DDGAdapter], config: ServerRunConfig) -> Path:
    layout = ensure_layout(config.output_root)
    shard_records = _selected_records(records, config.shard_index, config.shard_count)
    rows_dir = layout["outputs"] / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    log_path = layout["logs"] / f"shard_{config.shard_index:04d}.jsonl"
    selected_methods = set(config.methods)
    selected_adapters = [adapter for adapter in adapters if adapter.method_name in selected_methods]
    if len(selected_adapters) != len(selected_methods):
        available = {adapter.method_name for adapter in adapters}
        missing = sorted(selected_methods.difference(available))
        raise ValueError(f"Missing adapters for methods: {', '.join(missing)}")
    for adapter in selected_adapters:
        adapter.validate_environment()

    jobs: list[DDGJob] = []
    for record in shard_records:
        for adapter in selected_adapters:
            for structure_source in _structure_sources_for_adapter(adapter, config):
                if structure_source != "sequence_only" and getattr(record.structure_paths, structure_source) is None:
                    continue
                row_key = _case_key(record, adapter.method_name, structure_source)
                row_path = rows_dir / f"{row_key}.json"
                if config.resume and row_path.exists():
                    continue
                method_dir = layout["adapters"] / record.protein_id / record.mutation / structure_source / adapter.method_name
                jobs.append((record, adapter, structure_source, method_dir, row_path))

    parallel_jobs = max(1, config.parallel_jobs)
    parallelizable_jobs = [job for job in jobs if _can_run_ddg_job_in_parallel(job)]
    serial_jobs = [job for job in jobs if not _can_run_ddg_job_in_parallel(job)]
    if parallel_jobs > 1 and parallelizable_jobs:
        print(f"[DDG PARALLEL] running {len(parallelizable_jobs)} structure predictions with {parallel_jobs} workers")
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = {executor.submit(_run_ddg_job, job): job for job in parallelizable_jobs}
            progress = tqdm(as_completed(futures), total=len(futures), desc="ddG structure predictions", unit="prediction", dynamic_ncols=True)
            for future in progress:
                job = futures[future]
                record, adapter, structure_source, _method_dir, _row_path = job
                progress.set_postfix(
                    case=record.protein_id,
                    mutation=record.mutation,
                    method=adapter.method_name,
                    source=structure_source,
                )
                row_path, row = future.result()
                _persist_ddg_row(
                    row_path,
                    row,
                    log_path=log_path,
                    output_root=config.output_root,
                    records=records,
                    rows_dir=rows_dir,
                    config=config,
                )
                if config.fail_fast and row["status"] != "ok":
                    raise RuntimeError(row["error_message"] or row["status"])
    else:
        serial_jobs = parallelizable_jobs + serial_jobs

    progress = tqdm(serial_jobs, desc="ddG predictions", unit="prediction", dynamic_ncols=True)
    for job in progress:
        record, adapter, structure_source, _method_dir, _row_path = job
        progress.set_postfix(
            case=record.protein_id,
            mutation=record.mutation,
            method=adapter.method_name,
            source=structure_source,
        )
        row_path, row = _run_ddg_job(job)
        _persist_ddg_row(
            row_path,
            row,
            log_path=log_path,
            output_root=config.output_root,
            records=records,
            rows_dir=rows_dir,
            config=config,
        )
        if config.fail_fast and row["status"] != "ok":
            raise RuntimeError(row["error_message"] or row["status"])

    _write_server_outputs(config.output_root, records, _collect_rows(rows_dir), config)
    return layout["outputs"] / "results.csv"


def merge_results(output_roots: list[Path], merged_output_root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for root in output_roots:
        results_path = root / "outputs" / "results.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing shard results: {results_path}")
        with results_path.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    rows = sorted(rows, key=lambda row: (row["protein_id"], row["mutation"], row["method"], row["structure_source"]))
    layout = ensure_layout(merged_output_root)
    _write_csv(layout["outputs"] / "results.csv", rows, RESULT_FIELDS)
    _write_json(
        layout["outputs"] / "metrics.json",
        {
            "metrics": compute_metrics(rows),
            "successful_predictions": sum(row.get("status") == "ok" for row in rows),
        },
    )
    return layout["outputs"] / "results.csv"


def write_server_plan(
    output_root: Path,
    *,
    processed_csv: Path,
    shard_count: int,
    methods: tuple[str, ...],
    foldx_runs: int,
    rosetta_runs: int,
    rosetta_top_k: int,
    parallel_jobs: int,
    openfold_python: str,
    openfold_runner_yaml: Path | None,
    openfold_ckpt: Path | None,
) -> Path:
    commands_path = output_root / "server_commands.sh"
    structure_csv = output_root / "data" / "structure_dataset.csv"
    structure_shard_dir = output_root / "data" / "structure_shards"
    openfold_results_csv = output_root / "data" / "openfold_results.csv"
    canonical_json = output_root / "data" / "canonical_records.json"
    structure_output = output_root / "structure"
    ddg_output = output_root / "ddg"
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"python -m openfold3.experiment_pipeline.ddg_pipeline.server_run prepare-structure-dataset --processed-csv {processed_csv} --output-csv {structure_csv} --rejected-csv {output_root / 'data' / 'structure_rejected.csv'}",
        f"python -m openfold3.experiment_pipeline.ddg_pipeline.server_run split-csv --input-csv {structure_csv} --output-dir {structure_shard_dir} --shard-count {shard_count}",
    ]
    for shard_index in range(shard_count):
        commands.append(
            " ".join(
                [
                    "python -m openfold3.experiment_pipeline.ddg_pipeline.server_run run-openfold-shard",
                    f"--structure-csv {structure_shard_dir / f'shard_{shard_index:04d}.csv'}",
                    f"--output-root {structure_output / f'shard_{shard_index:04d}'}",
                    f"--openfold-python {openfold_python}",
                    *([] if openfold_runner_yaml is None else [f"--openfold-runner-yaml {openfold_runner_yaml}"]),
                    *([] if openfold_ckpt is None else [f"--openfold-inference-ckpt-path {openfold_ckpt}"]),
                ]
            )
        )
    structure_roots = " ".join(str(structure_output / f"shard_{index:04d}") for index in range(shard_count))
    commands.append(
        f"python -m openfold3.experiment_pipeline.ddg_pipeline.server_run merge-structure-results --output-csv {openfold_results_csv} --input-roots {structure_roots}"
    )
    commands.append(
        " ".join(
            [
                "python -m openfold3.experiment_pipeline.ddg_pipeline.server_run prepare-ddg-dataset",
                f"--structure-csv {structure_csv}",
                f"--output-json {canonical_json}",
                f"--rejected-csv {output_root / 'data' / 'ddg_rejected.csv'}",
                f"--openfold-results-csv {openfold_results_csv}",
            ]
        )
    )
    for shard_index in range(shard_count):
        commands.append(
            " ".join(
                [
                    "python -m openfold3.experiment_pipeline.ddg_pipeline.server_run run-ddg-shard",
                    f"--canonical-json {canonical_json}",
                    f"--output-root {ddg_output / f'shard_{shard_index:04d}'}",
                    f"--shard-index {shard_index}",
                    f"--shard-count {shard_count}",
                    f"--methods {','.join(methods)}",
                    f"--foldx-runs {foldx_runs}",
                    f"--rosetta-runs {rosetta_runs}",
                    f"--rosetta-top-k {rosetta_top_k}",
                    f"--parallel-jobs {parallel_jobs}",
                ]
            )
        )
    merge_roots = " ".join(str(ddg_output / f"shard_{index:04d}") for index in range(shard_count))
    commands.append(
        f"python -m openfold3.experiment_pipeline.ddg_pipeline.server_run merge-results --output-root {output_root / 'merged'} --input-roots {merge_roots}"
    )
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    commands_path.chmod(0o755)
    return commands_path


def _shell_join(parts: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def write_batched_server_plan(
    output_root: Path,
    *,
    processed_csv: Path,
    batch_count: int,
    methods: tuple[str, ...],
    structure_sources: tuple[str, ...],
    foldx_runs: int,
    rosetta_runs: int,
    rosetta_top_k: int,
    parallel_jobs: int,
    openfold_python: str,
    openfold_runner_yaml: Path | None,
    openfold_ckpt: Path | None,
) -> Path:
    commands_path = output_root / "server_batched_commands.sh"
    structure_csv = output_root / "data" / "structure_dataset.csv"
    structure_batch_dir = output_root / "data" / "structure_batches"
    batch_results_dir = output_root / "data" / "structure_batch_results"
    canonical_batch_dir = output_root / "data" / "canonical_batches"
    structure_output = output_root / "structure_batches"
    ddg_output = output_root / "ddg_batches"
    logs_dir = output_root / "logs"

    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        _shell_join(
            [
                "python",
                "-m",
                "openfold3.experiment_pipeline.ddg_pipeline.server_run",
                "prepare-structure-dataset",
                "--processed-csv",
                processed_csv,
                "--output-csv",
                structure_csv,
                "--rejected-csv",
                output_root / "data" / "structure_rejected.csv",
            ]
        ),
        _shell_join(
            [
                "python",
                "-m",
                "openfold3.experiment_pipeline.ddg_pipeline.server_run",
                "split-csv",
                "--input-csv",
                structure_csv,
                "--output-dir",
                structure_batch_dir,
                "--shard-count",
                batch_count,
            ]
        ),
        _shell_join(["mkdir", "-p", logs_dir, batch_results_dir, canonical_batch_dir]),
        "",
    ]

    for batch_index in range(batch_count):
        batch_name = f"batch_{batch_index:04d}"
        batch_csv = structure_batch_dir / f"shard_{batch_index:04d}.csv"
        openfold_root = structure_output / batch_name / "openfold3"
        foldx_root = structure_output / batch_name / "foldx"
        batch_results_csv = batch_results_dir / f"{batch_name}.csv"
        canonical_json = canonical_batch_dir / f"{batch_name}.json"
        ddg_root = ddg_output / batch_name

        openfold_command = [
            "python",
            "-m",
            "openfold3.experiment_pipeline.ddg_pipeline.server_run",
            "run-openfold-shard",
            "--structure-csv",
            batch_csv,
            "--output-root",
            openfold_root,
            "--openfold-python",
            openfold_python,
        ]
        if openfold_runner_yaml is not None:
            openfold_command.extend(["--openfold-runner-yaml", openfold_runner_yaml])
        if openfold_ckpt is not None:
            openfold_command.extend(["--openfold-inference-ckpt-path", openfold_ckpt])

        foldx_command = [
            "python",
            "-m",
            "openfold3.experiment_pipeline.structure.cli",
            "--dataset",
            batch_csv,
            "--output-dir",
            foldx_root,
            "--backend",
            "foldx",
        ]

        commands.extend(
            [
                f"echo '[BATCH {batch_index + 1}/{batch_count}] structure generation'",
                f"{_shell_join(openfold_command)} > {shlex.quote(str(logs_dir / f'{batch_name}_openfold3.log'))} 2>&1 &",
                "OPENFOLD_PID=$!",
                f"{_shell_join(foldx_command)} > {shlex.quote(str(logs_dir / f'{batch_name}_foldx.log'))} 2>&1 &",
                "FOLDX_PID=$!",
                "OPENFOLD_STATUS=0",
                "FOLDX_STATUS=0",
                'wait "$OPENFOLD_PID" || OPENFOLD_STATUS=$?',
                'wait "$FOLDX_PID" || FOLDX_STATUS=$?',
                'if [ "$OPENFOLD_STATUS" -ne 0 ] || [ "$FOLDX_STATUS" -ne 0 ]; then',
                f"  echo '[BATCH {batch_index + 1}/{batch_count}] structure generation failed' >&2",
                '  exit 1',
                "fi",
                _shell_join(
                    [
                        "python",
                        "-m",
                        "openfold3.experiment_pipeline.ddg_pipeline.server_run",
                        "merge-structure-results",
                        "--output-csv",
                        batch_results_csv,
                        "--input-roots",
                        openfold_root,
                        foldx_root,
                    ]
                ),
                _shell_join(
                    [
                        "python",
                        "-m",
                        "openfold3.experiment_pipeline.ddg_pipeline.server_run",
                        "prepare-ddg-dataset",
                        "--structure-csv",
                        batch_csv,
                        "--output-json",
                        canonical_json,
                        "--rejected-csv",
                        output_root / "data" / f"{batch_name}_ddg_rejected.csv",
                        "--openfold-results-csv",
                        batch_results_csv,
                    ]
                ),
                f"echo '[BATCH {batch_index + 1}/{batch_count}] ddG predictions'",
                _shell_join(
                    [
                        "python",
                        "-m",
                        "openfold3.experiment_pipeline.ddg_pipeline.server_run",
                        "run-ddg-shard",
                        "--canonical-json",
                        canonical_json,
                        "--output-root",
                        ddg_root,
                        "--shard-index",
                        0,
                        "--shard-count",
                        1,
                        "--methods",
                        ",".join(methods),
                        "--structure-sources",
                        ",".join(structure_sources),
                        "--foldx-runs",
                        foldx_runs,
                        "--rosetta-runs",
                        rosetta_runs,
                        "--rosetta-top-k",
                        rosetta_top_k,
                        "--parallel-jobs",
                        parallel_jobs,
                    ]
                ),
                "",
            ]
        )

    merge_roots = [ddg_output / f"batch_{index:04d}" for index in range(batch_count)]
    commands.append(
        _shell_join(
            [
                "python",
                "-m",
                "openfold3.experiment_pipeline.ddg_pipeline.server_run",
                "merge-results",
                "--output-root",
                output_root / "merged",
                "--input-roots",
                *merge_roots,
            ]
        )
    )

    commands_path.parent.mkdir(parents=True, exist_ok=True)
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    commands_path.chmod(0o755)
    return commands_path


def _parse_methods(text: str) -> tuple[str, ...]:
    return tuple(method.strip() for method in text.split(",") if method.strip())


def _parse_structure_sources(text: str) -> tuple[str, ...]:
    sources = _parse_methods(text)
    unsupported = sorted(set(sources).difference({"experimental", "openfold"}))
    if unsupported:
        raise ValueError(
            "Unsupported ddG structure sources: "
            + ", ".join(unsupported)
            + ". Use experimental/openfold; foldx outputs are mutant structures, not WT ddG inputs."
        )
    return sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Server-run orchestration for OpenFold -> ddG benchmark jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_structure = subparsers.add_parser("prepare-structure-dataset")
    prepare_structure.add_argument("--processed-csv", type=Path, required=True)
    prepare_structure.add_argument("--output-csv", type=Path, required=True)
    prepare_structure.add_argument("--rejected-csv", type=Path, required=True)
    prepare_structure.add_argument("--max-records", type=int, default=None)
    prepare_structure.add_argument("--allow-position-residue-mismatch", action="store_true")

    prepare_ddg = subparsers.add_parser("prepare-ddg-dataset")
    prepare_ddg.add_argument("--structure-csv", type=Path, required=True)
    prepare_ddg.add_argument("--output-json", type=Path, required=True)
    prepare_ddg.add_argument("--rejected-csv", type=Path, required=True)
    prepare_ddg.add_argument("--openfold-results-csv", type=Path, default=None)
    prepare_ddg.add_argument("--max-records", type=int, default=None)

    split = subparsers.add_parser("split-csv")
    split.add_argument("--input-csv", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--shard-count", type=int, required=True)

    openfold_shard = subparsers.add_parser("run-openfold-shard")
    openfold_shard.add_argument("--structure-csv", type=Path, required=True)
    openfold_shard.add_argument("--output-root", type=Path, required=True)
    openfold_shard.add_argument("--openfold-python", default="python")
    openfold_shard.add_argument("--openfold-runner-yaml", type=Path, default=None)
    openfold_shard.add_argument("--openfold-inference-ckpt-path", type=Path, default=None)

    merge_structure = subparsers.add_parser("merge-structure-results")
    merge_structure.add_argument("--output-csv", type=Path, required=True)
    merge_structure.add_argument("--input-roots", type=Path, nargs="+", required=True)

    run_shard = subparsers.add_parser("run-ddg-shard")
    run_shard.add_argument("--canonical-json", type=Path, required=True)
    run_shard.add_argument("--output-root", type=Path, required=True)
    run_shard.add_argument("--shard-index", type=int, required=True)
    run_shard.add_argument("--shard-count", type=int, required=True)
    run_shard.add_argument("--methods", default="foldx,rosetta,esm2")
    run_shard.add_argument("--structure-sources", default="experimental,openfold")
    run_shard.add_argument("--foldx-runs", type=int, default=5)
    run_shard.add_argument("--rosetta-runs", type=int, default=20)
    run_shard.add_argument("--rosetta-top-k", type=int, default=3)
    run_shard.add_argument("--parallel-jobs", type=int, default=1)
    run_shard.add_argument("--no-resume", action="store_true")
    run_shard.add_argument("--fail-fast", action="store_true")

    merge = subparsers.add_parser("merge-results")
    merge.add_argument("--output-root", type=Path, required=True)
    merge.add_argument("--input-roots", type=Path, nargs="+", required=True)

    plan = subparsers.add_parser("write-server-plan")
    plan.add_argument("--processed-csv", type=Path, required=True)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--shard-count", type=int, default=max(1, (os.cpu_count() or 1) // 8))
    plan.add_argument("--methods", default="foldx,rosetta,esm2")
    plan.add_argument("--foldx-runs", type=int, default=5)
    plan.add_argument("--rosetta-runs", type=int, default=20)
    plan.add_argument("--rosetta-top-k", type=int, default=3)
    plan.add_argument("--parallel-jobs", type=int, default=1)
    plan.add_argument("--openfold-python", default="python")
    plan.add_argument("--openfold-runner-yaml", type=Path, default=None)
    plan.add_argument("--openfold-inference-ckpt-path", type=Path, default=None)

    batched_plan = subparsers.add_parser("write-batched-server-plan")
    batched_plan.add_argument("--processed-csv", type=Path, required=True)
    batched_plan.add_argument("--output-root", type=Path, required=True)
    batched_plan.add_argument("--batch-count", type=int, default=max(1, (os.cpu_count() or 1) // 8))
    batched_plan.add_argument("--methods", default="foldx,rosetta,esm2")
    batched_plan.add_argument("--structure-sources", default="experimental,openfold")
    batched_plan.add_argument("--foldx-runs", type=int, default=5)
    batched_plan.add_argument("--rosetta-runs", type=int, default=20)
    batched_plan.add_argument("--rosetta-top-k", type=int, default=3)
    batched_plan.add_argument("--parallel-jobs", type=int, default=1)
    batched_plan.add_argument("--openfold-python", default="python")
    batched_plan.add_argument("--openfold-runner-yaml", type=Path, default=None)
    batched_plan.add_argument("--openfold-inference-ckpt-path", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-structure-dataset":
        rows = build_structure_dataset(
            args.processed_csv,
            args.output_csv,
            args.rejected_csv,
            max_records=args.max_records,
            require_position_residue_id_match=False,
        )
        print(f"Prepared structure dataset rows: {len(rows)}")
        print(f"Output: {args.output_csv}")
        if not rows:
            rejected = pd.read_csv(args.rejected_csv).to_dict(orient="records")
            reason_counts = _reject_reason_counts(rejected)
            print("Reject reasons:")
            for reason, count in list(reason_counts.items())[:10]:
                print(f"  {count}: {reason}")
            raise RuntimeError(f"No valid structure dataset rows; inspect {args.rejected_csv}")
        return 0
    if args.command == "prepare-ddg-dataset":
        records = build_canonical_dataset(
            args.structure_csv,
            args.output_json,
            args.rejected_csv,
            openfold_results_csv=args.openfold_results_csv,
            max_records=args.max_records,
        )
        print(f"Prepared canonical ddG records: {len(records)}")
        print(f"Output: {args.output_json}")
        return 0
    if args.command == "split-csv":
        paths = split_csv(args.input_csv, args.output_dir, args.shard_count)
        print(f"Wrote CSV shards: {len(paths)}")
        return 0
    if args.command == "run-openfold-shard":
        results_path = run_openfold_shard(
            args.structure_csv,
            args.output_root.resolve(),
            openfold_python=args.openfold_python,
            openfold_runner_yaml=args.openfold_runner_yaml,
            openfold_ckpt=args.openfold_inference_ckpt_path,
        )
        print(f"OpenFold shard results: {results_path}")
        return 0
    if args.command == "merge-structure-results":
        output_csv = merge_structure_results([root.resolve() for root in args.input_roots], args.output_csv.resolve())
        print(f"Merged structure results: {output_csv}")
        return 0
    if args.command == "run-ddg-shard":
        records = load_canonical_records(args.canonical_json)
        methods = _parse_methods(args.methods)
        adapters = build_adapters(
            methods,
            foldx_runs=args.foldx_runs,
            rosetta_runs=args.rosetta_runs,
            rosetta_top_k=args.rosetta_top_k,
        )
        results_path = run_ddg_shard(
            records,
            adapters,
            ServerRunConfig(
                output_root=args.output_root.resolve(),
                methods=methods,
                structure_sources=_parse_structure_sources(args.structure_sources),
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                resume=not args.no_resume,
                fail_fast=bool(args.fail_fast),
                parallel_jobs=args.parallel_jobs,
            ),
        )
        print(f"Shard results: {results_path}")
        return 0
    if args.command == "merge-results":
        results_path = merge_results([root.resolve() for root in args.input_roots], args.output_root.resolve())
        print(f"Merged results: {results_path}")
        return 0
    if args.command == "write-server-plan":
        commands_path = write_server_plan(
            args.output_root.resolve(),
            processed_csv=args.processed_csv.resolve(),
            shard_count=args.shard_count,
            methods=_parse_methods(args.methods),
            foldx_runs=args.foldx_runs,
            rosetta_runs=args.rosetta_runs,
            rosetta_top_k=args.rosetta_top_k,
            parallel_jobs=args.parallel_jobs,
            openfold_python=args.openfold_python,
            openfold_runner_yaml=args.openfold_runner_yaml,
            openfold_ckpt=args.openfold_inference_ckpt_path,
        )
        print(f"Server command plan: {commands_path}")
        return 0
    if args.command == "write-batched-server-plan":
        commands_path = write_batched_server_plan(
            args.output_root.resolve(),
            processed_csv=args.processed_csv.resolve(),
            batch_count=args.batch_count,
            methods=_parse_methods(args.methods),
            structure_sources=_parse_structure_sources(args.structure_sources),
            foldx_runs=args.foldx_runs,
            rosetta_runs=args.rosetta_runs,
            rosetta_top_k=args.rosetta_top_k,
            parallel_jobs=args.parallel_jobs,
            openfold_python=args.openfold_python,
            openfold_runner_yaml=args.openfold_runner_yaml,
            openfold_ckpt=args.openfold_inference_ckpt_path,
        )
        print(f"Batched server command plan: {commands_path}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
