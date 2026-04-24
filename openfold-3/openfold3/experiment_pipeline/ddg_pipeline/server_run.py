from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from openfold3.benchmark.structure_source import extract_protein_sequence

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


def _openfold_paths_by_case(structure_results_csv: Path | None) -> dict[str, str]:
    if structure_results_csv is None:
        return {}
    if not structure_results_csv.exists():
        raise FileNotFoundError(f"OpenFold structure results file does not exist: {structure_results_csv}")
    frame = pd.read_csv(structure_results_csv)
    required = {"case_id", "openfold3_status", "openfold3_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OpenFold structure results are missing columns: {', '.join(sorted(missing))}")
    mapping: dict[str, str] = {}
    for row in frame.to_dict(orient="records"):
        if str(row.get("openfold3_status", "")).strip() != "success" and str(row.get("openfold3_status", "")).strip() != "ok":
            continue
        path = str(row.get("openfold3_path", "")).strip()
        if path and Path(path).exists():
            mapping[str(row["case_id"])] = path
    return mapping


def build_structure_dataset(
    processed_csv: Path,
    output_csv: Path,
    rejected_csv: Path,
    *,
    max_records: int | None = None,
    require_position_residue_id_match: bool = True,
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
            sequence = extract_protein_sequence(pdb_path, str(record["chain"]))
            if position < 1 or position > len(sequence):
                raise ValueError("position_outside_sequence")
            if sequence[position - 1] != wt:
                raise ValueError("sequence_wt_mismatch")
            row = dict(record)
            row["sequence"] = sequence
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            rejected.append({**record, "reject_reason": reason})
        if max_records is not None and len(rows) >= max_records:
            break

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    rejected_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rejected).to_csv(rejected_csv, index=False)
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
    rows: list[dict[str, Any]] = []
    for root in input_roots:
        results_path = root / "results.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing structure results: {results_path}")
        frame = pd.read_csv(results_path)
        rows.extend(frame.to_dict(orient="records"))
    rows = sorted(rows, key=lambda row: str(row.get("case_id", "")))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


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
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    )


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
                try:
                    result = adapter.predict(record, structure_source, method_dir)
                    row = _row_from_result(record, adapter, result)
                    if result.status != "ok" and config.fail_fast:
                        raise RuntimeError(result.error_message or result.status)
                except Exception as exc:  # noqa: BLE001
                    method_dir.mkdir(parents=True, exist_ok=True)
                    (method_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    row = _failed_row(record, adapter, structure_source, method_dir, exc)
                    if config.fail_fast:
                        _write_json(row_path, row)
                        raise
                _write_json(row_path, row)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                _write_server_outputs(config.output_root, records, _collect_rows(rows_dir), config)

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
                    "python -m openfold3.experiment_pipeline.structure.cli",
                    f"--dataset {structure_shard_dir / f'shard_{shard_index:04d}.csv'}",
                    f"--output-dir {structure_output / f'shard_{shard_index:04d}'}",
                    "--backend openfold3",
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


def _parse_methods(text: str) -> tuple[str, ...]:
    return tuple(method.strip() for method in text.split(",") if method.strip())


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
    plan.add_argument("--openfold-python", default="python")
    plan.add_argument("--openfold-runner-yaml", type=Path, default=None)
    plan.add_argument("--openfold-inference-ckpt-path", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-structure-dataset":
        rows = build_structure_dataset(
            args.processed_csv,
            args.output_csv,
            args.rejected_csv,
            max_records=args.max_records,
            require_position_residue_id_match=not args.allow_position_residue_mismatch,
        )
        print(f"Prepared structure dataset rows: {len(rows)}")
        print(f"Output: {args.output_csv}")
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
                structure_sources=_parse_methods(args.structure_sources),
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                resume=not args.no_resume,
                fail_fast=bool(args.fail_fast),
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
            openfold_python=args.openfold_python,
            openfold_runner_yaml=args.openfold_runner_yaml,
            openfold_ckpt=args.openfold_inference_ckpt_path,
        )
        print(f"Server command plan: {commands_path}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
