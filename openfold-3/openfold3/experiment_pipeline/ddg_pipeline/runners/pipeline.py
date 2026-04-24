from __future__ import annotations

import csv
import json
import math
import platform
import random
import traceback
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..adapters import DDGAdapter
from ..adapters.base import AdapterResult
from ..data.canonical import CanonicalMutationRecord, write_canonical_records


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    output_root: Path
    methods: tuple[str, ...]
    structure_sources: tuple[str, ...] = ("experimental", "openfold", "foldx")
    seed: int = 17


def ensure_layout(root: str | Path) -> dict[str, Path]:
    root_path = Path(root).expanduser().resolve()
    layout = {
        name: root_path / name
        for name in ("config", "data", "structures", "adapters", "runners", "outputs", "logs", "tests")
    }
    root_path.mkdir(parents=True, exist_ok=True)
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def layout_tree(root: str | Path) -> str:
    root_path = Path(root).expanduser().resolve()
    lines = [root_path.name + "/"]
    for name in ("config", "data", "structures", "adapters", "runners", "outputs", "logs", "tests"):
        lines.append(f"├── {name}/")
    return "\n".join(lines)


def _safe_float(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank_value = (index + end - 1) / 2.0 + 1.0
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = rank_value
        index = end
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    if var_x == 0.0 or var_y == 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None] | None:
    paired = [
        (float(row["ddg"]), float(row["experimental_ddg"]))
        for row in rows
        if row.get("status") == "ok"
        and row.get("unit") == "kcal/mol"
        and row.get("experimental_ddg") is not None
        and not math.isnan(float(row["ddg"]))
    ]
    if len(paired) < 5:
        return None
    predicted = [left for left, _ in paired]
    experimental = [right for _, right in paired]
    pearson = _pearson(predicted, experimental)
    spearman = _pearson(_rank(predicted), _rank(experimental))
    rmse = math.sqrt(sum((pred - exp) ** 2 for pred, exp in paired) / len(paired))
    return {"pearson": pearson, "spearman": spearman, "rmse": rmse}


def _method_versions(adapters: list[DDGAdapter]) -> dict[str, dict[str, Any]]:
    return {
        adapter.method_name: {
            "class": type(adapter).__name__,
            "module": type(adapter).__module__,
        }
        for adapter in adapters
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _results_fieldnames() -> list[str]:
    return [
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
    ]


def _write_outputs(
    layout: dict[str, Path],
    records: list[CanonicalMutationRecord],
    results_rows: list[dict[str, Any]],
    structure_validation_rows: list[dict[str, Any]],
    adapters: list[DDGAdapter],
    config: PipelineConfig,
) -> None:
    _write_json(layout["config"] / "config.json", {
        "output_root": str(config.output_root),
        "methods": list(config.methods),
        "structure_sources": list(config.structure_sources),
        "seed": config.seed,
    })
    _write_json(layout["config"] / "reproducibility.json", {
        "seed": config.seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "method_versions": _method_versions(adapters),
    })
    write_canonical_records(layout["data"] / "canonical_records.json", records)
    _write_csv(
        layout["outputs"] / "results.csv",
        [
            {
                "protein_id": row["protein_id"],
                "mutation": row["mutation"],
                "representation": row["representation"],
                "structure_source": row["structure_source"],
                "method": row["method"],
                "unit": row["unit"],
                "ddg": row["ddg"],
                "ddg_raw": row["ddg_raw"],
                "ddg_std": row["ddg_std"],
                "n_runs_requested": row["n_runs_requested"],
                "n_runs_valid": row["n_runs_valid"],
                "status": row["status"],
                "error_message": row["error_message"],
            }
            for row in results_rows
        ],
        _results_fieldnames(),
    )
    _write_csv(
        layout["structures"] / "structure_validation.csv",
        structure_validation_rows,
        ["protein_id", "mutation", "structure_source", "structure_path", "state"],
    )
    metrics = compute_metrics(results_rows)
    if metrics is None:
        successful_predictions = sum(1 for row in results_rows if row["status"] == "ok")
        _write_json(
            layout["outputs"] / "metrics.json",
            {
                "skipped": True,
                "reason": "fewer_than_5_successful_predictions",
                "successful_predictions": successful_predictions,
            },
        )
    else:
        _write_json(layout["outputs"] / "metrics.json", metrics)


def _selected_adapters(adapters: list[DDGAdapter], methods: tuple[str, ...]) -> list[DDGAdapter]:
    adapter_map = {adapter.method_name: adapter for adapter in adapters}
    missing = [method for method in methods if method not in adapter_map]
    if missing:
        raise ValueError(f"Unsupported methods requested: {', '.join(missing)}")
    return [adapter_map[method] for method in methods]


def _structure_sources_for_adapter(adapter: DDGAdapter, config: PipelineConfig) -> tuple[str, ...]:
    return ("sequence_only",) if adapter.sequence_based else config.structure_sources


def run_pipeline(records: list[CanonicalMutationRecord], adapters: list[DDGAdapter], config: PipelineConfig) -> dict[str, Path]:
    random.seed(config.seed)
    layout = ensure_layout(config.output_root)
    print(layout_tree(config.output_root))
    log_path = layout["logs"] / "pipeline.log"
    log_path.write_text("", encoding="utf-8")

    selected_adapters = _selected_adapters(adapters, config.methods)
    for adapter in selected_adapters:
        adapter.validate_environment()

    results_rows: list[dict[str, Any]] = []
    structure_validation_rows: list[dict[str, Any]] = []
    success_count = 0
    execution_trace_sections: list[str] = []
    model_pairing_payload: list[dict[str, Any]] = []
    rosetta_environment_path: Path | None = None
    rosetta_protocol_path: Path | None = None
    esm_environment_path: Path | None = None
    esm_model_info_path: Path | None = None

    try:
        for record in records:
            for adapter in selected_adapters:
                for structure_source in _structure_sources_for_adapter(adapter, config):
                    structure_path = None if structure_source == "sequence_only" else getattr(record.structure_paths, structure_source)
                    structure_validation_rows.append({
                        "protein_id": record.protein_id,
                        "mutation": record.mutation,
                        "structure_source": structure_source,
                        "structure_path": structure_path,
                        "state": "present" if structure_path else "missing",
                    })
                    method_dir = layout["adapters"] / record.protein_id / record.mutation / structure_source / adapter.method_name
                    try:
                        result = adapter.predict(record, structure_source, method_dir)
                    except Exception as exc:  # noqa: BLE001
                        tb = traceback.format_exc()
                        method_dir.mkdir(parents=True, exist_ok=True)
                        (method_dir / "traceback.txt").write_text(tb, encoding="utf-8")
                        with log_path.open("a", encoding="utf-8") as handle:
                            handle.write(tb + "\n")
                        result = AdapterResult(
                            method=adapter.method_name,
                            representation=adapter.representation(),
                            raw_value=None,
                            ddg=None,
                            ddg_std=None,
                            n_runs_requested=None,
                            n_runs_valid=None,
                            per_run_ddg=(),
                            success=False,
                            status="failed",
                            error_message=f"{type(exc).__name__}: {exc}",
                            units=None,
                            structure_source=structure_source,
                            input_path=method_dir / "input.json",
                            raw_output_path=method_dir / "raw_output.json",
                            normalized_output_path=method_dir / "normalized_result.json",
                        )

                    results_rows.append({
                        "protein_id": record.protein_id,
                        "mutation": record.mutation,
                        "representation": result.representation,
                        "method": adapter.method_name,
                        "structure_source": structure_source,
                        "ddg_raw": _safe_float(result.raw_value),
                        "ddg": _safe_float(result.ddg),
                        "ddg_std": _safe_float(result.ddg_std),
                        "n_runs_requested": result.n_runs_requested,
                        "n_runs_valid": result.n_runs_valid,
                        "status": result.status,
                        "error_message": result.error_message,
                        "unit": result.units,
                        "experimental_ddg": record.experimental_ddg,
                    })
                    if adapter.method_name == "foldx" and result.raw_output_path.exists():
                        raw_payload = json.loads(result.raw_output_path.read_text(encoding="utf-8"))
                        trace_path = raw_payload.get("execution_trace_path")
                        if trace_path:
                            execution_trace_sections.append(Path(str(trace_path)).read_text(encoding="utf-8"))
                        model_pairing_payload.append(
                            {
                                "protein_id": record.protein_id,
                                "mutation": record.mutation,
                                "structure_source": structure_source,
                                "method": adapter.method_name,
                                "model_pairing": raw_payload.get("model_pairing", []),
                            }
                        )
                    if adapter.method_name == "rosetta" and result.raw_output_path.exists():
                        raw_payload = json.loads(result.raw_output_path.read_text(encoding="utf-8"))
                        environment_path = raw_payload.get("environment_check_path")
                        protocol_path = raw_payload.get("protocol_path")
                        if environment_path and rosetta_environment_path is None:
                            rosetta_environment_path = Path(str(environment_path))
                        if protocol_path and rosetta_protocol_path is None:
                            rosetta_protocol_path = Path(str(protocol_path))
                    if adapter.method_name == "esm2" and result.raw_output_path.exists():
                        raw_payload = json.loads(result.raw_output_path.read_text(encoding="utf-8"))
                        environment_path = raw_payload.get("environment_check_path")
                        model_info_path = raw_payload.get("model_info_path")
                        if environment_path and esm_environment_path is None:
                            esm_environment_path = Path(str(environment_path))
                        if model_info_path and esm_model_info_path is None:
                            esm_model_info_path = Path(str(model_info_path))
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "protein_id": record.protein_id,
                                    "mutation": record.mutation,
                                    "method": adapter.method_name,
                                    "structure_source": structure_source,
                                    "status": result.status,
                                    "input_path": str(result.input_path),
                                    "raw_output_path": str(result.raw_output_path),
                                    "normalized_output_path": str(result.normalized_output_path),
                                    "error_message": result.error_message,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    if result.status == "ok":
                        success_count += 1
                    else:
                        raise RuntimeError(
                            f"Adapter {adapter.method_name} failed for {record.protein_id} {record.mutation} "
                            f"({structure_source}): {result.error_message or result.status}"
                        )
    finally:
        _write_outputs(layout, records, results_rows, structure_validation_rows, selected_adapters, config)
        if model_pairing_payload:
            _write_json(layout["outputs"] / "foldx_model_pairing.json", model_pairing_payload)
        if execution_trace_sections:
            (layout["outputs"] / "foldx_execution_trace.md").write_text(
                "\n\n---\n\n".join(execution_trace_sections),
                encoding="utf-8",
            )
        if rosetta_environment_path and rosetta_environment_path.exists():
            shutil.copy2(rosetta_environment_path, layout["outputs"] / "rosetta_environment_check.json")
        if rosetta_protocol_path and rosetta_protocol_path.exists():
            shutil.copy2(rosetta_protocol_path, layout["outputs"] / "rosetta_protocol.md")
        if esm_environment_path and esm_environment_path.exists():
            shutil.copy2(esm_environment_path, layout["outputs"] / "esm_environment_check.json")
        if esm_model_info_path and esm_model_info_path.exists():
            shutil.copy2(esm_model_info_path, layout["outputs"] / "esm_model_info.json")

    if success_count == 0:
        raise RuntimeError("No valid ddG predictions")

    metrics = compute_metrics(results_rows)
    if metrics is not None:
        print(f"Pearson: {metrics['pearson']}")
        print(f"Spearman: {metrics['spearman']}")
        print(f"RMSE: {metrics['rmse']}")
    banner = " + ".join(method.upper() for method in config.methods) + " DDG PIPELINE VALIDATED"
    print(banner)
    return {key: path for key, path in layout.items()}
