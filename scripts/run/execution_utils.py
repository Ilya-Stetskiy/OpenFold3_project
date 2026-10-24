#!/usr/bin/env python3
"""Reusable execution contract utilities for DDG stage executors."""

from __future__ import annotations

import csv
import hashlib
import json
import socket
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "configs" / "execution_contract.yaml"

STAGE_FILTERS = {
    "foldx_experimental": {"method": "foldx", "structure_source": "experimental", "input_ready": "true"},
    "rosetta_experimental": {"method": "rosetta", "structure_source": "experimental", "input_ready": "true"},
    "esm2_cpu_optional": {"method": "esm2", "structure_source": "sequence_only", "input_ready": "true"},
    "esm2_gpu_optional": {"method": "esm2", "structure_source": "sequence_only", "input_ready": "true"},
    "foldx_openfold": {"method": "foldx", "structure_source": "openfold", "input_ready": "true"},
    "rosetta_openfold": {"method": "rosetta", "structure_source": "openfold", "input_ready": "true"},
}


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv_dicts(path: Path | str) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _refuse_existing(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file without --force: {path}")


def write_csv_dicts(path: Path | str, rows: list[dict[str, Any]], force: bool = False) -> None:
    path = Path(path)
    _refuse_existing(path, force)
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def write_json(path: Path | str, data: Any, force: bool = False) -> None:
    path = Path(path)
    _refuse_existing(path, force)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def safe_write_text(path: Path | str, text: str, force: bool = False) -> None:
    path = Path(path)
    _refuse_existing(path, force)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def build_job_output_dir(dataset: str, method: str, structure_source: str, mutation_record_id: str, root: Path | None = None) -> Path:
    root = root or ROOT
    return root / "results" / dataset / method / structure_source / mutation_record_id


def build_job_manifest(
    *,
    dataset_name: str,
    stage_group: str,
    stage_name: str,
    job: dict[str, Any],
    job_status: str = "pending",
    input_files: list[str] | None = None,
    output_files: list[str] | None = None,
    parsed_result_path: str = "",
    error_message: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "stage_group": stage_group,
        "stage_name": stage_name,
        "job_id": job.get("job_id", ""),
        "mutation_record_id": job.get("mutation_record_id", ""),
        "method": job.get("method", ""),
        "structure_source": job.get("structure_source", ""),
        "input_ready": _truthy(job.get("input_ready", False)),
        "job_status": job_status,
        "started_at": "",
        "finished_at": "",
        "runtime_seconds": "",
        "input_files": input_files or [],
        "output_files": output_files or [],
        "parsed_result_path": parsed_result_path,
        "error_message": error_message,
        "notes": notes,
    }


def write_job_manifest(job_dir: Path | str, manifest: dict[str, Any], force: bool = False) -> Path:
    path = Path(job_dir) / "job_manifest.json"
    write_json(path, manifest, force=force)
    return path


def write_stage_manifest(dataset: str, stage_name: str, rows: list[dict[str, Any]], force: bool = False, root: Path | None = None) -> tuple[Path, Path]:
    root = root or ROOT
    csv_path = root / "results" / dataset / "_manifests" / f"{stage_name}_jobs.csv"
    json_path = root / "results" / dataset / "_manifests" / f"{stage_name}_jobs.json"
    write_csv_dicts(csv_path, rows, force=force)
    write_json(json_path, {"stage_name": stage_name, "jobs": rows}, force=force)
    return csv_path, json_path


def create_completion_marker(
    *,
    dataset_name: str,
    stage_group: str,
    status: str,
    outputs: list[str],
    summary: dict[str, Any],
    command: str,
    notes: str = "",
    force: bool = False,
    root: Path | None = None,
) -> Path:
    root = root or ROOT
    marker = root / "data" / "run_state" / dataset_name / f"{stage_group}.completed.json"
    payload = {
        "dataset_name": dataset_name,
        "stage_group": stage_group,
        "status": status,
        "started_at": "",
        "completed_at": utc_now_iso(),
        "host": socket.gethostname(),
        "command": command,
        "outputs": outputs,
        "summary": summary,
        "notes": notes,
    }
    write_json(marker, payload, force=force)
    return marker


def read_completion_marker(dataset_name: str, stage_group: str, root: Path | None = None) -> dict[str, Any] | None:
    root = root or ROOT
    marker = root / "data" / "run_state" / dataset_name / f"{stage_group}.completed.json"
    if not marker.exists():
        return None
    return json.loads(marker.read_text(encoding="utf-8"))


def summarize_job_statuses(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("job_status", "") or "unknown") for row in rows))


def filter_jobs_for_stage(rows: list[dict[str, Any]], stage_name: str) -> list[dict[str, Any]]:
    criteria = STAGE_FILTERS.get(stage_name)
    if not criteria:
        return []
    selected = []
    for row in rows:
        if row.get("method") != criteria["method"]:
            continue
        if row.get("structure_source") != criteria["structure_source"]:
            continue
        if _truthy(row.get("input_ready", False)) != _truthy(criteria["input_ready"]):
            continue
        selected.append(row)
    return selected


def stage_group_for_stage(run_stages: dict[str, Any], stage_name: str) -> str:
    for stage in run_stages.get("stages", []):
        if stage.get("stage_name") == stage_name:
            return str(stage.get("stage_group", ""))
    return ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _csv_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)

