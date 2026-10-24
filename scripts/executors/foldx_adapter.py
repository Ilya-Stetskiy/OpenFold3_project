#!/usr/bin/env python3
"""FoldX adapter helpers for the current_internal experimental executor."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def load_foldx_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing FoldX config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"FoldX config must be a mapping: {path}")
    config.setdefault("foldx", {})
    config.setdefault("paths", {})
    return config


def validate_foldx_binary(binary_path: str | Path) -> tuple[bool, str]:
    if not binary_path:
        return False, "binary_path is empty"
    path = Path(binary_path)
    if not path.exists():
        return False, f"binary does not exist: {path}"
    if not path.is_file():
        return False, f"binary is not a file: {path}"
    if not os.access(path, os.X_OK):
        return False, f"binary is not executable: {path}"
    return True, ""


def build_foldx_mutation_code(chain_id: str, mutation: str) -> str:
    if not chain_id:
        raise ValueError("chain_id is required")
    match = MUTATION_RE.match(str(mutation).strip().upper())
    if not match:
        raise ValueError(f"Invalid single-site mutation: {mutation}")
    wild_type, position, mutant = match.groups()
    return f"{wild_type}{chain_id.strip()}{position}{mutant};"


def write_individual_list(job_dir: Path | str, chain_id: str, mutation: str) -> Path:
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "individual_list.txt"
    path.write_text(build_foldx_mutation_code(chain_id, mutation) + "\n", encoding="utf-8")
    return path


def build_repair_pdb_command(binary_path: str | Path, input_pdb_name: str) -> list[str]:
    return [str(binary_path), "--command=RepairPDB", f"--pdb={input_pdb_name}"]


def build_buildmodel_command(binary_path: str | Path, repaired_pdb_name: str, individual_list_name: str, number_of_runs: int) -> list[str]:
    return [
        str(binary_path),
        "--command=BuildModel",
        f"--pdb={repaired_pdb_name}",
        f"--mutant-file={individual_list_name}",
        f"--numberOfRuns={number_of_runs}",
    ]


def detect_foldx_outputs(job_dir: Path | str) -> dict[str, list[str]]:
    job_dir = Path(job_dir)
    candidates = []
    for root in [job_dir, job_dir / "raw", job_dir / "foldx_work"]:
        if root.exists():
            candidates.extend(path for path in root.rglob("*") if path.is_file())
    return {
        "difference_files": sorted(_rel(path, job_dir) for path in candidates if path.name.lower().startswith("dif_")),
        "average_files": sorted(_rel(path, job_dir) for path in candidates if "average" in path.name.lower()),
        "raw_files": sorted(_rel(path, job_dir) for path in candidates),
    }


def parse_foldx_ddg(job_dir: Path | str) -> dict[str, Any]:
    job_dir = Path(job_dir)
    outputs = detect_foldx_outputs(job_dir)
    search_order = outputs["difference_files"] + outputs["average_files"] + outputs["raw_files"]
    seen: set[str] = set()
    errors: list[str] = []
    for relative in search_order:
        if relative in seen:
            continue
        seen.add(relative)
        path = job_dir / relative
        try:
            value = _parse_ddg_from_text(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            errors.append(f"{relative}: {exc}")
            continue
        if value is not None:
            return {
                "foldx_ddg": value,
                "foldx_unit": "kcal/mol",
                "raw_output_files": outputs["raw_files"],
                "parser_status": "success",
                "error_message": "",
            }
    return {
        "foldx_ddg": None,
        "foldx_unit": "kcal/mol",
        "raw_output_files": outputs["raw_files"],
        "parser_status": "parse_failed",
        "error_message": "; ".join(errors) or "No parseable FoldX ddG value found",
    }


def validate_foldx_job_inputs(job: dict[str, Any], root: Path | None = None, structure_path_field: str = "experimental_structure_path") -> tuple[bool, list[str]]:
    errors = []
    if not job.get("chain_id"):
        errors.append("missing_chain_id")
    if not job.get(structure_path_field):
        errors.append(f"missing_{structure_path_field}")
    elif not _resolve_path(str(job[structure_path_field]), root).exists():
        errors.append(f"{structure_path_field}_not_found")
    try:
        build_foldx_mutation_code(str(job.get("chain_id", "")), str(job.get("mutation", "")))
    except ValueError as exc:
        errors.append(str(exc))
    return not errors, errors


def _resolve_path(path: str, root: Path | None) -> Path:
    resolved = Path(path)
    if resolved.is_absolute() or root is None:
        return resolved
    return root / resolved


def parsed_result_payload(dataset: str, job: dict[str, Any], parse_result: dict[str, Any], number_of_runs: int) -> dict[str, Any]:
    return {
        "dataset_name": dataset,
        "method": "foldx",
        "structure_source": job.get("structure_source", "experimental"),
        "mutation_record_id": job.get("mutation_record_id", ""),
        "job_id": job.get("job_id", ""),
        "mutation": job.get("mutation", ""),
        "chain_id": job.get("chain_id", ""),
        "foldx_ddg": parse_result.get("foldx_ddg"),
        "foldx_unit": "kcal/mol",
        "number_of_runs": number_of_runs,
        "raw_output_files": parse_result.get("raw_output_files", []),
        "parser_status": parse_result.get("parser_status", ""),
        "error_message": parse_result.get("error_message", ""),
    }


def write_parsed_result(path: Path | str, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _parse_ddg_from_text(text: str) -> float | None:
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    for line in lines:
        parts = re.split(r"\t+|\s{2,}", line)
        lowered = [part.lower() for part in parts]
        if any("ddg" in part or "total energy" in part for part in lowered):
            continue
        numeric = [float(match.group(0)) for match in FLOAT_RE.finditer(line)]
        if numeric:
            return numeric[-1]
    for line in lines:
        if "ddg" in line.lower() or "total energy" in line.lower():
            numeric = [float(match.group(0)) for match in FLOAT_RE.finditer(line)]
            if numeric:
                return numeric[-1]
    return None


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
