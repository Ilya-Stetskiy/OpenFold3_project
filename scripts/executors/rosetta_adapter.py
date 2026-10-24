#!/usr/bin/env python3
"""Rosetta adapter helpers for the current_internal experimental executor."""

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


def load_rosetta_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing Rosetta config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Rosetta config must be a mapping: {path}")
    config.setdefault("rosetta", {})
    config.setdefault("paths", {})
    return config


def validate_rosetta_binary(binary_path: str | Path) -> tuple[bool, str]:
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


def parse_single_site_mutation(mutation: str) -> tuple[str, str, str]:
    match = MUTATION_RE.match(str(mutation).strip().upper())
    if not match:
        raise ValueError(f"Invalid single-site mutation: {mutation}")
    return match.groups()


def write_mutation_file(job_dir: Path | str, chain_id: str, mutation: str) -> Path:
    if not chain_id:
        raise ValueError("chain_id is required")
    wild_type, position, mutant = parse_single_site_mutation(mutation)
    path = Path(job_dir) / "mutation_file.txt"
    Path(job_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(f"total 1\n1\n{wild_type} {position} {mutant}\n", encoding="utf-8")
    return path


def build_rosetta_command(binary_path: str | Path, input_pdb_name: str, mutation_file_name: str, scorefile_name: str, number_of_runs: int, extra_args: list[str] | None = None) -> list[str]:
    command = [
        str(binary_path),
        "-s",
        input_pdb_name,
        "-mut_file",
        mutation_file_name,
        "-ddg::iterations",
        str(number_of_runs),
        "-scorefile",
        scorefile_name,
    ]
    command.extend(extra_args or [])
    return command


def detect_rosetta_outputs(job_dir: Path | str) -> dict[str, list[str]]:
    job_dir = Path(job_dir)
    candidates = []
    for root in [job_dir, job_dir / "raw", job_dir / "rosetta_work"]:
        if root.exists():
            candidates.extend(path for path in root.rglob("*") if path.is_file())
    return {
        "score_files": sorted(_rel(path, job_dir) for path in candidates if path.suffix in {".sc", ".score"} or "score" in path.name.lower()),
        "raw_files": sorted(_rel(path, job_dir) for path in candidates),
    }


def parse_rosetta_ddg(job_dir: Path | str) -> dict[str, Any]:
    job_dir = Path(job_dir)
    outputs = detect_rosetta_outputs(job_dir)
    search_order = outputs["score_files"] + outputs["raw_files"]
    seen: set[str] = set()
    errors: list[str] = []
    for relative in search_order:
        if relative in seen:
            continue
        seen.add(relative)
        path = job_dir / relative
        try:
            value = _parse_score_text(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            errors.append(f"{relative}: {exc}")
            continue
        if value is not None:
            return {
                "rosetta_ddg": value,
                "rosetta_unit": "Rosetta Energy Units",
                "raw_output_files": outputs["raw_files"],
                "parser_status": "success",
                "error_message": "",
            }
    return {
        "rosetta_ddg": None,
        "rosetta_unit": "Rosetta Energy Units",
        "raw_output_files": outputs["raw_files"],
        "parser_status": "parse_failed",
        "error_message": "; ".join(errors) or "No parseable Rosetta ddG/score value found",
    }


def validate_rosetta_job_inputs(job: dict[str, Any], root: Path | None = None) -> tuple[bool, list[str]]:
    errors = []
    if not job.get("chain_id"):
        errors.append("missing_chain_id")
    if not job.get("experimental_structure_path"):
        errors.append("missing_experimental_structure_path")
    elif not _resolve_path(str(job["experimental_structure_path"]), root).exists():
        errors.append("experimental_structure_path_not_found")
    try:
        parse_single_site_mutation(str(job.get("mutation", "")))
    except ValueError as exc:
        errors.append(str(exc))
    return not errors, errors


def parsed_result_payload(dataset: str, job: dict[str, Any], parse_result: dict[str, Any], number_of_runs: int) -> dict[str, Any]:
    return {
        "dataset_name": dataset,
        "method": "rosetta",
        "structure_source": "experimental",
        "mutation_record_id": job.get("mutation_record_id", ""),
        "job_id": job.get("job_id", ""),
        "mutation": job.get("mutation", ""),
        "chain_id": job.get("chain_id", ""),
        "rosetta_ddg": parse_result.get("rosetta_ddg"),
        "rosetta_unit": "Rosetta Energy Units",
        "number_of_runs": number_of_runs,
        "raw_output_files": parse_result.get("raw_output_files", []),
        "parser_status": parse_result.get("parser_status", ""),
        "error_message": parse_result.get("error_message", ""),
    }


def write_parsed_result(path: Path | str, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _parse_score_text(text: str) -> float | None:
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    header: list[str] | None = None
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0].lower() in {"score:", "score"} and any(part.lower() in {"ddg", "total_score"} for part in parts):
            header = parts
            continue
        if header and parts[0].lower() in {"score:", "score"}:
            columns = dict(zip(header[1:], parts[1:]))
            for key in ["ddg", "total_score", "score"]:
                if key in columns:
                    try:
                        return float(columns[key])
                    except ValueError:
                        continue
        if "ddg" in line.lower() or "total_score" in line.lower():
            numeric = [float(match.group(0)) for match in FLOAT_RE.finditer(line)]
            if numeric:
                return numeric[-1]
    for line in lines:
        numeric = [float(match.group(0)) for match in FLOAT_RE.finditer(line)]
        if numeric:
            return numeric[-1]
    return None


def _resolve_path(path: str, root: Path | None) -> Path:
    resolved = Path(path)
    if resolved.is_absolute() or root is None:
        return resolved
    return root / resolved


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()

