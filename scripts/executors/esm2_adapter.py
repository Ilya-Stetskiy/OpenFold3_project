#!/usr/bin/env python3
"""ESM2 adapter helpers for the current_internal GPU executor."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def load_esm2_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing ESM2 config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"ESM2 config must be a mapping: {path}")
    config.setdefault("esm2", {})
    config.setdefault("paths", {})
    return config


def validate_python_bin(python_bin: str | Path) -> tuple[bool, str]:
    if not python_bin:
        return False, "python_bin is empty"
    resolved = shutil.which(str(python_bin)) or str(python_bin)
    path = Path(resolved)
    if not path.exists():
        return False, f"python executable does not exist: {python_bin}"
    if not os.access(path, os.X_OK):
        return False, f"python executable is not executable: {python_bin}"
    return True, ""


def parse_single_site_mutation(mutation: str) -> tuple[str, int, str]:
    match = MUTATION_RE.match(str(mutation).strip().upper())
    if not match:
        raise ValueError(f"Invalid single-site mutation: {mutation}")
    wild_type, position, mutant = match.groups()
    return wild_type, int(position), mutant


def read_fasta(path: Path | str) -> dict[str, str]:
    path = Path(path)
    sequences: dict[str, list[str]] = {}
    current: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                sequences.setdefault(current, [])
                continue
            if current is None:
                raise ValueError(f"FASTA sequence encountered before header: {path}")
            sequences[current].append(line)
    return {name: "".join(parts).upper() for name, parts in sequences.items()}


def sequence_id_for_job(job: dict[str, Any]) -> str:
    uniprot_id = str(job.get("uniprot_id", "")).strip() or "unknown_uniprot"
    pdb_id = str(job.get("pdb_id", "")).strip()
    chain_id = str(job.get("chain_id", "")).strip()
    return f"current_internal__{uniprot_id}__{pdb_id}_{chain_id}"


def find_sequence_for_job(job: dict[str, Any], fasta_path: Path | str) -> tuple[str, str]:
    sequences = read_fasta(fasta_path)
    preferred = sequence_id_for_job(job)
    if preferred in sequences:
        return preferred, sequences[preferred]
    pdb_id = str(job.get("pdb_id", "")).strip().upper()
    chain_id = str(job.get("chain_id", "")).strip()
    suffix = f"__{pdb_id}_{chain_id}"
    matches = [(name, seq) for name, seq in sequences.items() if name.upper().endswith(suffix.upper())]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No FASTA sequence found for {preferred}")
    raise ValueError(f"Ambiguous FASTA sequences for {preferred}: {[name for name, _ in matches]}")


def validate_esm2_job_inputs(job: dict[str, Any], root: Path | None = None) -> tuple[bool, list[str]]:
    errors = []
    fasta_path = str(job.get("sequence_input_fasta_path", "")).strip()
    if not fasta_path:
        errors.append("missing_sequence_input_fasta_path")
    elif not _resolve_path(fasta_path, root).exists():
        errors.append("sequence_input_fasta_path_not_found")
    try:
        parse_single_site_mutation(str(job.get("mutation", "")))
    except ValueError as exc:
        errors.append(str(exc))
    if not job.get("mutation_record_id"):
        errors.append("missing_mutation_record_id")
    return not errors, errors


def write_job_input_json(job_dir: Path | str, job: dict[str, Any], fasta_path: Path | str, root: Path | None = None, force: bool = False) -> Path:
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    sequence_id, sequence = find_sequence_for_job(job, _resolve_path(str(fasta_path), root))
    wild_type, position, mutant = parse_single_site_mutation(str(job.get("mutation", "")))
    if position < 1 or position > len(sequence):
        raise ValueError(f"Mutation position {position} is outside sequence length {len(sequence)}")
    sequence_wt = sequence[position - 1]
    payload = {
        "dataset_name": job.get("dataset_name", ""),
        "job_id": job.get("job_id", ""),
        "mutation_record_id": job.get("mutation_record_id", ""),
        "method": "esm2",
        "structure_source": "sequence_only",
        "protein_id": job.get("protein_id", ""),
        "uniprot_id": job.get("uniprot_id", ""),
        "pdb_id": job.get("pdb_id", ""),
        "chain_id": job.get("chain_id", ""),
        "mutation": job.get("mutation", ""),
        "wild_type_aa": wild_type,
        "position": position,
        "mutant_aa": mutant,
        "experimental_ddg": job.get("experimental_ddg", ""),
        "sequence_id": sequence_id,
        "sequence": sequence,
        "sequence_length": len(sequence),
        "sequence_wt_aa": sequence_wt,
        "sequence_wt_matches_mutation": sequence_wt == wild_type,
    }
    path = job_dir / "job_input.json"
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file without --force: {path}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_esm2_score_command(
    python_bin: str | Path,
    scorer_script: str | Path,
    input_json: str | Path,
    output_json: str | Path,
    config: dict[str, Any],
) -> list[str]:
    esm2 = config.get("esm2", {})
    command = [
        str(python_bin),
        str(scorer_script),
        "--input-json",
        str(input_json),
        "--output-json",
        str(output_json),
        "--model-name",
        str(esm2.get("model_name", "facebook/esm2_t33_650M_UR50D")),
        "--device",
        str(esm2.get("device", "cuda")),
        "--scoring-mode",
        str(esm2.get("scoring_mode", "masked_marginal")),
    ]
    max_length = esm2.get("max_sequence_length")
    if max_length not in {None, ""}:
        command.extend(["--max-sequence-length", str(max_length)])
    return command


def parse_esm2_result(job_dir: Path | str) -> dict[str, Any]:
    path = Path(job_dir) / "parsed_result.json"
    if not path.exists():
        return {
            "esm2_score": None,
            "esm2_score_unit": "log_probability_difference",
            "parser_status": "parse_failed",
            "error_message": "parsed_result.json was not created",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "esm2_score": None,
            "esm2_score_unit": "log_probability_difference",
            "parser_status": "parse_failed",
            "error_message": f"Invalid parsed_result.json: {exc}",
        }
    if data.get("parser_status") != "success":
        return {
            "esm2_score": data.get("esm2_score"),
            "esm2_score_unit": data.get("esm2_score_unit", "log_probability_difference"),
            "parser_status": data.get("parser_status", "parse_failed"),
            "error_message": data.get("error_message", ""),
        }
    return {
        "esm2_score": data.get("esm2_score"),
        "esm2_score_unit": data.get("esm2_score_unit", "log_probability_difference"),
        "parser_status": "success",
        "error_message": "",
    }


def parsed_result_payload(dataset: str, job: dict[str, Any], score_payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    esm2 = config.get("esm2", {})
    return {
        "dataset_name": dataset,
        "method": "esm2",
        "structure_source": "sequence_only",
        "mutation_record_id": job.get("mutation_record_id", ""),
        "job_id": job.get("job_id", ""),
        "mutation": job.get("mutation", ""),
        "chain_id": job.get("chain_id", ""),
        "esm2_score": score_payload.get("esm2_score"),
        "esm2_score_unit": score_payload.get("esm2_score_unit", "log_probability_difference"),
        "model_name": esm2.get("model_name", ""),
        "scoring_mode": esm2.get("scoring_mode", "masked_marginal"),
        "parser_status": score_payload.get("parser_status", ""),
        "error_message": score_payload.get("error_message", ""),
    }


def _resolve_path(path: str, root: Path | None) -> Path:
    resolved = Path(path)
    if resolved.is_absolute() or root is None:
        return resolved
    return root / resolved
