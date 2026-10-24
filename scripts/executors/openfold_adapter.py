#!/usr/bin/env python3
"""OpenFold WT adapter helpers for current_internal sequence targets."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


def load_openfold_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing OpenFold config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"OpenFold config must be a mapping: {path}")
    config.setdefault("openfold", {})
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


def validate_openfold_import(python_bin: str | Path, repo_dir: Path, project_root: Path) -> tuple[bool, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{repo_dir}:{project_root}" + (f":{existing}" if existing else "")
    try:
        completed = subprocess.run(
            [str(python_bin), "-c", "import openfold3; print(openfold3.__name__)"],
            cwd=repo_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout).strip()
    return True, ""


def read_sequence_manifest(path: Path | str) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing sequence manifest: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def validate_sequence_row(row: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = []
    for field in ["dataset_name", "sequence_id", "pdb_id", "chain_id", "sequence"]:
        if not str(row.get(field, "")).strip():
            errors.append(f"missing_{field}")
    sequence = str(row.get("sequence", "")).strip().upper()
    if sequence and not sequence.isalpha():
        errors.append("invalid_sequence_characters")
    return not errors, errors


def select_sequence_rows(
    rows: list[dict[str, str]],
    *,
    dataset: str,
    max_sequences: int | None = None,
    sequence_id: str | None = None,
    pdb_id: str | None = None,
) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("dataset_name", dataset) == dataset]
    if sequence_id:
        selected = [row for row in selected if row.get("sequence_id") == sequence_id]
    if pdb_id:
        selected = [row for row in selected if row.get("pdb_id", "").upper() == pdb_id.upper()]
    selected = _deduplicate_by_sequence_id(selected)
    if max_sequences is not None:
        selected = selected[:max_sequences]
    return selected


def batch_sequence_rows(rows: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    batch_size = max(1, int(batch_size))
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def build_query_payload(rows: list[dict[str, str]], *, use_msas: bool = True, use_templates: bool = True) -> dict[str, Any]:
    queries = {}
    for row in rows:
        sequence_id = row["sequence_id"]
        chain_id = row.get("chain_id", "A") or "A"
        queries[sequence_id] = {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": [chain_id],
                    "description": sequence_id,
                    "sequence": row["sequence"],
                }
            ],
            "use_msas": bool(use_msas),
            "use_main_msas": bool(use_msas),
            "use_paired_msas": False,
        }
        if not use_templates:
            queries[sequence_id]["chains"][0]["template_alignment_file_path"] = None
            queries[sequence_id]["chains"][0]["template_entry_chain_ids"] = None
    return {"queries": queries}


def write_query_json(path: Path | str, rows: list[dict[str, str]], *, use_msas: bool, use_templates: bool, force: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_query_payload(rows, use_msas=use_msas, use_templates=use_templates)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_openfold_predict_command(
    python_bin: str | Path,
    query_json: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[str]:
    openfold = config.get("openfold", {})
    command = [
        str(python_bin),
        "-m",
        "openfold3.run_openfold",
        "predict",
        "--query_json",
        str(query_json),
        "--output_dir",
        str(output_dir),
        "--use_msa_server",
        str(bool(openfold.get("use_msa_server", False))).lower(),
        "--use_templates",
        str(bool(openfold.get("use_templates", False))).lower(),
    ]
    if openfold.get("runner_yaml"):
        command.extend(["--runner_yaml", str(openfold["runner_yaml"])])
    if openfold.get("inference_ckpt_path"):
        command.extend(["--inference_ckpt_path", str(openfold["inference_ckpt_path"])])
    if openfold.get("inference_ckpt_name"):
        command.extend(["--inference_ckpt_name", str(openfold["inference_ckpt_name"])])
    if openfold.get("num_diffusion_samples") is not None:
        command.extend(["--num_diffusion_samples", str(openfold["num_diffusion_samples"])])
    if openfold.get("num_model_seeds") is not None:
        command.extend(["--num_model_seeds", str(openfold["num_model_seeds"])])
    return command


def model_path_from_confidence_path(path: Path) -> Path | None:
    name = path.name
    for suffix in ("_confidences_aggregated.json", "_confidences.json"):
        if name.endswith(suffix):
            return path.with_name(f"{name.removesuffix(suffix)}_model.cif")
    return None


def select_best_summary_row(output_dir: Path | str, query_id: str) -> dict[str, Any]:
    return _select_best_summary_candidate(output_dir, query_id)[2]


def _select_best_summary_candidate(output_dir: Path | str, query_id: str) -> tuple[Path, Path, dict[str, Any]]:
    output_dir = Path(output_dir)
    candidates: list[tuple[float, int, dict[str, Any], Path, Path]] = []
    for summary_path in sorted(output_dir.rglob("summary.jsonl")):
        for line_number, line in enumerate(summary_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if str(payload.get("query_id", "")) != query_id:
                continue
            score = payload.get("sample_ranking_score")
            confidence_path = payload.get("aggregated_confidence_path")
            if score is None or confidence_path is None:
                continue
            confidence = _resolve_output_path(Path(str(confidence_path)), summary_path.parent)
            model = model_path_from_confidence_path(confidence)
            if model is not None and confidence.exists() and model.exists():
                candidates.append((float(score), -line_number, payload, confidence, model))
    if not candidates:
        raise RuntimeError(f"No ranked OpenFold summary row with model/confidence found for query_id={query_id}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = candidates[0]
    tied = [candidate for candidate in candidates if candidate[:2] == best[:2]]
    if len(tied) > 1:
        raise RuntimeError(f"Ambiguous OpenFold summary ranking for query_id={query_id}")
    return best[3], best[4], best[2]


def select_output_cif(output_dir: Path | str, query_id: str) -> Path:
    output_dir = Path(output_dir)
    try:
        row = select_best_summary_row(output_dir, query_id)
        confidence = _resolve_output_path(Path(str(row["aggregated_confidence_path"])), output_dir)
        model = model_path_from_confidence_path(confidence)
        if model is not None and model.exists():
            return model
    except RuntimeError:
        pass
    cif_candidates = [path for path in output_dir.rglob("*.cif") if path.is_file()]
    if len(cif_candidates) == 1:
        return cif_candidates[0]
    first_sample = _select_first_sample_cif(cif_candidates)
    if first_sample is not None:
        return first_sample
    preferred = [path for path in cif_candidates if _is_preferred_cif_name(path)]
    if len(preferred) == 1:
        return preferred[0]
    raise RuntimeError(f"OpenFold completed with ambiguous or missing CIF outputs for query_id={query_id}")


def normalize_openfold_outputs(output_dir: Path | str, query_id: str, target_dir: Path | str, force: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    source_confidence, source_model, summary_row = _select_best_summary_candidate(output_dir, query_id)
    if not source_model.exists():
        raise RuntimeError(f"Selected OpenFold model does not exist: {source_model}")
    if not source_confidence.exists():
        raise RuntimeError(f"Selected OpenFold confidence does not exist: {source_confidence}")
    model_target = target_dir / "model.cif"
    pdb_target = target_dir / "model.pdb"
    confidence_target = target_dir / "confidence.json"
    result_target = target_dir / "openfold_result.json"
    for path in [model_target, pdb_target, confidence_target, result_target]:
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing file without --force: {path}")
    shutil.copy2(source_model, model_target)
    shutil.copy2(source_confidence, confidence_target)
    conversion = convert_cif_to_pdb(model_target, pdb_target, force=force)
    result = {
        "query_id": query_id,
        "model_path": str(model_target),
        "pdb_path": str(pdb_target),
        "confidence_path": str(confidence_target),
        "source_model_path": str(source_model),
        "source_confidence_path": str(source_confidence),
        "conversion": conversion,
        "summary_row": summary_row,
        "parser_status": "success",
        "error_message": "",
    }
    result_target.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return result


def convert_cif_to_pdb(cif_path: Path | str, pdb_path: Path | str, force: bool = False) -> dict[str, Any]:
    cif_path = Path(cif_path)
    pdb_path = Path(pdb_path)
    if pdb_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file without --force: {pdb_path}")
    atoms = [
        atom
        for atom in parse_atom_site(cif_path)
        if first_model_atom(atom) and is_protein_atom(atom)
    ]
    if not atoms:
        raise RuntimeError(f"No first-model protein atoms found in CIF: {cif_path}")
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    chains = sorted({chain_id_for_atom(atom) for atom in atoms if chain_id_for_atom(atom)})
    with pdb_path.open("w", encoding="utf-8") as handle:
        for serial, atom in enumerate(atoms, start=1):
            handle.write(pdb_atom_line(serial, atom))
        handle.write("END\n")
    return {
        "status": "success",
        "source_cif": str(cif_path),
        "pdb_path": str(pdb_path),
        "atom_count": len(atoms),
        "chains": chains,
    }


def validate_cif_to_pdb_conversion(cif_path: Path | str, pdb_path: Path | str, force: bool = False) -> dict[str, Any]:
    try:
        result = convert_cif_to_pdb(cif_path, pdb_path, force=force)
        pdb = Path(pdb_path)
        atom_lines = [line for line in pdb.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("ATOM")]
        residue_keys = {(line[21].strip(), line[22:26].strip(), line[26].strip()) for line in atom_lines if len(line) >= 27}
        if not pdb.exists() or not atom_lines or not residue_keys:
            return {
                "status": "failed",
                "source_cif": str(cif_path),
                "pdb_path": str(pdb_path),
                "atom_count": len(atom_lines),
                "residue_count": len(residue_keys),
                "error_message": "converted PDB has no ATOM records or residues",
            }
        result["residue_count"] = len(residue_keys)
        result["error_message"] = ""
        return result
    except Exception as exc:
        return {
            "status": "failed",
            "source_cif": str(cif_path),
            "pdb_path": str(pdb_path),
            "atom_count": 0,
            "residue_count": 0,
            "error_message": str(exc),
        }


def parse_atom_site(cif_path: Path | str) -> list[dict[str, str]]:
    lines = Path(cif_path).read_text(encoding="utf-8", errors="replace").splitlines()
    atoms: list[dict[str, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        tags: list[str] = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            tags.append(lines[i].strip())
            i += 1
        if not tags or not all(tag.startswith("_atom_site.") for tag in tags):
            continue
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped or stripped == "#" or stripped == "loop_" or stripped.startswith("_"):
                break
            tokens = tokenize_cif_line(lines[i])
            if len(tokens) >= len(tags):
                atoms.append({tag.split(".", 1)[1]: tokens[index] for index, tag in enumerate(tags)})
            i += 1
        break
    return atoms


def tokenize_cif_line(line: str) -> list[str]:
    tokens: list[str] = []
    token = ""
    quote = ""
    for char in line:
        if quote:
            if char == quote:
                tokens.append(token)
                token = ""
                quote = ""
            else:
                token += char
        elif char in {"'", '"'}:
            quote = char
        elif char.isspace():
            if token:
                tokens.append(token)
                token = ""
        else:
            token += char
    if token:
        tokens.append(token)
    return tokens


def atom_value(atom: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = atom.get(key, "")
        if value not in {"", ".", "?"}:
            return value
    return ""


def first_model_atom(atom: dict[str, str]) -> bool:
    model = atom_value(atom, "pdbx_PDB_model_num")
    return model in {"", "1"}


def is_protein_atom(atom: dict[str, str]) -> bool:
    group = atom_value(atom, "group_PDB")
    comp = atom_value(atom, "label_comp_id", "auth_comp_id").upper()
    return group == "ATOM" and comp in AA3_TO_1


def chain_id_for_atom(atom: dict[str, str]) -> str:
    return atom_value(atom, "auth_asym_id", "label_asym_id") or "A"


def residue_id_for_atom(atom: dict[str, str]) -> str:
    return atom_value(atom, "auth_seq_id", "label_seq_id")


def insertion_code_for_atom(atom: dict[str, str]) -> str:
    value = atom_value(atom, "pdbx_PDB_ins_code")
    return "" if value in {".", "?"} else value


def pdb_atom_line(serial: int, atom: dict[str, str]) -> str:
    atom_name = atom_value(atom, "auth_atom_id", "label_atom_id")[:4]
    comp = atom_value(atom, "auth_comp_id", "label_comp_id").upper()[:3]
    chain_id = chain_id_for_atom(atom)[:1]
    res_id_text = residue_id_for_atom(atom)
    try:
        res_id = int(float(res_id_text))
    except ValueError:
        res_id = 0
    ins = insertion_code_for_atom(atom)[:1]
    x = float(atom_value(atom, "Cartn_x") or 0.0)
    y = float(atom_value(atom, "Cartn_y") or 0.0)
    z = float(atom_value(atom, "Cartn_z") or 0.0)
    occupancy = float(atom_value(atom, "occupancy") or 1.0)
    b_factor = float(atom_value(atom, "B_iso_or_equiv") or 0.0)
    element = atom_value(atom, "type_symbol")[:2].rjust(2)
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} {comp:>3s} {chain_id:1s}"
        f"{res_id:4d}{ins:1s}   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_factor:6.2f}          {element}\n"
    )


def _deduplicate_by_sequence_id(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    result = []
    for row in rows:
        sequence_id = row.get("sequence_id", "")
        if sequence_id in seen:
            continue
        seen.add(sequence_id)
        result.append(row)
    return result


def _resolve_output_path(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def _select_first_sample_cif(cif_candidates: list[Path]) -> Path | None:
    sample_pattern = re.compile(r"_sample_(\d+)_model\.cif$")
    indexed: list[tuple[int, Path]] = []
    for path in cif_candidates:
        match = sample_pattern.search(path.name)
        if match is not None:
            indexed.append((int(match.group(1)), path))
    if len(indexed) != len(cif_candidates):
        return None
    sample_one = [path for index, path in indexed if index == 1]
    return sample_one[0] if len(sample_one) == 1 else None


def _is_preferred_cif_name(path: Path) -> bool:
    name = path.name.lower()
    return "final" in name or ("model" in name and "intermediate" not in name)
