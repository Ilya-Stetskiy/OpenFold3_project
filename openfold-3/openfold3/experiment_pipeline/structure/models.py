from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _require_str(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Missing required field: {key}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Empty required field: {key}")
    return text


def _require_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Missing required field: {key}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer field: {key}") from exc
    if parsed < 1:
        raise ValueError(f"{key} must be >= 1")
    return parsed


def _require_float(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Missing required field: {key}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float field: {key}") from exc


def _normalize_residue(row: Mapping[str, object], key: str) -> str:
    residue = _require_str(row, key).upper()
    if len(residue) != 1:
        raise ValueError(f"{key} must be a single-letter amino acid code")
    return residue


@dataclass(frozen=True, slots=True)
class MutationCase:
    protein_id: str
    pdb_id: str
    chain: str
    position: int
    wt_residue: str
    mut_residue: str
    experimental_ddg: float
    mutation_id: str
    pdb_residue_id: str
    pdb_path: Path

    @property
    def case_id(self) -> str:
        return f"{self.protein_id}__{self.mutation_id}"

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "MutationCase":
        protein_id = _require_str(row, "protein_id")
        pdb_id = _require_str(row, "pdb_id").upper()
        chain = _require_str(row, "chain").upper()
        position = _require_int(row, "position")
        wt_residue = _normalize_residue(row, "wt_residue")
        mut_residue = _normalize_residue(row, "mut_residue")
        experimental_ddg = _require_float(row, "experimental_ddg")
        mutation_id = _require_str(row, "mutation_id")
        pdb_residue_id = _require_str(row, "pdb_residue_id")
        pdb_path = Path(_require_str(row, "pdb_path")).expanduser().resolve()
        return cls(
            protein_id=protein_id,
            pdb_id=pdb_id,
            chain=chain,
            position=position,
            wt_residue=wt_residue,
            mut_residue=mut_residue,
            experimental_ddg=experimental_ddg,
            mutation_id=mutation_id,
            pdb_residue_id=pdb_residue_id,
            pdb_path=pdb_path,
        )


@dataclass(frozen=True, slots=True)
class BackendResult:
    backend_name: str
    status: str
    output_dir: Path
    artifact_paths: tuple[Path, ...] = ()
    message: str = ""


@dataclass(frozen=True, slots=True)
class CaseManifest:
    case: MutationCase
    case_dir: Path
    input_dir: Path
    case_json_path: Path
    mutant_fasta_path: Path
    wild_type_sequence: str
    mutated_sequence: str
    backend_results: tuple[BackendResult, ...] = ()
