from __future__ import annotations

import re
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests

from .cif_utils import parse_structure_records
from .models import MutationInput

CANONICAL_AA_1 = frozenset("ACDEFGHIKLMNPQRSTVWY")
CANONICAL_AA_3_TO_1 = {
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
}


@dataclass(frozen=True, slots=True)
class ResolvedStructureSource:
    source_kind: Literal["structure_path", "pdb_id"]
    source_path: Path
    pdb_id: str | None = None
    cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class ValidatedMutationSite:
    chain_id: str
    residue_id: str
    residue_name_3: str
    residue_name_1: str


def default_structure_cache_dir() -> Path:
    return Path.home() / ".openfold3" / "benchmark" / "mmcif_cache"


def normalize_pdb_id(value: str) -> str:
    cleaned = str(value).strip().upper()
    if not cleaned:
        raise ValueError("PDB ID is empty")
    if not re.fullmatch(r"[A-Z0-9]{4}", cleaned):
        raise ValueError(f"Unsupported PDB ID format: {value!r}")
    return cleaned


def _normalize_residue_code(value: str, field_name: str) -> str:
    residue = str(value).strip().upper()
    if residue not in CANONICAL_AA_1:
        raise ValueError(f"{field_name} must be a canonical amino acid, got {value!r}")
    return residue


def _canonical_residues_by_site(structure_path: Path) -> "OrderedDict[tuple[str, str], str]":
    residues: "OrderedDict[tuple[str, str], str]" = OrderedDict()
    for atom in parse_structure_records(structure_path):
        key = (atom.chain_id, atom.residue_id)
        residues.setdefault(key, atom.residue_name.upper())
    return residues


def download_mmcif(
    pdb_id: str,
    *,
    cache_dir: str | Path | None = None,
    session: requests.Session | None = None,
    force: bool = False,
) -> tuple[Path, bool]:
    normalized_id = normalize_pdb_id(pdb_id)
    cache_root = Path(cache_dir or default_structure_cache_dir()).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    canonical_path = cache_root / f"{normalized_id}.cif"
    lower_path = cache_root / f"{normalized_id.lower()}.cif"
    for candidate in (canonical_path, lower_path):
        if candidate.exists() and not force:
            if candidate != canonical_path:
                shutil.copy2(candidate, canonical_path)
            return canonical_path, True

    client = session or requests.Session()
    response = client.get(
        f"https://files.rcsb.org/download/{normalized_id}.cif",
        timeout=60,
    )
    response.raise_for_status()
    canonical_path.write_text(response.text, encoding="utf-8")
    return canonical_path, False


def resolve_structure_source(
    *,
    structure_path: str | Path | None = None,
    pdb_id: str | None = None,
    cache_dir: str | Path | None = None,
    session: requests.Session | None = None,
    force_download: bool = False,
) -> ResolvedStructureSource:
    has_structure_path = structure_path is not None
    has_pdb_id = pdb_id is not None
    if has_structure_path == has_pdb_id:
        raise ValueError("Provide exactly one of structure_path or pdb_id")

    if has_structure_path:
        resolved_path = Path(structure_path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Structure path does not exist: {resolved_path}")
        return ResolvedStructureSource(
            source_kind="structure_path",
            source_path=resolved_path,
        )

    assert pdb_id is not None
    cif_path, cache_hit = download_mmcif(
        pdb_id,
        cache_dir=cache_dir,
        session=session,
        force=force_download,
    )
    return ResolvedStructureSource(
        source_kind="pdb_id",
        source_path=cif_path.resolve(),
        pdb_id=normalize_pdb_id(pdb_id),
        cache_hit=cache_hit,
    )


def validate_mutation_site(
    structure_path: str | Path,
    mutation: MutationInput,
) -> ValidatedMutationSite:
    structure_path = Path(structure_path).expanduser().resolve()
    if mutation.position_1based < 1:
        raise ValueError(
            f"position_1based must be positive, got {mutation.position_1based}"
        )

    expected_residue = _normalize_residue_code(mutation.from_residue, "from_residue")
    _normalize_residue_code(mutation.to_residue, "to_residue")

    residues = _canonical_residues_by_site(structure_path)
    residue_key = (str(mutation.chain_id), str(mutation.position_1based))
    residue_name_3 = residues.get(residue_key)
    if residue_name_3 is None:
        insertion_variants = sorted(
            residue_id
            for chain_id, residue_id in residues
            if chain_id == mutation.chain_id
            and residue_id.startswith(str(mutation.position_1based))
        )
        if insertion_variants:
            raise ValueError(
                "Mutation site resolves only to insertion-coded residues, which are "
                f"unsupported in v1: {mutation.chain_id}:{insertion_variants}"
            )
        raise ValueError(
            "Could not find residue "
            f"{mutation.chain_id}:{mutation.position_1based} in {structure_path}"
        )

    residue_name_1 = CANONICAL_AA_3_TO_1.get(residue_name_3)
    if residue_name_1 is None:
        raise ValueError(
            "Mutation site is not a canonical protein residue: "
            f"{mutation.chain_id}:{mutation.position_1based}={residue_name_3}"
        )
    if residue_name_1 != expected_residue:
        raise ValueError(
            "Expected residue "
            f"{expected_residue} at {mutation.chain_id}:{mutation.position_1based}, "
            f"found {residue_name_1}"
        )

    return ValidatedMutationSite(
        chain_id=str(mutation.chain_id),
        residue_id=str(mutation.position_1based),
        residue_name_3=residue_name_3,
        residue_name_1=residue_name_1,
    )


def extract_protein_sequence(
    structure_path: str | Path,
    chain_id: str,
) -> str:
    residues = _canonical_residues_by_site(Path(structure_path).expanduser().resolve())
    sequence: list[str] = []
    for (current_chain_id, _residue_id), residue_name_3 in residues.items():
        if current_chain_id != chain_id:
            continue
        residue_name_1 = CANONICAL_AA_3_TO_1.get(residue_name_3)
        if residue_name_1 is None:
            raise ValueError(
                f"Encountered non-canonical residue {residue_name_3} in chain {chain_id}"
            )
        sequence.append(residue_name_1)
    if not sequence:
        raise ValueError(f"Could not find protein residues for chain {chain_id}")
    return "".join(sequence)
