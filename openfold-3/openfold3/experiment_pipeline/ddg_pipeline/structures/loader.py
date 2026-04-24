from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openfold3.benchmark.cif_utils import parse_structure_records
from openfold3.benchmark.structure_source import CANONICAL_AA_3_TO_1


@dataclass(frozen=True, slots=True)
class MutationSite:
    chain_id: str
    residue_id: str
    residue_name_3: str
    residue_name_1: str
    ca_coord: tuple[float, float, float] | None


@dataclass(frozen=True, slots=True)
class LoadedStructure:
    path: Path
    residues_by_site: dict[tuple[str, str], MutationSite]


def load_structure(path: str | Path) -> LoadedStructure:
    structure_path = Path(path).expanduser().resolve()
    atoms = parse_structure_records(structure_path)
    residues: "OrderedDict[tuple[str, str], list[Any]]" = OrderedDict()
    for atom in atoms:
        residues.setdefault((str(atom.chain_id), str(atom.residue_id)), []).append(atom)

    residues_by_site: dict[tuple[str, str], MutationSite] = {}
    for (chain_id, residue_id), residue_atoms in residues.items():
        residue_name_3 = residue_atoms[0].residue_name.upper()
        residue_name_1 = CANONICAL_AA_3_TO_1.get(residue_name_3)
        if residue_name_1 is None:
            continue
        ca_atom = next((atom for atom in residue_atoms if atom.atom_name == "CA"), None)
        residues_by_site[(chain_id, residue_id)] = MutationSite(
            chain_id=chain_id,
            residue_id=residue_id,
            residue_name_3=residue_name_3,
            residue_name_1=residue_name_1,
            ca_coord=None if ca_atom is None else (ca_atom.x, ca_atom.y, ca_atom.z),
        )
    if not residues_by_site:
        raise ValueError(f"No canonical protein residues found in {structure_path}")
    return LoadedStructure(path=structure_path, residues_by_site=residues_by_site)


def extract_mutation_site(
    structure: LoadedStructure,
    position: int,
    expected_wt: str | None = None,
) -> MutationSite:
    if position < 1:
        raise ValueError(f"position must be >= 1, got {position}")

    residue_id = str(position)
    candidates = [
        site
        for (chain_id, current_residue_id), site in sorted(structure.residues_by_site.items())
        if current_residue_id == residue_id
    ]
    if not candidates:
        insertion_variants = sorted(
            f"{chain_id}:{current_residue_id}"
            for (chain_id, current_residue_id) in structure.residues_by_site
            if current_residue_id.startswith(residue_id)
        )
        if insertion_variants:
            raise ValueError(
                f"Mutation site {position} resolves only to insertion-coded residues in {structure.path}: "
                f"{', '.join(insertion_variants)}"
            )
        raise ValueError(f"Could not find residue id {position} in {structure.path}")

    if expected_wt is not None:
        candidates = [site for site in candidates if site.residue_name_1 == expected_wt]
        if not candidates:
            raise ValueError(
                f"Expected residue {expected_wt} at residue id {position} in {structure.path}"
            )

    if len(candidates) > 1:
        candidate_ids = ", ".join(f"{site.chain_id}:{site.residue_id}" for site in candidates)
        raise ValueError(f"Ambiguous mutation site for position {position} in {structure.path}: {candidate_ids}")

    return candidates[0]
