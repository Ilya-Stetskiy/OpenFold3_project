from __future__ import annotations

import math
import shlex
from collections import OrderedDict, defaultdict
from pathlib import Path

from .models import AtomRecord, ResidueRecord, StructureSummary

ATOM_SITE_PREFIX = "_atom_site."


def _tokenize(line: str) -> list[str]:
    return shlex.split(line, posix=True)


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.dist(left, right)


def parse_pdb_atom_records(pdb_path: Path) -> list[AtomRecord]:
    records: list[AtomRecord] = []
    for raw_line in pdb_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.startswith(("ATOM", "HETATM")):
            continue
        line = raw_line.rstrip("\n")
        chain_id = line[21].strip() or "?"
        residue_id = line[22:26].strip() or "?"
        insertion_code = line[26].strip()
        if insertion_code:
            residue_id = f"{residue_id}{insertion_code}"
        b_factor_raw = line[60:66].strip()
        records.append(
            AtomRecord(
                chain_id=chain_id,
                residue_name=line[17:20].strip(),
                residue_id=residue_id,
                atom_name=line[12:16].strip(),
                x=float(line[30:38].strip()),
                y=float(line[38:46].strip()),
                z=float(line[46:54].strip()),
                b_factor=None if not b_factor_raw else float(b_factor_raw),
                group_pdb=line[0:6].strip(),
            )
        )

    if not records:
        raise ValueError(f"No ATOM/HETATM records found in {pdb_path}")
    return records


def parse_atom_site_records(cif_path: Path) -> list[AtomRecord]:
    headers: list[str] = []
    records: list[AtomRecord] = []
    in_atom_site_loop = False

    for raw_line in cif_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "loop_":
            headers = []
            in_atom_site_loop = False
            continue
        if line.startswith(ATOM_SITE_PREFIX):
            headers.append(line[len(ATOM_SITE_PREFIX) :])
            in_atom_site_loop = True
            continue
        if in_atom_site_loop and headers and line.startswith("_"):
            break
        if in_atom_site_loop and headers and line == "#":
            break
        if not in_atom_site_loop or not headers:
            continue

        parts = _tokenize(line)
        if len(parts) != len(headers):
            raise ValueError(
                f"Unexpected atom_site row width in {cif_path}: "
                f"expected {len(headers)} columns, got {len(parts)}"
            )
        row = dict(zip(headers, parts, strict=True))
        auth_chain = row.get("auth_asym_id", "?")
        label_chain = row.get("label_asym_id", "?")
        chain_id = auth_chain if auth_chain not in {"?", "."} else label_chain
        auth_seq_id = row.get("auth_seq_id", "?")
        label_seq_id = row.get("label_seq_id", "?")
        residue_id = auth_seq_id if auth_seq_id not in {"?", "."} else label_seq_id
        b_factor_raw = row.get("B_iso_or_equiv")
        records.append(
            AtomRecord(
                chain_id=chain_id,
                residue_name=row["auth_comp_id"],
                residue_id=residue_id,
                atom_name=row["auth_atom_id"],
                x=float(row["Cartn_x"]),
                y=float(row["Cartn_y"]),
                z=float(row["Cartn_z"]),
                b_factor=(
                    None
                    if b_factor_raw in {None, "?", "."}
                    else float(b_factor_raw)
                ),
                group_pdb=row.get("group_PDB"),
            )
        )

    if not records:
        raise ValueError(f"No atom_site records found in {cif_path}")
    return records


def parse_structure_records(structure_path: Path) -> list[AtomRecord]:
    suffix = structure_path.suffix.lower()
    if suffix == ".pdb":
        return parse_pdb_atom_records(structure_path)
    return parse_atom_site_records(structure_path)


def summarize_structure(
    cif_path: Path, chain_groups: tuple[tuple[str, ...], ...] = ()
) -> StructureSummary:
    atoms = parse_structure_records(cif_path)
    residues: "OrderedDict[tuple[str, str], list[AtomRecord]]" = OrderedDict()
    chain_atoms: dict[str, list[AtomRecord]] = defaultdict(list)

    for atom in atoms:
        residues.setdefault((atom.chain_id, atom.residue_id), []).append(atom)
        chain_atoms[atom.chain_id].append(atom)

    residues_by_chain: dict[str, list[ResidueRecord]] = defaultdict(list)
    for (chain_id, residue_id), residue_atoms in residues.items():
        ca_atom = next((atom for atom in residue_atoms if atom.atom_name == "CA"), None)
        b_factors = [atom.b_factor for atom in residue_atoms if atom.b_factor is not None]
        residues_by_chain[chain_id].append(
            ResidueRecord(
                chain_id=chain_id,
                residue_name=residue_atoms[0].residue_name,
                residue_id=residue_id,
                atom_names=tuple(atom.atom_name for atom in residue_atoms),
                ca_coord=None
                if ca_atom is None
                else (ca_atom.x, ca_atom.y, ca_atom.z),
                b_factor_mean=None
                if not b_factors
                else sum(b_factors) / len(b_factors),
            )
        )

    chain_ids = tuple(residues_by_chain.keys())
    inferred_chain_groups = chain_groups or tuple((chain_id,) for chain_id in chain_ids)
    min_atom_distance: float | None = None
    atom_contacts_5a = 0
    ca_contacts_8a = 0
    chain_pair_min_distances: dict[str, float] = {}

    chain_id_list = list(chain_ids)
    for left_index, left_chain in enumerate(chain_id_list):
        for right_chain in chain_id_list[left_index + 1 :]:
            pair_key = f"{left_chain}-{right_chain}"
            pair_min_distance: float | None = None
            for left_atom in chain_atoms[left_chain]:
                left_coord = (left_atom.x, left_atom.y, left_atom.z)
                for right_atom in chain_atoms[right_chain]:
                    distance = _distance(
                        left_coord, (right_atom.x, right_atom.y, right_atom.z)
                    )
                    if pair_min_distance is None or distance < pair_min_distance:
                        pair_min_distance = distance
                    if min_atom_distance is None or distance < min_atom_distance:
                        min_atom_distance = distance
                    if distance <= 5.0:
                        atom_contacts_5a += 1
            if pair_min_distance is not None:
                chain_pair_min_distances[pair_key] = pair_min_distance

            for left_residue in residues_by_chain[left_chain]:
                if left_residue.ca_coord is None:
                    continue
                for right_residue in residues_by_chain[right_chain]:
                    if right_residue.ca_coord is None:
                        continue
                    if _distance(left_residue.ca_coord, right_residue.ca_coord) <= 8.0:
                        ca_contacts_8a += 1

    return StructureSummary(
        atom_count=len(atoms),
        residue_count=len(residues),
        chain_ids=chain_ids,
        residues_by_chain={key: tuple(value) for key, value in residues_by_chain.items()},
        inferred_chain_groups=inferred_chain_groups,
        min_inter_chain_atom_distance=min_atom_distance,
        interface_atom_contacts_5a=atom_contacts_5a,
        interface_ca_contacts_8a=ca_contacts_8a,
        chain_pair_min_distances=chain_pair_min_distances,
    )
