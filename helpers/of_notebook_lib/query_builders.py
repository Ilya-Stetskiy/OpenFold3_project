from __future__ import annotations

from copy import deepcopy


CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"


def normalize_molecules(molecules: list[dict]) -> list[dict]:
    normalized: list[dict] = []

    for index, molecule in enumerate(molecules, start=1):
        mol = deepcopy(molecule)

        molecule_type = mol.get("molecule_type") or mol.get("type")
        if not molecule_type:
            raise ValueError(f"Molecule #{index} is missing 'molecule_type'")

        chain_ids = mol.get("chain_ids")
        if chain_ids is None and "id" in mol:
            chain_ids = [mol["id"]]
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]
        if not chain_ids:
            raise ValueError(f"Molecule #{index} is missing chain identifiers")

        normalized_molecule = {
            "molecule_type": str(molecule_type).lower(),
            "chain_ids": [str(chain_id) for chain_id in chain_ids],
        }

        if "sequence" in mol and mol["sequence"] is not None:
            normalized_molecule["sequence"] = str(mol["sequence"]).strip().upper()

        for field in ("smiles", "ccd_codes", "modifications"):
            if field in mol and mol[field] is not None:
                normalized_molecule[field] = deepcopy(mol[field])

        normalized.append(normalized_molecule)

    return normalized


def apply_point_mutation(
    sequence: str,
    position_1based: int,
    new_residue: str,
) -> str:
    if position_1based < 1 or position_1based > len(sequence):
        raise ValueError(
            f"Mutation position {position_1based} is outside sequence length {len(sequence)}"
        )

    residue = str(new_residue).upper()
    if residue not in CANONICAL_AA:
        raise ValueError(f"Unsupported residue '{new_residue}'")

    zero_based = position_1based - 1
    return sequence[:zero_based] + residue + sequence[zero_based + 1 :]


def apply_mutation_to_molecules(
    molecules: list[dict],
    chain_id: str,
    position_1based: int,
    new_residue: str,
) -> list[dict]:
    work_molecules = normalize_molecules(molecules)
    target_chain = str(chain_id)

    changed = False
    for molecule in work_molecules:
        if target_chain in molecule["chain_ids"]:
            if "sequence" not in molecule:
                raise ValueError(f"Chain {target_chain} does not have a sequence field")
            molecule["sequence"] = apply_point_mutation(
                sequence=molecule["sequence"],
                position_1based=position_1based,
                new_residue=new_residue,
            )
            changed = True

    if not changed:
        raise ValueError(f"Chain '{target_chain}' was not found in molecules")

    return work_molecules


def build_single_query_payload(query_name: str, molecules: list[dict]) -> dict:
    return {
        "queries": {
            query_name: {
                "chains": normalize_molecules(molecules),
            }
        }
    }


def build_mutation_scan_payload(
    query_prefix: str,
    molecules: list[dict],
    mutation_chain_id: str,
    position_1based: int,
    amino_acids: str | list[str],
    include_wt: bool = True,
) -> dict:
    work_molecules = normalize_molecules(molecules)
    residues = list(amino_acids) if isinstance(amino_acids, str) else amino_acids
    residues = [str(residue).upper() for residue in residues]

    payload = {"queries": {}}
    target_chain = str(mutation_chain_id)

    wt_residue = None
    for molecule in work_molecules:
        if target_chain in molecule["chain_ids"]:
            wt_residue = molecule["sequence"][position_1based - 1]
            break

    if wt_residue is None:
        raise ValueError(f"Chain '{target_chain}' was not found in molecules")

    if include_wt:
        payload["queries"][f"{query_prefix}__WT"] = {
            "chains": deepcopy(work_molecules),
        }

    for residue in residues:
        mutation_label = f"{target_chain}_{wt_residue}{position_1based}{residue}"
        payload["queries"][f"{query_prefix}__{mutation_label}"] = {
            "chains": apply_mutation_to_molecules(
                work_molecules,
                chain_id=target_chain,
                position_1based=position_1based,
                new_residue=residue,
            )
        }

    return payload
