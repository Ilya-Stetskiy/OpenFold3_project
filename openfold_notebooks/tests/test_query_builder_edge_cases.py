from __future__ import annotations

import pytest

from of_notebook_lib.query_builders import apply_mutation_to_molecules, apply_point_mutation, build_mutation_scan_payload, normalize_molecules


def test_normalize_molecules_raises_on_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="missing 'molecule_type'"):
        normalize_molecules([{"chain_ids": ["A"], "sequence": "AAAA"}])

    with pytest.raises(ValueError, match="missing chain identifiers"):
        normalize_molecules([{"molecule_type": "protein", "sequence": "AAAA"}])


def test_apply_point_mutation_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="outside sequence length"):
        apply_point_mutation("AAAA", 0, "G")

    with pytest.raises(ValueError, match="Unsupported residue"):
        apply_point_mutation("AAAA", 1, "Z")


def test_apply_mutation_to_molecules_rejects_missing_chain_and_sequence() -> None:
    with pytest.raises(ValueError, match="does not have a sequence field"):
        apply_mutation_to_molecules(
            [{"molecule_type": "ligand", "chain_ids": ["L"]}],
            chain_id="L",
            position_1based=1,
            new_residue="A",
        )

    with pytest.raises(ValueError, match="was not found"):
        apply_mutation_to_molecules(
            [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": "AAAA"}],
            chain_id="B",
            position_1based=1,
            new_residue="A",
        )


def test_build_mutation_scan_payload_rejects_missing_target_chain() -> None:
    with pytest.raises(ValueError, match="was not found"):
        build_mutation_scan_payload(
            query_prefix="scan",
            molecules=[{"molecule_type": "protein", "chain_ids": ["A"], "sequence": "AAAA"}],
            mutation_chain_id="B",
            position_1based=1,
            amino_acids="AG",
        )
