from __future__ import annotations

from of_notebook_lib.query_builders import (
    apply_mutation_to_molecules,
    build_mutation_scan_payload,
    build_single_query_payload,
    normalize_molecules,
)


def test_normalize_molecules_preserves_expected_fields(big_ace_molecules: list[dict]) -> None:
    normalized = normalize_molecules(big_ace_molecules)

    assert len(normalized) == 2
    assert normalized[0]["molecule_type"] == "protein"
    assert normalized[0]["chain_ids"] == ["A"]
    assert normalized[1]["chain_ids"] == ["B"]
    assert "sequence" in normalized[0]


def test_apply_mutation_to_molecules_changes_only_target_chain(
    big_ace_molecules: list[dict],
) -> None:
    mutated = apply_mutation_to_molecules(
        big_ace_molecules,
        chain_id="B",
        position_1based=4,
        new_residue="G",
    )

    assert mutated[1]["sequence"][3] == "G"
    assert mutated[0]["sequence"] == big_ace_molecules[0]["sequence"]
    assert mutated[1]["sequence"] != big_ace_molecules[1]["sequence"]


def test_build_single_query_payload_uses_requested_name(big_ace_molecules: list[dict]) -> None:
    payload = build_single_query_payload("demo_case", big_ace_molecules)

    assert set(payload) == {"queries"}
    assert set(payload["queries"]) == {"demo_case"}
    assert len(payload["queries"]["demo_case"]["chains"]) == 2


def test_build_mutation_scan_payload_includes_wt_and_mutants(
    big_ace_molecules: list[dict],
) -> None:
    payload = build_mutation_scan_payload(
        query_prefix="scan_case",
        molecules=big_ace_molecules,
        mutation_chain_id="B",
        position_1based=4,
        amino_acids="AG",
        include_wt=True,
    )

    assert set(payload["queries"]) == {
        "scan_case__WT",
        "scan_case__B_F4A",
        "scan_case__B_F4G",
    }
