from __future__ import annotations

import shutil
from pathlib import Path

from openfold3_length_benchmark.composition import (
    collect_entry_compositions,
    extract_entry_composition,
    parse_pdb_ids,
)


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2].parent
    / "openfold-3"
    / "openfold3"
    / "tests"
    / "test_data"
    / "mmcifs"
)


def test_parse_pdb_ids_normalizes_deduplicates_and_clips() -> None:
    parsed = parse_pdb_ids("2crb\n2CRB, 5kc1 1psm", max_entries=2)
    assert parsed == ["2CRB", "5KC1"]


def test_extract_entry_composition_monomer() -> None:
    composition = extract_entry_composition(FIXTURE_ROOT / "2crb.cif")

    assert composition.status == "ok"
    assert composition.pdb_id == "2CRB"
    assert composition.chain_lengths == {"A": 97}
    assert composition.total_protein_length == 97
    assert composition.molecules == [
        {
            "molecule_type": "protein",
            "chain_ids": ["A"],
            "sequence": "GSSGSSGMEGPLNLAHQQSRRADRLLAAGKYEEAISCHRKATTYLSEAMKLTESEQAHLSLELQRDSHMKQLLLIQERWKRAKREERLKAHSGPSSG",
        }
    ]


def test_extract_entry_composition_groups_identical_sequences() -> None:
    composition = extract_entry_composition(FIXTURE_ROOT / "5kc1.cif")

    assert composition.status == "ok"
    assert composition.chain_count == 12
    assert composition.molecule_count == 1
    assert composition.total_protein_length == 226 * 12
    assert composition.molecules[0]["chain_ids"] == ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def test_extract_entry_composition_excludes_non_protein_components() -> None:
    composition = extract_entry_composition(FIXTURE_ROOT / "2q2k.cif")

    assert composition.status == "ok"
    assert composition.chain_lengths == {"A": 70, "B": 70}
    assert composition.total_protein_length == 140
    categories = {item.category for item in composition.excluded_components}
    assert "dna_polymer" in categories
    assert "non-polymer" in categories


def test_collect_entry_compositions_reuses_cached_files(tmp_path: Path) -> None:
    cache_dir = tmp_path / "mmcif"
    cache_dir.mkdir()
    shutil.copy2(FIXTURE_ROOT / "2crb.cif", cache_dir / "2CRB.cif")

    compositions = collect_entry_compositions("2crb", cache_dir=cache_dir)

    assert len(compositions) == 1
    assert compositions[0].status == "ok"
    assert compositions[0].source_path == (cache_dir / "2CRB.cif").resolve()
