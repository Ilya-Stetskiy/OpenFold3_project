from __future__ import annotations

from pathlib import Path

import pytest

from openfold3.benchmark.models import MutationInput
from openfold3.benchmark.structure_source import (
    extract_protein_sequence,
    normalize_pdb_id,
    resolve_structure_source,
    validate_mutation_site,
)

from .test_ddg_benchmark_harness import _write_foldx_ready_cif, _write_minimal_pdb


class _DummyResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _RecordingSession:
    def __init__(self, text: str):
        self.text = text
        self.urls: list[str] = []

    def get(self, url: str, timeout: int = 60):
        self.urls.append(f"{url}|timeout={timeout}")
        return _DummyResponse(self.text)


def test_normalize_pdb_id_uppercases_and_validates() -> None:
    assert normalize_pdb_id(" 1kd8 ") == "1KD8"

    with pytest.raises(ValueError, match="Unsupported PDB ID format"):
        normalize_pdb_id("bad-id")


def test_resolve_structure_source_accepts_existing_structure_path(tmp_path: Path) -> None:
    structure_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(structure_path)

    resolved = resolve_structure_source(structure_path=structure_path)

    assert resolved.source_kind == "structure_path"
    assert resolved.source_path == structure_path.resolve()
    assert resolved.pdb_id is None
    assert resolved.cache_hit is False


def test_resolve_structure_source_downloads_and_reuses_cached_pdb_id(tmp_path: Path) -> None:
    cif_path = tmp_path / "template.cif"
    _write_foldx_ready_cif(cif_path)
    session = _RecordingSession(cif_path.read_text(encoding="utf-8"))

    first = resolve_structure_source(
        pdb_id="1kd8",
        cache_dir=tmp_path / "cache",
        session=session,
    )
    second = resolve_structure_source(
        pdb_id="1KD8",
        cache_dir=tmp_path / "cache",
        session=session,
    )

    assert first.source_kind == "pdb_id"
    assert first.pdb_id == "1KD8"
    assert first.cache_hit is False
    assert first.source_path.exists()
    assert second.cache_hit is True
    assert second.source_path == first.source_path
    assert len(session.urls) == 1


def test_validate_mutation_site_accepts_matching_residue(tmp_path: Path) -> None:
    structure_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(structure_path)

    site = validate_mutation_site(
        structure_path,
        MutationInput(chain_id="A", from_residue="L", position_1based=1, to_residue="A"),
    )

    assert site.chain_id == "A"
    assert site.residue_id == "1"
    assert site.residue_name_3 == "LEU"
    assert site.residue_name_1 == "L"


def test_validate_mutation_site_rejects_residue_mismatch(tmp_path: Path) -> None:
    structure_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(structure_path)

    with pytest.raises(ValueError, match="Expected residue A"):
        validate_mutation_site(
            structure_path,
            MutationInput(
                chain_id="A",
                from_residue="A",
                position_1based=1,
                to_residue="V",
            ),
        )


def test_validate_mutation_site_rejects_missing_chain_or_position(tmp_path: Path) -> None:
    structure_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(structure_path)

    with pytest.raises(ValueError, match="Could not find residue B:9"):
        validate_mutation_site(
            structure_path,
            MutationInput(
                chain_id="B",
                from_residue="L",
                position_1based=9,
                to_residue="A",
            ),
        )


def test_validate_mutation_site_rejects_noncanonical_target(tmp_path: Path) -> None:
    structure_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(structure_path)

    with pytest.raises(ValueError, match="canonical amino acid"):
        validate_mutation_site(
            structure_path,
            MutationInput(
                chain_id="A",
                from_residue="L",
                position_1based=1,
                to_residue="Z",
            ),
        )


def test_extract_protein_sequence_reads_chain_sequence(tmp_path: Path) -> None:
    structure_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(structure_path)

    assert extract_protein_sequence(structure_path, "A") == "L"
    assert extract_protein_sequence(structure_path, "B") == "L"


def test_extract_protein_sequence_ignores_terminal_caps_in_mmcif_fixture() -> None:
    structure_path = (
        Path(__file__).resolve().parent / "test_data" / "mmcifs" / "1kd8.cif"
    )

    sequence = extract_protein_sequence(structure_path, "A")

    assert sequence.startswith("EVKQ")
    assert "ACE" not in sequence
