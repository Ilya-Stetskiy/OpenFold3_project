from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from openfold3_length_benchmark.benchmarking import compute_structure_rmsd, run_rmsd_benchmark


def _write_pdb(path: Path, coords: list[tuple[float, float, float]]) -> None:
    lines = []
    for index, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {index:5d} {'CA':>4} {'ALA':>3} {'A'}{index:4d} "
            f"   {x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{20.00:6.2f}          {'C':>2}"
        )
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_compute_structure_rmsd_superposes_transformed_model(tmp_path: Path) -> None:
    ref_path = tmp_path / "ref.pdb"
    pred_path = tmp_path / "pred.pdb"
    ref_coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    pred_coords = [
        (10.0, 5.0, -2.0),
        (10.0, 6.0, -2.0),
        (9.0, 5.0, -2.0),
    ]
    _write_pdb(ref_path, ref_coords)
    _write_pdb(pred_path, pred_coords)

    result = compute_structure_rmsd(pred_path, ref_path, atom_set="ca")

    assert result["coverage"]["matched_atom_count"] == 3
    assert result["coverage"]["matching_mode"] == "exact"
    assert result["rmsd_after_superposition"] < 1e-6


def test_compute_structure_rmsd_falls_back_to_ordinal_residue_matching(tmp_path: Path) -> None:
    pred_path = tmp_path / "pred.cif"
    ref_path = tmp_path / "ref.cif"
    pred_path.write_text(
        """data_pred
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.auth_atom_id
_atom_site.label_comp_id
_atom_site.auth_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 C CA CA GLY GLY A A 1 1 0.0 0.0 0.0
ATOM 2 C CA CA GLY GLY A A 2 2 1.0 0.0 0.0
#
""",
        encoding="utf-8",
    )
    ref_path.write_text(
        """data_ref
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.auth_atom_id
_atom_site.label_comp_id
_atom_site.auth_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 C CA CA GLY GLY A A 101 101 0.0 0.0 0.0
ATOM 2 C CA CA GLY GLY A A 102 102 1.0 0.0 0.0
#
""",
        encoding="utf-8",
    )

    result = compute_structure_rmsd(pred_path, ref_path, atom_set="ca")

    assert result["coverage"]["matched_atom_count"] == 2
    assert result["coverage"]["matching_mode"] == "ordinal_chain_residue"
    assert result["rmsd_after_superposition"] < 1e-6


def test_run_rmsd_benchmark_writes_expected_rows(tmp_path: Path, monkeypatch) -> None:
    pred_root = tmp_path / "pred"
    ref_dir = tmp_path / "refs"
    output_dir = tmp_path / "rmsd"
    pred_root.mkdir()
    ref_dir.mkdir()

    ref_path = ref_dir / "1UBQ.pdb"
    pred_path = pred_root / "sample_1_model.pdb"
    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    _write_pdb(ref_path, coords)
    _write_pdb(pred_path, coords)

    sample = SimpleNamespace(
        sample_name="sample_1",
        query_name="1UBQ",
        seed_name="seed_1",
        model_path=pred_path,
        avg_plddt=91.0,
        ptm=0.8,
        iptm=0.7,
        sample_ranking_score=0.9,
        gpde=0.1,
        has_clash=0.0,
    )
    monkeypatch.setattr(
        "openfold3_length_benchmark.benchmarking.collect_samples",
        lambda output_dir: [sample],
    )

    run_rmsd_benchmark(
        pred_root=pred_root,
        ref_dir=ref_dir,
        output_dir=output_dir,
        atom_set="ca",
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "rmsd_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["query"] == "1UBQ"
    assert rows[0]["coverage"]["matched_atom_count"] == 3
    assert rows[0]["aggregated_confidence"]["sample_ranking_score"] == 0.9
