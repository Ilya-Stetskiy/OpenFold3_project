from __future__ import annotations

from pathlib import Path

import pytest

from openfold3.benchmark.cif_utils import parse_atom_site_records
from openfold3.benchmark.harness import DdgBenchmarkHarness
from openfold3.benchmark.methods import FoldXBuildModelMethod, _resolve_executable_path
from openfold3.benchmark.models import BenchmarkCase, MutationInput


def _write_openfold_like_cif(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "data_openfold_like",
                "#",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_seq_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.B_iso_or_equiv",
                "ATOM 1 N N LEU A 1 0.000 0.000 0.000 90.0",
                "ATOM 2 C CA LEU A 1 1.458 0.000 0.000 90.0",
                "ATOM 3 C C LEU A 1 1.958 1.420 0.000 90.0",
                "ATOM 4 O O LEU A 1 1.200 2.360 0.000 90.0",
                "ATOM 5 C CB LEU A 1 1.958 -0.780 -1.220 80.0",
                "ATOM 6 N N LEU B 1 5.000 0.000 0.000 91.0",
                "ATOM 7 C CA LEU B 1 6.458 0.000 0.000 91.0",
                "ATOM 8 C C LEU B 1 6.958 1.420 0.000 91.0",
                "ATOM 9 O O LEU B 1 6.200 2.360 0.000 91.0",
                "ATOM 10 C CB LEU B 1 6.958 -0.780 -1.220 81.0",
                "#",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_parse_atom_site_records_accepts_openfold_label_only_columns(tmp_path: Path) -> None:
    cif_path = tmp_path / "openfold_like_model.cif"
    _write_openfold_like_cif(cif_path)

    atoms = parse_atom_site_records(cif_path)

    assert len(atoms) == 10
    assert atoms[0].chain_id == "A"
    assert atoms[0].residue_name == "LEU"
    assert atoms[0].atom_name == "N"
    assert atoms[5].chain_id == "B"


def test_foldx_buildmodel_runs_for_openfold_like_label_only_cif(tmp_path: Path) -> None:
    if _resolve_executable_path("foldx", env_var_name="FOLDX_BINARY") is None:
        pytest.skip("FoldX not available")

    cif_path = tmp_path / "openfold_like_model.cif"
    _write_openfold_like_cif(cif_path)
    case = BenchmarkCase(
        case_id="openfold-like-buildmodel-cif",
        structure_path=cif_path,
        mutations=(MutationInput("A", "L", 1, "A"),),
    )
    harness = DdgBenchmarkHarness(methods=[FoldXBuildModelMethod()])

    report = harness.run_case(case)

    assert report.results[0].status == "ok"
    assert report.results[0].details["prepared_from_cif"] is True
