import json
from pathlib import Path

import openfold3.benchmark.methods as benchmark_methods
from openfold3.benchmark.cif_utils import (
    parse_atom_site_records,
    parse_pdb_atom_records,
    summarize_structure,
)
from openfold3.benchmark.harness import DdgBenchmarkHarness, MethodResult
from openfold3.benchmark.methods import (
    ExternalToolMethod,
    FoldXBuildModelMethod,
    HeliXonBindingDdgMethod,
    OpenFoldConfidenceMethod,
    PromptDdgMethod,
    RosettaScoreMethod,
    Saambe3DMethod,
    StructureInterfaceMethod,
    multiscale_methods,
    _resolve_executable_path,
    _resolve_rosetta_binary,
    _resolve_rosetta_database,
)
from openfold3.benchmark.models import BenchmarkCase, MutationInput


def _write_minimal_cif(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "data_structure",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_entity_id",
                "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.auth_seq_id",
                "_atom_site.auth_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.auth_atom_id",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.pdbx_formal_charge",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.pdbx_PDB_model_num",
                "_atom_site.id",
                "ATOM C CA . LEU A 1 1 . 1 LEU A CA 90.0 ? 0.0 0.0 0.0 1 1",
                "ATOM C CB . LEU A 1 1 . 1 LEU A CB 80.0 ? 0.0 0.0 1.5 1 2",
                "ATOM C CA . LEU B 2 1 . 1 LEU B CA 91.0 ? 0.0 0.0 7.0 1 3",
                "ATOM C CB . LEU B 2 1 . 1 LEU B CB 81.0 ? 0.0 0.0 5.0 1 4",
                "#",
            ]
        ),
        encoding="utf-8",
    )


def _write_foldx_ready_cif(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "data_structure",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_entity_id",
                "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.auth_seq_id",
                "_atom_site.auth_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.auth_atom_id",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.pdbx_formal_charge",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.pdbx_PDB_model_num",
                "_atom_site.id",
                "ATOM N N . LEU A 1 1 . 1 LEU A N 90.0 ? 0.000 0.000 0.000 1 1",
                "ATOM C CA . LEU A 1 1 . 1 LEU A CA 90.0 ? 1.458 0.000 0.000 1 2",
                "ATOM C C . LEU A 1 1 . 1 LEU A C 90.0 ? 1.958 1.420 0.000 1 3",
                "ATOM O O . LEU A 1 1 . 1 LEU A O 90.0 ? 1.200 2.360 0.000 1 4",
                "ATOM C CB . LEU A 1 1 . 1 LEU A CB 80.0 ? 1.958 -0.780 -1.220 1 5",
                "ATOM N N . LEU B 2 1 . 1 LEU B N 91.0 ? 5.000 0.000 0.000 1 6",
                "ATOM C CA . LEU B 2 1 . 1 LEU B CA 91.0 ? 6.458 0.000 0.000 1 7",
                "ATOM C C . LEU B 2 1 . 1 LEU B C 91.0 ? 6.958 1.420 0.000 1 8",
                "ATOM O O . LEU B 2 1 . 1 LEU B O 91.0 ? 6.200 2.360 0.000 1 9",
                "ATOM C CB . LEU B 2 1 . 1 LEU B CB 81.0 ? 6.958 -0.780 -1.220 1 10",
                "#",
            ]
        ),
        encoding="utf-8",
    )


def _write_minimal_pdb(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ATOM      1  N   LEU A   1       0.000   0.000   0.000  1.00 90.00           N",
                "ATOM      2  CA  LEU A   1       1.458   0.000   0.000  1.00 90.00           C",
                "ATOM      3  C   LEU A   1       1.958   1.420   0.000  1.00 90.00           C",
                "ATOM      4  O   LEU A   1       1.200   2.360   0.000  1.00 90.00           O",
                "ATOM      5  CB  LEU A   1       1.958  -0.780  -1.220  1.00 80.00           C",
                "ATOM      6  N   LEU B   1       5.000   0.000   0.000  1.00 91.00           N",
                "ATOM      7  CA  LEU B   1       6.458   0.000   0.000  1.00 91.00           C",
                "ATOM      8  C   LEU B   1       6.958   1.420   0.000  1.00 91.00           C",
                "ATOM      9  O   LEU B   1       6.200   2.360   0.000  1.00 91.00           O",
                "ATOM     10  CB  LEU B   1       6.958  -0.780  -1.220  1.00 81.00           C",
                "TER",
                "END",
            ]
        ),
        encoding="utf-8",
    )


def test_parse_atom_site_records_reads_atoms(tmp_path):
    cif_path = tmp_path / "mini.cif"
    _write_minimal_cif(cif_path)

    atoms = parse_atom_site_records(cif_path)

    assert len(atoms) == 4
    assert atoms[0].chain_id == "A"
    assert atoms[2].chain_id == "B"
    assert atoms[0].atom_name == "CA"


def test_summarize_structure_computes_contacts(tmp_path):
    cif_path = tmp_path / "mini.cif"
    _write_minimal_cif(cif_path)

    summary = summarize_structure(cif_path)

    assert summary.atom_count == 4
    assert summary.residue_count == 2
    assert summary.interface_atom_contacts_5a >= 1
    assert summary.interface_ca_contacts_8a == 1
    assert summary.min_inter_chain_atom_distance == 3.5


def test_parse_pdb_atom_records_reads_atoms(tmp_path):
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)

    atoms = parse_pdb_atom_records(pdb_path)

    assert len(atoms) == 10
    assert atoms[0].chain_id == "A"
    assert atoms[5].chain_id == "B"
    assert atoms[0].atom_name == "N"


def test_harness_runs_local_methods(tmp_path):
    cif_path = tmp_path / "mini.cif"
    confidence_path = tmp_path / "confidence.json"
    _write_minimal_cif(cif_path)
    confidence_path.write_text(
        json.dumps(
            {
                "avg_plddt": 89.9,
                "iptm": 0.78,
                "ptm": 0.80,
                "gpde": 0.41,
                "has_clash": 0.0,
                "sample_ranking_score": 1.17,
            }
        ),
        encoding="utf-8",
    )
    case = BenchmarkCase(
        case_id="mini",
        structure_path=cif_path,
        confidence_path=confidence_path,
    )
    harness = DdgBenchmarkHarness(
        methods=[OpenFoldConfidenceMethod(), StructureInterfaceMethod()]
    )

    report = harness.run_case(case)

    assert report.case_id == "mini"
    assert report.structure_summary["chain_lengths"] == {"A": 1, "B": 1}
    assert report.results[0].status == "ok"
    assert report.results[0].score == 1.17
    assert report.results[1].score == 1.0


def test_external_tool_requires_mutation_and_binary(tmp_path):
    cif_path = tmp_path / "mini.cif"
    _write_minimal_cif(cif_path)
    case = BenchmarkCase(case_id="mini", structure_path=cif_path)
    harness = DdgBenchmarkHarness(
        methods=[ExternalToolMethod(name="foldx", executable="foldx")]
    )

    report = harness.run_case(case)

    assert report.results[0].status == "unavailable"
    assert report.results[0].details["reason"] == "mutation_spec_missing"


def test_external_tool_reaches_pdb_conversion_requirement(tmp_path):
    cif_path = tmp_path / "mini.cif"
    _write_minimal_cif(cif_path)
    case = BenchmarkCase(
        case_id="mini",
        structure_path=cif_path,
        mutations=(MutationInput("A", "L", 1, "A"),),
    )
    harness = DdgBenchmarkHarness(
        methods=[ExternalToolMethod(name="foldx", executable="foldx")]
    )

    report = harness.run_case(case)

    assert report.results[0].details["reason"] == "pdb_conversion_required"


def test_resolve_executable_prefers_environment_override(tmp_path, monkeypatch):
    fake_foldx = tmp_path / "foldx"
    fake_foldx.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_foldx.chmod(0o755)

    monkeypatch.setenv("FOLDX_BINARY", str(fake_foldx))

    assert _resolve_executable_path("foldx", env_var_name="FOLDX_BINARY") == str(fake_foldx)


def test_external_tool_resolves_local_foldx_install_for_pdb(tmp_path):
    structure_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(structure_path)
    case = BenchmarkCase(
        case_id="mini-pdb",
        structure_path=structure_path,
        mutations=(MutationInput("A", "L", 1, "A"),),
    )
    harness = DdgBenchmarkHarness(methods=[ExternalToolMethod(name="foldx", executable="foldx")])

    report = harness.run_case(case)

    assert report.results[0].status == "ready"
    assert report.results[0].details["resolved_executable"].endswith("/tools/bin/foldx")


def test_foldx_buildmodel_runs_for_local_pdb(tmp_path):
    if _resolve_executable_path("foldx", env_var_name="FOLDX_BINARY") is None:
        raise AssertionError("Expected local FoldX installation for integration test")

    structure_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(structure_path)
    case = BenchmarkCase(
        case_id="mini-buildmodel",
        structure_path=structure_path,
        mutations=(MutationInput("A", "L", 1, "A"),),
    )
    harness = DdgBenchmarkHarness(methods=[FoldXBuildModelMethod()])

    report = harness.run_case(case)

    assert report.results[0].status == "ok"
    assert report.results[0].units == "kcal/mol"
    assert isinstance(report.results[0].score, float)
    assert Path(report.results[0].details["dif_path"]).exists()
    assert Path(report.results[0].details["mutant_model_path"]).exists()
    assert Path(report.results[0].details["wt_model_path"]).exists()
    assert Path(report.results[0].details["mutant_summary_path"]).exists()
    assert Path(report.results[0].details["wt_summary_path"]).exists()
    assert report.results[0].details["protocol"] == "BuildModel+AnalyseComplex"


def test_foldx_buildmodel_runs_for_local_cif(tmp_path):
    if _resolve_executable_path("foldx", env_var_name="FOLDX_BINARY") is None:
        raise AssertionError("Expected local FoldX installation for integration test")

    structure_path = tmp_path / "mini.cif"
    _write_foldx_ready_cif(structure_path)
    case = BenchmarkCase(
        case_id="mini-buildmodel-cif",
        structure_path=structure_path,
        mutations=(MutationInput("A", "L", 1, "A"),),
    )
    harness = DdgBenchmarkHarness(methods=[FoldXBuildModelMethod()])

    report = harness.run_case(case)

    assert report.results[0].status == "ok"
    assert report.results[0].details["prepared_from_cif"] is True
    assert Path(report.results[0].details["prepared_input_pdb_path"]).exists()


def test_resolve_rosetta_binary_and_database_prefer_environment_override(tmp_path, monkeypatch):
    fake_binary = tmp_path / "score_jd2"
    fake_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_binary.chmod(0o755)
    fake_database = tmp_path / "database"
    fake_database.mkdir()

    monkeypatch.setenv("ROSETTA_SCORE_JD2_BINARY", str(fake_binary))
    monkeypatch.setenv("ROSETTA_DATABASE", str(fake_database))

    assert _resolve_rosetta_binary("score_jd2", env_var_name="ROSETTA_SCORE_JD2_BINARY") == str(fake_binary)
    assert _resolve_rosetta_database() == str(fake_database)


def test_rosetta_score_method_runs_with_fake_score_binary(tmp_path, monkeypatch):
    fake_binary = tmp_path / "score_jd2"
    fake_binary.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "scorefile='' ",
                "while [ $# -gt 0 ]; do",
                "  if [ \"$1\" = \"-out:file:scorefile\" ]; then",
                "    scorefile=\"$2\"",
                "    shift 2",
                "    continue",
                "  fi",
                "  shift",
                "done",
                "mkdir -p \"$(dirname \"$scorefile\")\"",
                "cat > \"$scorefile\" <<'EOF'",
                "SCORE: total_score description",
                "SCORE: -123.45 model",
                "EOF",
                "echo fake_rosetta_done",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_binary.chmod(0o755)
    fake_database = tmp_path / "database"
    fake_database.mkdir()
    monkeypatch.setenv("ROSETTA_SCORE_JD2_BINARY", str(fake_binary))
    monkeypatch.setenv("ROSETTA_DATABASE", str(fake_database))

    structure_path = tmp_path / "mini.cif"
    _write_foldx_ready_cif(structure_path)
    case = BenchmarkCase(
        case_id="mini-rosetta",
        structure_path=structure_path,
        mutations=(MutationInput("A", "L", 1, "A"),),
    )
    harness = DdgBenchmarkHarness(methods=[RosettaScoreMethod()])

    report = harness.run_case(case)

    assert report.results[0].status == "ok"
    assert report.results[0].score == -123.45
    assert report.results[0].units == "rosetta_energy"
    assert report.results[0].details["prepared_from_cif"] is True
    assert Path(report.results[0].details["scorefile_path"]).exists()


def test_rosetta_score_method_infers_database_from_binary_location(tmp_path, monkeypatch):
    rosetta_root = tmp_path / "rosetta.binary.ubuntu.release-408" / "main"
    fake_binary = rosetta_root / "source" / "bin" / "score_jd2"
    fake_binary.parent.mkdir(parents=True, exist_ok=True)
    fake_binary.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "scorefile=''",
                "while [ $# -gt 0 ]; do",
                "  if [ \"$1\" = \"-out:file:scorefile\" ]; then",
                "    scorefile=\"$2\"",
                "    shift 2",
                "    continue",
                "  fi",
                "  shift",
                "done",
                "mkdir -p \"$(dirname \"$scorefile\")\"",
                "cat > \"$scorefile\" <<'EOF'",
                "SCORE: total_score description",
                "SCORE: -10.0 model",
                "EOF",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_binary.chmod(0o755)
    (rosetta_root / "database").mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("ROSETTA_SCORE_JD2_BINARY", str(fake_binary))
    monkeypatch.delenv("ROSETTA_DATABASE", raising=False)
    monkeypatch.setattr(benchmark_methods, "_resolve_rosetta_database", lambda: None)

    structure_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(structure_path)
    case = BenchmarkCase(case_id="mini-rosetta-infer-db", structure_path=structure_path)
    harness = DdgBenchmarkHarness(methods=[RosettaScoreMethod()])

    report = harness.run_case(case)

    assert report.results[0].status == "ok"
    assert report.results[0].score == -10.0
    assert report.results[0].details["database_inferred_from_binary"] is True


def test_saambe_3d_method_runs_with_fake_script(tmp_path, monkeypatch):
    fake_script = tmp_path / "fake_saambe.py"
    fake_script.write_text(
        "\n".join(
            [
                "import pathlib",
                "import sys",
                "",
                "args = sys.argv[1:]",
                "output = pathlib.Path(args[args.index('-o') + 1])",
                "output.write_text('1.23 Destabilizing\\n', encoding='utf-8')",
                "print('fake_saambe_done')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("SAAMBE_3D_SCRIPT", str(fake_script))

    structure_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(structure_path)
    case = BenchmarkCase(
        case_id="mini-saambe",
        structure_path=structure_path,
        mutations=(MutationInput("A", "L", 1, "A"),),
    )
    harness = DdgBenchmarkHarness(methods=[Saambe3DMethod()])

    report = harness.run_case(case)

    assert report.results[0].status == "ok"
    assert report.results[0].score == 1.23
    assert report.results[0].units == "kcal/mol"
    assert report.results[0].details["prediction_label"] == "Destabilizing"
    assert Path(report.results[0].details["output_path"]).exists()


def test_prompt_ddg_method_reports_missing_checkpoint(tmp_path, monkeypatch):
    fake_wrapper = tmp_path / "prompt_ddg_infer.py"
    fake_wrapper.write_text("print('noop')\n", encoding="utf-8")
    monkeypatch.setenv("PROMPT_DDG_INFER_SCRIPT", str(fake_wrapper))
    original_resolve_existing_path = benchmark_methods._resolve_existing_path

    def _fake_resolve_existing_path(env_var_name=None, relative_path=None):
        if env_var_name == "PROMPT_DDG_CHECKPOINT":
            return None
        return original_resolve_existing_path(
            env_var_name=env_var_name,
            relative_path=relative_path,
        )

    monkeypatch.setattr(
        benchmark_methods,
        "_resolve_existing_path",
        _fake_resolve_existing_path,
    )

    structure_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(structure_path)
    case = BenchmarkCase(
        case_id="mini-prompt",
        structure_path=structure_path,
        mutations=(MutationInput("A", "L", 1, "A"),),
    )
    harness = DdgBenchmarkHarness(methods=[PromptDdgMethod()])

    report = harness.run_case(case)

    assert report.results[0].status == "unavailable"
    assert report.results[0].details["reason"] == "prompt_ddg_checkpoint_not_found"


def test_prompt_ddg_method_runs_with_fake_wrapper(tmp_path, monkeypatch):
    fake_wrapper = tmp_path / "prompt_ddg_infer.py"
    fake_wrapper.write_text(
        "\n".join(
            [
                "import json",
                "print(json.dumps({'status': 'ok', 'score': 2.5, 'fold_scores': [2.0, 2.5, 3.0]}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_checkpoint = tmp_path / "ddg_model.ckpt"
    fake_checkpoint.write_text("fake", encoding="utf-8")
    monkeypatch.setenv("PROMPT_DDG_INFER_SCRIPT", str(fake_wrapper))
    monkeypatch.setenv("PROMPT_DDG_CHECKPOINT", str(fake_checkpoint))

    structure_path = tmp_path / "mini.cif"
    _write_foldx_ready_cif(structure_path)
    case = BenchmarkCase(
        case_id="mini-prompt-ok",
        structure_path=structure_path,
        mutations=(MutationInput("A", "L", 1, "A"),),
    )
    harness = DdgBenchmarkHarness(methods=[PromptDdgMethod()])

    report = harness.run_case(case)

    assert report.results[0].status == "ok"
    assert report.results[0].score == 2.5
    assert report.results[0].units == "kcal/mol"
    assert report.results[0].details["prepared_from_cif"] is True
    assert report.results[0].details["payload"]["fold_scores"] == [2.0, 2.5, 3.0]
    assert Path(report.results[0].details["prepared_input_pdb_path"]).exists()


def test_helixon_method_runs_with_fake_predictor(tmp_path, monkeypatch):
    fake_script = tmp_path / "fake_helixon.py"
    fake_script.write_text(
        "\n".join(
            [
                "print('Predicted ddG: -0.30')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_model = tmp_path / "model.pt"
    fake_model.write_text("fake", encoding="utf-8")
    wt_model = tmp_path / "wt.pdb"
    mut_model = tmp_path / "mut.pdb"
    _write_minimal_pdb(wt_model)
    _write_minimal_pdb(mut_model)

    def _fake_foldx_run(self, context):
        return MethodResult(
            method="foldx",
            status="ok",
            score=0.0,
            units="kcal/mol",
            details={
                "wt_model_path": str(wt_model),
                "mutant_model_path": str(mut_model),
                "runtime_seconds": 0.01,
            },
        )

    monkeypatch.setenv("HELIXON_BINDING_DDG_SCRIPT", str(fake_script))
    monkeypatch.setenv("HELIXON_BINDING_DDG_MODEL", str(fake_model))
    monkeypatch.setattr(FoldXBuildModelMethod, "run", _fake_foldx_run)

    structure_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(structure_path)
    case = BenchmarkCase(
        case_id="mini-helixon",
        structure_path=structure_path,
        mutations=(MutationInput("A", "L", 1, "A"),),
    )
    harness = DdgBenchmarkHarness(methods=[HeliXonBindingDdgMethod()])

    report = harness.run_case(case)

    assert report.results[0].status == "ok"
    assert report.results[0].score == -0.30
    assert report.results[0].units == "kcal/mol"
    assert report.results[0].details["wt_model_path"] == str(wt_model)
    assert report.results[0].details["mutant_model_path"] == str(mut_model)


def test_multiscale_methods_include_five_primary_scales():
    names = [method.name for method in multiscale_methods()]

    assert names == [
        "foldx",
        "rosetta_score",
        "saambe_3d",
        "helixon_binding_ddg",
        "prompt_ddg",
    ]
