from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from openfold3.benchmark.harness import MethodResult
from openfold3.experiment_pipeline.ddg_pipeline.adapters import (
    AVAILABLE_METHODS,
    BackendRunResult,
    DDGAdapter,
    ESM2Adapter,
    FoldXAdapter,
    RosettaDDGAdapter,
    default_adapters,
)
from openfold3.experiment_pipeline.ddg_pipeline.adapters.builtin import _discover_repaired_structure
from openfold3.experiment_pipeline.ddg_pipeline.data.canonical import CanonicalMutationRecord, StructurePaths, parse_record
from openfold3.experiment_pipeline.ddg_pipeline.data.mutation_engine import apply_mutation
from openfold3.experiment_pipeline.ddg_pipeline.main import main as ddg_main
from openfold3.experiment_pipeline.ddg_pipeline.runners.pipeline import (
    PipelineConfig,
    compute_metrics,
    ensure_layout,
    layout_tree,
    run_pipeline,
)
from openfold3.experiment_pipeline.ddg_pipeline.server_run import (
    ServerRunConfig,
    _parse_structure_sources,
    build_canonical_dataset,
    build_structure_dataset,
    merge_structure_results,
    run_openfold_shard,
    run_ddg_shard,
    split_csv,
)
from openfold3.experiment_pipeline.ddg_pipeline.structures.loader import extract_mutation_site, load_structure


DEMO_PDB = Path(__file__).parent / "test_data" / "ddg_pipeline" / "demo_input.pdb"


def _patch_repair_output(monkeypatch):
    real_subprocess_run = subprocess.run

    def _patched_subprocess_run(command, cwd=None, **kwargs):
        result = real_subprocess_run(command, cwd=cwd, **kwargs)
        if "--command" in command and command[command.index("--command") + 1] == "RepairPDB" and cwd is not None:
            pdb_name = command[command.index("--pdb") + 1]
            if "--output-dir" in command:
                output_dir = command[command.index("--output-dir") + 1]
                repaired = Path(cwd) / output_dir / f"{Path(pdb_name).stem}_Repair.pdb"
            else:
                repaired = Path(cwd) / f"{Path(pdb_name).stem}_Repair.pdb"
            repaired.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(cwd) / pdb_name, repaired)
        return result

    monkeypatch.setattr("openfold3.experiment_pipeline.ddg_pipeline.adapters.builtin.subprocess.run", _patched_subprocess_run)


def _single_record(structure_path: Path = DEMO_PDB) -> object:
    return parse_record(
        {
            "protein_id": "p1",
            "sequence": "L",
            "mutation": "L1A",
            "position": 1,
            "wt": "L",
            "mut": "A",
            "structure_paths": {
                "experimental": str(structure_path),
                "openfold": str(structure_path),
                "foldx": str(structure_path),
            },
            "experimental_ddg": -0.1,
        }
    )


def _write_noncontiguous_pdb(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ATOM      1  N   LEU A   1       0.000   0.000   0.000  1.00 90.00           N",
                "ATOM      2  CA  LEU A   1       1.458   0.000   0.000  1.00 90.00           C",
                "ATOM      3  C   LEU A   1       1.958   1.420   0.000  1.00 90.00           C",
                "ATOM      4  O   LEU A   1       1.200   2.360   0.000  1.00 90.00           O",
                "ATOM      5  CB  LEU A   1       1.958  -0.780  -1.220  1.00 80.00           C",
                "ATOM      6  N   LYS A   3       3.500   1.420   0.000  1.00 90.00           N",
                "ATOM      7  CA  LYS A   3       4.958   1.420   0.000  1.00 90.00           C",
                "ATOM      8  C   LYS A   3       5.458   2.840   0.000  1.00 90.00           C",
                "ATOM      9  O   LYS A   3       4.700   3.780   0.000  1.00 90.00           O",
                "ATOM     10  CB  LYS A   3       5.458   0.640  -1.220  1.00 80.00           C",
                "TER",
                "END",
            ]
        ),
        encoding="utf-8",
    )


@dataclass(frozen=True, slots=True)
class SequenceCountingAdapter(DDGAdapter):
    method_name: str = "sequence_test"
    sequence_based: bool = True

    def prepare_input(self, record, structure_source, work_dir):
        path = work_dir / "input.json"
        path.write_text(json.dumps({"structure_source": structure_source}, indent=2), encoding="utf-8")
        return path

    def run(self, prepared_input_path, record, structure_source, work_dir):
        raw_output_path = work_dir / "raw_output.json"
        raw_output_path.write_text(
            json.dumps(
                {
                    "backend_status": "ok",
                    "ddg": 1.0,
                    "ddg_std": 0.0,
                    "n_runs_requested": 1,
                    "n_runs_valid": 1,
                    "per_run_ddg": [1.0],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return BackendRunResult(status="ok", raw_output_path=raw_output_path)

    def parse_output(self, raw_output_path):
        return 1.0

    def extract_summary(self, raw_output_path):
        return {"ddg_std": 0.0, "n_runs_requested": 1, "n_runs_valid": 1, "per_run_ddg": [1.0]}

    def normalize(self, raw_value):
        return raw_value


def test_stage0_layout_and_tree(tmp_path, capsys):
    layout = ensure_layout(tmp_path / "ddg_pipeline_run")
    tree = layout_tree(tmp_path / "ddg_pipeline_run")
    for name in ("config", "data", "structures", "adapters", "runners", "outputs", "logs", "tests"):
        assert layout[name].exists()
        assert f"{name}/" in tree
    print(tree)
    assert "ddg_pipeline_run/" in capsys.readouterr().out


def test_stage1_default_methods_are_foldx_only():
    assert set(AVAILABLE_METHODS) == {"foldx", "rosetta", "esm2"}
    adapters = default_adapters()
    assert len(adapters) == 3
    assert any(isinstance(adapter, FoldXAdapter) for adapter in adapters)
    assert any(isinstance(adapter, RosettaDDGAdapter) for adapter in adapters)
    assert any(isinstance(adapter, ESM2Adapter) for adapter in adapters)


def test_stage2_rejects_position_mismatch():
    with pytest.raises(ValueError, match="Mutation string position 3 disagrees with explicit position 4"):
        parse_record(
            {
                "protein_id": "bad",
                "sequence": "ACDE",
                "mutation": "D3N",
                "position": 4,
                "wt": "D",
                "mut": "N",
                "structure_paths": {"experimental": None, "openfold": None, "foldx": None},
                "experimental_ddg": None,
            }
        )


def test_stage2_apply_mutation():
    assert apply_mutation("MAG", "A2V") == "MVG"
    assert apply_mutation("ALKK", "K3R") == "ALRK"


def test_stage3_extract_site_uses_real_residue_id(tmp_path):
    pdb_path = tmp_path / "noncontig.pdb"
    _write_noncontiguous_pdb(pdb_path)
    structure = load_structure(pdb_path)
    site = extract_mutation_site(structure, 3, expected_wt="K")
    assert site.chain_id == "A"
    assert site.residue_id == "3"


def test_stage3_missing_residue_fails_explicitly(tmp_path):
    pdb_path = tmp_path / "noncontig.pdb"
    _write_noncontiguous_pdb(pdb_path)
    structure = load_structure(pdb_path)
    with pytest.raises(ValueError, match="Could not find residue id 2"):
        extract_mutation_site(structure, 2, expected_wt="A")


def test_repaired_structure_is_used_test(tmp_path, monkeypatch):
    record = _single_record()
    adapter = FoldXAdapter(number_of_runs=2)
    expected_repaired_path = (tmp_path / "foldx" / "repair" / "demo_input_Repair.pdb").resolve()

    def _fake_run(self, context):
        assert context.case.structure_path.resolve() == expected_repaired_path
        work_dir = tmp_path / "benchmark"
        work_dir.mkdir(parents=True, exist_ok=True)
        raw_path = work_dir / "Raw.fxout"
        raw_path.write_text(
            "Pdb\ttotal energy\n"
            "demo_input_1_0.pdb\t2.0\n"
            "WT_demo_input_1_0.pdb\t1.0\n",
            encoding="utf-8",
        )
        return MethodResult(
            method="foldx",
            status="ok",
            score=1.0,
            units="kcal/mol",
            details={
                "work_dir": str(work_dir),
                "generated_mutant_models": [str((work_dir / "demo_input_1_0.pdb").resolve())],
                "generated_wt_models": [str((work_dir / "WT_demo_input_1_0.pdb").resolve())],
                "model_pairing": [
                    {
                        "run_index": 0,
                        "mutant_model_path": str((work_dir / "demo_input_1_0.pdb").resolve()),
                        "wt_model_path": str((work_dir / "WT_demo_input_1_0.pdb").resolve()),
                    }
                ],
                "per_run_buildmodel_ddg": [1.0],
                "raw_path": str(raw_path),
                "stdout_tail": "ok",
            },
        )

    _patch_repair_output(monkeypatch)
    monkeypatch.setattr("openfold3.experiment_pipeline.ddg_pipeline.adapters.builtin.FoldXBuildModelMethod.run", _fake_run)

    result = adapter.predict(record, "experimental", tmp_path / "foldx")
    raw_payload = json.loads(result.raw_output_path.read_text(encoding="utf-8"))
    assert raw_payload["buildmodel_input_path"] == raw_payload["repaired_structure_path"]
    assert Path(raw_payload["repaired_structure_path"]).resolve() == expected_repaired_path


def test_multiple_runs_pairing_test(tmp_path, monkeypatch):
    _patch_repair_output(monkeypatch)
    record = _single_record()
    result = FoldXAdapter(number_of_runs=5).predict(record, "experimental", tmp_path / "foldx")
    raw_payload = json.loads(result.raw_output_path.read_text(encoding="utf-8"))
    pairing = raw_payload["model_pairing"]
    assert len(pairing) == 5
    assert len(raw_payload["generated_mutant_models"]) == 5
    assert len(raw_payload["generated_wt_models"]) == 5
    repaired_stem = Path(raw_payload["repaired_structure_path"]).stem
    for index, pair in enumerate(pairing):
        assert Path(pair["mutant_model_path"]).name.startswith(f"{repaired_stem}_1_")
        assert Path(pair["wt_model_path"]).name.startswith(f"WT_{repaired_stem}_1_")
        assert pair["run_index"] == index


def test_multiple_runs_aggregation_test(tmp_path, monkeypatch):
    _patch_repair_output(monkeypatch)
    record = _single_record()
    result = FoldXAdapter(number_of_runs=5).predict(record, "experimental", tmp_path / "foldx")
    raw_payload = json.loads(result.raw_output_path.read_text(encoding="utf-8"))
    per_run = [float(value) for value in raw_payload["per_run_ddg"]]
    assert raw_payload["n_runs_requested"] == 5
    assert raw_payload["n_runs_valid"] >= 1
    expected_mean = sum(per_run) / len(per_run)
    assert math.isclose(float(raw_payload["ddg"]), expected_mean, rel_tol=1e-9, abs_tol=1e-12)


def test_stage6_sequence_based_adapter_runs_once(tmp_path):
    record = _single_record()
    config = PipelineConfig(
        output_root=tmp_path / "run",
        methods=("sequence_test",),
        structure_sources=("experimental", "openfold", "foldx"),
        seed=17,
    )
    run_pipeline([record], [SequenceCountingAdapter()], config)
    rows = list(csv.DictReader((config.output_root / "outputs" / "results.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["structure_source"] == "sequence_only"


def test_fail_fast_on_missing_residue_test(tmp_path, monkeypatch):
    pdb_path = tmp_path / "noncontig.pdb"
    _write_noncontiguous_pdb(pdb_path)
    record = parse_record(
        {
            "protein_id": "bad",
            "sequence": "LA",
            "mutation": "A2V",
            "position": 2,
            "wt": "A",
            "mut": "V",
            "structure_paths": {"experimental": str(pdb_path), "openfold": str(pdb_path), "foldx": str(pdb_path)},
            "experimental_ddg": 0.1,
        }
    )
    monkeypatch.setattr(FoldXAdapter, "validate_environment", lambda self: None)
    with pytest.raises(RuntimeError, match="Could not find residue id 2"):
        run_pipeline(
            [record],
            [FoldXAdapter()],
            PipelineConfig(output_root=tmp_path / "run", methods=("foldx",), structure_sources=("experimental",), seed=17),
        )


def test_fail_fast_on_missing_foldx_test(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "openfold3.experiment_pipeline.ddg_pipeline.adapters.builtin._resolve_executable_path",
        lambda executable, env_var_name=None: None,
    )
    adapter = FoldXAdapter()
    with pytest.raises(RuntimeError, match="FoldX executable is required but was not found"):
        adapter.validate_environment()

    result = adapter.predict(_single_record(), "experimental", tmp_path / "foldx_missing")
    assert result.status == "failed"
    assert result.success is False
    assert "required but was not found" in str(result.error_message)


def test_real_repairpdb_materializes_repaired_structure_test(tmp_path):
    record = _single_record()
    result = FoldXAdapter(number_of_runs=2).predict(record, "experimental", tmp_path / "foldx")
    raw_payload = json.loads(result.raw_output_path.read_text(encoding="utf-8"))
    assert result.status == "ok"
    assert result.success is True
    assert Path(raw_payload["repaired_structure_path"]).exists()
    assert raw_payload["buildmodel_input_path"] == raw_payload["method_result"]["details"]["prepared_input_pdb_path"]
    assert Path(raw_payload["buildmodel_input_path"]).exists()
    assert Path(raw_payload["execution_trace_path"]).exists()


def test_repair_discovery_exact_path(tmp_path):
    repaired = tmp_path / "demo_input_Repair.pdb"
    repaired.write_text("ok\n", encoding="utf-8")
    assert _discover_repaired_structure(tmp_path, "demo_input") == repaired.resolve()


def test_repair_discovery_search_fallback(tmp_path):
    repaired = tmp_path / "nested" / "demo_input_Repair.pdb"
    repaired.parent.mkdir(parents=True, exist_ok=True)
    repaired.write_text("ok\n", encoding="utf-8")
    assert _discover_repaired_structure(tmp_path, "demo_input") == repaired.resolve()


def test_repair_discovery_ambiguous_matches_fail(tmp_path):
    first = tmp_path / "a" / "demo_input_Repair.pdb"
    second = tmp_path / "b" / "demo_input_Repair.pdb"
    first.parent.mkdir(parents=True, exist_ok=True)
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("1\n", encoding="utf-8")
    second.write_text("2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Ambiguous repaired structure candidates"):
        _discover_repaired_structure(tmp_path, "demo_input")


def test_repair_discovery_zero_matches_fail(tmp_path):
    with pytest.raises(FileNotFoundError, match="Could not locate repaired structure"):
        _discover_repaired_structure(tmp_path, "demo_input")


def test_stale_repaired_structure_is_not_reused_on_rerun(tmp_path, monkeypatch):
    record = _single_record()
    adapter = FoldXAdapter(number_of_runs=2)
    first = adapter.predict(record, "experimental", tmp_path / "foldx")
    assert first.status == "ok"

    def _fake_subprocess_run(command, cwd=None, capture_output=True, text=True, **kwargs):
        if "--command" in command and command[command.index("--command") + 1] == "RepairPDB":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="fake success\nOutput PDB ./demo_input_Repair.pdb\n",
                stderr="",
            )
        raise AssertionError("BuildModel must not start when repaired structure is missing")

    monkeypatch.setattr(
        "openfold3.experiment_pipeline.ddg_pipeline.adapters.builtin.subprocess.run",
        _fake_subprocess_run,
    )
    second = adapter.predict(record, "experimental", tmp_path / "foldx")
    raw_payload = json.loads(second.raw_output_path.read_text(encoding="utf-8"))
    assert second.status == "failed"
    assert second.error_message == "foldx_repaired_structure_missing"
    assert raw_payload["backend_status"] == "failed"


def test_broken_generated_model_list_test(tmp_path, monkeypatch):
    record = _single_record()
    adapter = FoldXAdapter()

    def _fake_run(self, context):
        work_dir = tmp_path / "benchmark"
        work_dir.mkdir(parents=True, exist_ok=True)
        raw_path = work_dir / "Raw.fxout"
        raw_path.write_text("Pdb\ttotal energy\nmutant_0.pdb\t1.0\n", encoding="utf-8")
        return MethodResult(
            method="foldx",
            status="failed",
            score=None,
            units="kcal/mol",
            details={"reason": "foldx_generated_model_pairing_failed", "error": "broken list", "raw_path": str(raw_path)},
        )

    monkeypatch.setattr("openfold3.experiment_pipeline.ddg_pipeline.adapters.builtin.FoldXBuildModelMethod.run", _fake_run)
    result = adapter.predict(record, "experimental", tmp_path / "foldx")
    assert result.status == "failed"
    assert result.success is False


def test_parsing_failure_test(tmp_path, monkeypatch):
    record = _single_record()
    adapter = FoldXAdapter()

    def _bad_run(self, prepared_input_path, record, structure_source, work_dir):
        raw_output_path = work_dir / "raw_output.json"
        raw_output_path.write_text("{bad json", encoding="utf-8")
        return BackendRunResult(status="ok", raw_output_path=raw_output_path)

    monkeypatch.setattr(FoldXAdapter, "run", _bad_run)
    with pytest.raises(json.JSONDecodeError):
        adapter.predict(record, "experimental", tmp_path / "foldx")


def test_artifact_completeness_test(tmp_path, monkeypatch):
    _patch_repair_output(monkeypatch)
    record = _single_record()
    config = PipelineConfig(output_root=tmp_path / "run", methods=("foldx",), structure_sources=("experimental",), seed=17)
    run_pipeline([record], [FoldXAdapter(number_of_runs=5)], config)

    raw_payload = json.loads(
        (
            tmp_path
            / "run"
            / "adapters"
            / "p1"
            / "L1A"
            / "experimental"
            / "foldx"
            / "raw_output.json"
        ).read_text(encoding="utf-8")
    )
    required_raw_fields = {
        "original_structure_path",
        "repaired_structure_path",
        "buildmodel_input_path",
        "mutation",
        "chain_id",
        "residue_id",
        "generated_mutant_models",
        "generated_wt_models",
        "model_pairing",
        "per_run_ddg",
        "ddg",
        "ddg_std",
        "n_runs_requested",
        "n_runs_valid",
        "backend_status",
        "backend_logs_path",
    }
    assert required_raw_fields.issubset(raw_payload)

    rows = list(csv.DictReader((config.output_root / "outputs" / "results.csv").open(encoding="utf-8")))
    assert rows[0].keys() == {
        "protein_id",
        "mutation",
        "representation",
        "structure_source",
        "method",
        "unit",
        "ddg_raw",
        "ddg",
        "ddg_std",
        "n_runs_requested",
        "n_runs_valid",
        "status",
        "error_message",
    }
    assert (config.output_root / "outputs" / "foldx_execution_trace.md").exists()
    assert (config.output_root / "outputs" / "foldx_model_pairing.json").exists()


def test_rosetta_fail_if_missing_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "openfold3.experiment_pipeline.ddg_pipeline.adapters.builtin._collect_rosetta_environment_check",
        lambda: {
            "ddg_monomer_binary": None,
            "relax_binary": None,
            "database_path": None,
            "missing_components": ["ddg_monomer.static.linuxgccrelease", "relax.static.linuxgccrelease", "rosetta_database"],
        },
    )
    adapter = RosettaDDGAdapter(number_of_runs=2, top_k=1)
    with pytest.raises(RuntimeError, match="Rosetta runtime is incomplete"):
        adapter.validate_environment()
    result = adapter.predict(_single_record(), "experimental", tmp_path / "rosetta_missing")
    assert result.status == "failed"
    assert result.success is False


def test_rosetta_output_parsed(tmp_path):
    adapter = RosettaDDGAdapter(number_of_runs=1, top_k=1)
    result = adapter.predict(_single_record(), "experimental", tmp_path / "rosetta")
    raw_payload = json.loads(result.raw_output_path.read_text(encoding="utf-8"))
    assert result.status == "ok"
    assert raw_payload["backend_status"] == "ok"
    assert isinstance(raw_payload["ddg"], float)
    assert raw_payload["unit"] == "REU"
    assert raw_payload["structure_source"] == "experimental"
    assert len(raw_payload["per_run_ddg"]) == 1
    assert raw_payload["rosetta_residue_number"] == 1


def test_rosetta_prepare_input_renumbers_noncontiguous_residues(tmp_path):
    pdb_path = tmp_path / "noncontig.pdb"
    _write_noncontiguous_pdb(pdb_path)
    record = parse_record(
        {
            "protein_id": "bad",
            "sequence": "LAK",
            "mutation": "K3A",
            "position": 3,
            "wt": "K",
            "mut": "A",
            "structure_paths": {"experimental": str(pdb_path), "openfold": str(pdb_path), "foldx": str(pdb_path)},
            "experimental_ddg": 0.1,
        }
    )
    prepared = RosettaDDGAdapter(number_of_runs=1, top_k=1).prepare_input(record, "experimental", tmp_path / "rosetta_prepare")
    payload = json.loads(prepared.read_text(encoding="utf-8"))
    assert payload["residue_id"] == "3"
    assert payload["rosetta_residue_number"] == 2
    assert payload["residue_mapping"]["A:1"] == 1
    assert payload["residue_mapping"]["A:3"] == 2
    assert Path(payload["input_pdb"]).exists()
    assert Path(payload["mutation_file"]).read_text(encoding="utf-8").splitlines()[2] == "K 2 A"


def test_rosetta_runs_parallel(tmp_path, monkeypatch):
    record = _single_record()
    adapter = RosettaDDGAdapter(number_of_runs=4, top_k=2)

    class DummyPool:
        def __init__(self, processes):
            self.processes = processes
            captured["processes"] = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, tasks):
            captured["n_tasks"] = len(tasks)
            return [
                {
                    "status": "ok",
                    "run_index": task["run_index"],
                    "command": ["ddg_monomer"],
                    "returncode": 0,
                    "stdout_path": str(Path(task["run_dir"]) / "stdout.txt"),
                    "stderr_path": str(Path(task["run_dir"]) / "stderr.txt"),
                    "prediction_path": str(Path(task["run_dir"]) / "ddg_predictions.out"),
                    "ddg": float(task["run_index"] + 1),
                    "wildtype_dg": 10.0 + task["run_index"],
                    "mutant_dg": 11.0 + task["run_index"],
                }
                for task in tasks
            ]

    captured: dict[str, int] = {}

    real_subprocess_run = subprocess.run

    def _patched_run(command, cwd=None, **kwargs):
        if Path(command[0]).name.startswith("relax") and cwd is not None:
            relax_dir = Path(command[command.index("-out:path:all") + 1])
            input_pdb = Path(command[command.index("-in:file:s") + 1])
            relax_dir.mkdir(parents=True, exist_ok=True)
            (relax_dir / f"{input_pdb.stem}_0001.pdb").write_text("ATOM\nEND\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        return real_subprocess_run(command, cwd=cwd, **kwargs)

    monkeypatch.setattr("openfold3.experiment_pipeline.ddg_pipeline.adapters.builtin.Pool", DummyPool)
    monkeypatch.setattr("openfold3.experiment_pipeline.ddg_pipeline.adapters.builtin.subprocess.run", _patched_run)
    result = adapter.predict(record, "experimental", tmp_path / "rosetta_parallel")
    assert result.status == "ok"
    assert captured["n_tasks"] == 4
    assert captured["processes"] >= 1


def test_rosetta_multiple_runs_aggregation(tmp_path, monkeypatch):
    record = _single_record()
    adapter = RosettaDDGAdapter(number_of_runs=4, top_k=2)

    class DummyPool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, tasks):
            wt_values = [10.0, 1.0, 10.0, 1.0]
            mut_values = [11.0, 11.0, 2.0, 2.0]
            return [
                {
                    "status": "ok",
                    "run_index": task["run_index"],
                    "command": ["ddg_monomer"],
                    "returncode": 0,
                    "stdout_path": str(Path(task["run_dir"]) / "stdout.txt"),
                    "stderr_path": str(Path(task["run_dir"]) / "stderr.txt"),
                    "prediction_path": str(Path(task["run_dir"]) / "ddg_predictions.out"),
                    "ddg": mut_values[task["run_index"]] - wt_values[task["run_index"]],
                    "wildtype_dg": wt_values[task["run_index"]],
                    "mutant_dg": mut_values[task["run_index"]],
                }
                for task in tasks
            ]

    real_subprocess_run = subprocess.run

    def _patched_run(command, cwd=None, **kwargs):
        if Path(command[0]).name.startswith("relax") and cwd is not None:
            relax_dir = Path(command[command.index("-out:path:all") + 1])
            input_pdb = Path(command[command.index("-in:file:s") + 1])
            relax_dir.mkdir(parents=True, exist_ok=True)
            (relax_dir / f"{input_pdb.stem}_0001.pdb").write_text("ATOM\nEND\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        return real_subprocess_run(command, cwd=cwd, **kwargs)

    monkeypatch.setattr("openfold3.experiment_pipeline.ddg_pipeline.adapters.builtin.Pool", DummyPool)
    monkeypatch.setattr("openfold3.experiment_pipeline.ddg_pipeline.adapters.builtin.subprocess.run", _patched_run)
    result = adapter.predict(record, "experimental", tmp_path / "rosetta_agg")
    raw_payload = json.loads(result.raw_output_path.read_text(encoding="utf-8"))
    assert result.status == "ok"
    assert raw_payload["per_run_ddg"] == [1.0, 10.0, -8.0, 1.0]
    assert raw_payload["selected_top_k_wildtype_dg"] == [1.0, 1.0]
    assert raw_payload["selected_top_k_mutant_dg"] == [2.0, 2.0]
    assert math.isclose(raw_payload["ddg"], 1.0, rel_tol=1e-9)
    assert raw_payload["n_runs_requested"] == 4
    assert raw_payload["n_runs_valid"] == 4


def test_rosetta_not_using_threads_only(tmp_path, monkeypatch):
    record = _single_record()
    adapter = RosettaDDGAdapter(number_of_runs=2, top_k=1)

    class DummyPool:
        def __init__(self, processes):
            captured["pool_used"] = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, tasks):
            return [
                {
                    "status": "ok",
                    "run_index": task["run_index"],
                    "command": ["ddg_monomer"],
                    "returncode": 0,
                    "stdout_path": str(Path(task["run_dir"]) / "stdout.txt"),
                    "stderr_path": str(Path(task["run_dir"]) / "stderr.txt"),
                    "prediction_path": str(Path(task["run_dir"]) / "ddg_predictions.out"),
                    "ddg": 1.0,
                    "wildtype_dg": 10.0,
                    "mutant_dg": 11.0,
                }
                for task in tasks
            ]

    captured = {"pool_used": False}
    real_subprocess_run = subprocess.run

    def _patched_run(command, cwd=None, **kwargs):
        if Path(command[0]).name.startswith("relax") and cwd is not None:
            relax_dir = Path(command[command.index("-out:path:all") + 1])
            input_pdb = Path(command[command.index("-in:file:s") + 1])
            relax_dir.mkdir(parents=True, exist_ok=True)
            (relax_dir / f"{input_pdb.stem}_0001.pdb").write_text("ATOM\nEND\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        return real_subprocess_run(command, cwd=cwd, **kwargs)

    monkeypatch.setattr("openfold3.experiment_pipeline.ddg_pipeline.adapters.builtin.Pool", DummyPool)
    monkeypatch.setattr("openfold3.experiment_pipeline.ddg_pipeline.adapters.builtin.subprocess.run", _patched_run)
    adapter.predict(record, "experimental", tmp_path / "rosetta_pool")
    assert captured["pool_used"] is True


def _sequence_only_record() -> object:
    return parse_record(
        {
            "protein_id": "seq1",
            "sequence": "MKTLLA",
            "mutation": "T3A",
            "position": 3,
            "wt": "T",
            "mut": "A",
            "structure_paths": {"experimental": None, "openfold": None, "foldx": None},
            "experimental_ddg": 0.5,
        }
    )


def test_esm2_runs_successfully(tmp_path):
    adapter = ESM2Adapter()
    result = adapter.predict(_sequence_only_record(), "sequence_only", tmp_path / "esm2")
    raw_payload = json.loads(result.raw_output_path.read_text(encoding="utf-8"))
    assert result.status == "ok"
    assert raw_payload["backend_status"] == "ok"
    assert isinstance(raw_payload["ddg"], float)
    assert raw_payload["unit"] == "log-probability"


def test_esm2_ddg_computed(tmp_path):
    adapter = ESM2Adapter()
    result = adapter.predict(_sequence_only_record(), "sequence_only", tmp_path / "esm2_ddg")
    raw_payload = json.loads(result.raw_output_path.read_text(encoding="utf-8"))
    assert result.ddg is not None
    assert math.isfinite(result.ddg)
    assert math.isclose(result.ddg, float(raw_payload["ddg"]), rel_tol=1e-9)


def test_esm2_single_execution_per_mutation(tmp_path):
    record = _sequence_only_record()
    config = PipelineConfig(
        output_root=tmp_path / "esm2_pipeline",
        methods=("esm2",),
        structure_sources=("experimental", "openfold", "foldx"),
        seed=17,
    )
    run_pipeline([record], [ESM2Adapter()], config)
    rows = list(csv.DictReader((config.output_root / "outputs" / "results.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["method"] == "esm2"
    assert rows[0]["structure_source"] == "sequence_only"
    assert rows[0]["representation"] == "sequence"
    assert rows[0]["unit"] == "log-probability"


def test_esm2_no_structure_dependency(tmp_path):
    adapter = ESM2Adapter()
    result = adapter.predict(_sequence_only_record(), "sequence_only", tmp_path / "esm2_no_structure")
    assert result.status == "ok"
    assert result.structure_source == "sequence_only"


def test_esm2_failure_if_invalid_sequence(tmp_path):
    adapter = ESM2Adapter()
    bad_record = CanonicalMutationRecord(
        protein_id="bad-esm",
        sequence="MKTLLA",
        mutation="A3V",
        position=3,
        wt="A",
        mut="V",
        structure_paths=StructurePaths(experimental=None, openfold=None, foldx=None),
        experimental_ddg=None,
    )
    with pytest.raises(ValueError, match="Wild-type residue mismatch"):
        adapter.prepare_input(bad_record, "sequence_only", tmp_path / "esm2_bad")


def test_foldx_and_rosetta_shared_schema(tmp_path, monkeypatch):
    _patch_repair_output(monkeypatch)
    record = _single_record()
    config = PipelineConfig(output_root=tmp_path / "mixed", methods=("foldx", "rosetta"), structure_sources=("experimental",), seed=17)
    run_pipeline([record], [FoldXAdapter(number_of_runs=2), RosettaDDGAdapter(number_of_runs=1, top_k=1)], config)
    rows = list(csv.DictReader((config.output_root / "outputs" / "results.csv").open(encoding="utf-8")))
    methods = {row["method"] for row in rows}
    assert methods == {"foldx", "rosetta"}
    for row in rows:
        assert set(row) == {
            "protein_id",
            "mutation",
            "representation",
            "structure_source",
            "method",
            "unit",
            "ddg_raw",
            "ddg",
            "ddg_std",
            "n_runs_requested",
            "n_runs_valid",
            "status",
            "error_message",
        }
        if row["method"] == "rosetta":
            assert row["representation"] == "structure"
            assert row["unit"] == "REU"
        if row["method"] == "foldx":
            assert row["unit"] == "kcal/mol"


def test_esm2_schema_matches_foldx_and_rosetta(tmp_path, monkeypatch):
    _patch_repair_output(monkeypatch)
    record = parse_record(
        {
            "protein_id": "demo-foldx",
            "sequence": "L",
            "mutation": "L1A",
            "position": 1,
            "wt": "L",
            "mut": "A",
            "structure_paths": {"experimental": str(DEMO_PDB), "openfold": str(DEMO_PDB), "foldx": str(DEMO_PDB)},
            "experimental_ddg": -0.1,
        }
    )
    config = PipelineConfig(output_root=tmp_path / "mixed_esm2", methods=("foldx", "rosetta", "esm2"), structure_sources=("experimental",), seed=17)
    run_pipeline(
        [record],
        [FoldXAdapter(number_of_runs=2), RosettaDDGAdapter(number_of_runs=1, top_k=1), ESM2Adapter()],
        config,
    )
    rows = list(csv.DictReader((config.output_root / "outputs" / "results.csv").open(encoding="utf-8")))
    methods = {row["method"] for row in rows}
    assert methods == {"foldx", "rosetta", "esm2"}
    esm_row = next(row for row in rows if row["method"] == "esm2")
    assert esm_row["representation"] == "sequence"
    assert esm_row["structure_source"] == "sequence_only"
    assert esm_row["unit"] == "log-probability"


def test_compute_metrics_requires_5_predictions():
    rows = [
        {"ddg": 1.0, "experimental_ddg": 1.5, "status": "ok"},
        {"ddg": -1.0, "experimental_ddg": -0.5, "status": "ok"},
        {"ddg": 0.5, "experimental_ddg": 0.6, "status": "ok"},
        {"ddg": 0.1, "experimental_ddg": 0.2, "status": "ok"},
    ]
    assert compute_metrics(rows) is None


def test_stage12_cli_demo_smoke(tmp_path, monkeypatch, capsys):
    _patch_repair_output(monkeypatch)
    exit_code = ddg_main(["--output-root", str(tmp_path / "demo"), "--demo"])
    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "FOLDX + ROSETTA + ESM2 DDG PIPELINE VALIDATED" in stdout


def test_server_run_split_csv_is_deterministic(tmp_path):
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("id,value\n0,a\n1,b\n2,c\n3,d\n4,e\n", encoding="utf-8")
    paths = split_csv(input_csv, tmp_path / "shards", 2)

    assert [path.name for path in paths] == ["shard_0000.csv", "shard_0001.csv"]
    assert list(csv.DictReader(paths[0].open(encoding="utf-8"))) == [
        {"id": "0", "value": "a"},
        {"id": "2", "value": "c"},
        {"id": "4", "value": "e"},
    ]
    assert list(csv.DictReader(paths[1].open(encoding="utf-8"))) == [
        {"id": "1", "value": "b"},
        {"id": "3", "value": "d"},
    ]


def test_server_run_incremental_resume_with_sequence_adapter(tmp_path):
    record = parse_record(
        {
            "protein_id": "server-seq",
            "sequence": "MKTLLA",
            "mutation": "T3A",
            "position": 3,
            "wt": "T",
            "mut": "A",
            "structure_paths": {"experimental": None, "openfold": None, "foldx": None},
            "experimental_ddg": 0.5,
        }
    )
    config = ServerRunConfig(
        output_root=tmp_path / "server",
        methods=("sequence_test",),
        structure_sources=("experimental",),
        shard_index=0,
        shard_count=1,
        resume=True,
    )

    results_path = run_ddg_shard([record], [SequenceCountingAdapter()], config)
    rows = list(csv.DictReader(results_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["method"] == "sequence_test"
    assert rows[0]["structure_source"] == "sequence_only"
    assert rows[0]["status"] == "ok"

    row_files = list((config.output_root / "outputs" / "rows").glob("*.json"))
    assert len(row_files) == 1
    mtime = row_files[0].stat().st_mtime_ns
    run_ddg_shard([record], [SequenceCountingAdapter()], config)
    assert row_files[0].stat().st_mtime_ns == mtime


def test_server_run_merge_structure_results(tmp_path):
    root_a = tmp_path / "structure_a"
    root_b = tmp_path / "structure_b"
    root_a.mkdir(parents=True)
    root_b.mkdir(parents=True)
    (root_a / "results.csv").write_text(
        "case_id,openfold3_status,foldx_status,openfold3_path,foldx_path\ncase_b,ok,,/tmp/b.cif,\n",
        encoding="utf-8",
    )
    (root_b / "results.csv").write_text(
        "case_id,openfold3_status,foldx_status,openfold3_path,foldx_path\ncase_a,ok,,/tmp/a.cif,\n",
        encoding="utf-8",
    )

    output_csv = merge_structure_results([root_a, root_b], tmp_path / "merged.csv")
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert [row["case_id"] for row in rows] == ["case_a", "case_b"]


def test_server_structure_dataset_requires_existing_pdb_path(tmp_path):
    input_csv = tmp_path / "processed.csv"
    input_csv.write_text(
        "protein_id,pdb_id,chain,position,wt_residue,mut_residue,experimental_ddg,mutation_id,pdb_residue_id,pdb_path,pdb_sequence\n"
        f"p1,1ABC,A,1,L,A,0.1,L1A,10,{tmp_path / 'missing.pdb'},L\n",
        encoding="utf-8",
    )

    rows = build_structure_dataset(input_csv, tmp_path / "structure.csv", tmp_path / "rejected.csv")

    assert rows == []
    rejected = list(csv.DictReader((tmp_path / "rejected.csv").open(encoding="utf-8")))
    assert "FileNotFoundError" in rejected[0]["reject_reason"]


def test_server_canonical_dataset_does_not_use_foldx_mutant_as_structure_source(tmp_path):
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text("MODEL\nEND\n", encoding="utf-8")
    structure_csv = tmp_path / "structure.csv"
    structure_csv.write_text(
        "protein_id,pdb_id,chain,position,wt_residue,mut_residue,experimental_ddg,mutation_id,pdb_residue_id,pdb_path,sequence\n"
        f"p1,1ABC,A,1,L,A,0.1,L1A,1,{pdb_path},L\n",
        encoding="utf-8",
    )
    openfold_path = tmp_path / "model.cif"
    foldx_path = tmp_path / "mutant.pdb"
    openfold_path.write_text("data_model\n#\n", encoding="utf-8")
    foldx_path.write_text("MODEL\nEND\n", encoding="utf-8")
    structure_results = tmp_path / "structure_results.csv"
    structure_results.write_text(
        "case_id,openfold3_status,foldx_status,openfold3_path,foldx_path\n"
        f"p1__L1A,ok,ok,{openfold_path},{foldx_path}\n",
        encoding="utf-8",
    )

    records = build_canonical_dataset(
        structure_csv,
        tmp_path / "canonical.json",
        tmp_path / "rejected.csv",
        openfold_results_csv=structure_results,
    )

    assert records[0].structure_paths.openfold == str(openfold_path)
    assert records[0].structure_paths.foldx is None


def test_server_run_rejects_foldx_structure_source():
    with pytest.raises(ValueError, match="foldx outputs are mutant structures"):
        _parse_structure_sources("experimental,foldx")


def test_server_run_openfold_shard_batches_queries_once(tmp_path, monkeypatch):
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text("MODEL\nEND\n", encoding="utf-8")
    structure_csv = tmp_path / "structure.csv"
    structure_csv.write_text(
        "protein_id,pdb_id,chain,position,wt_residue,mut_residue,experimental_ddg,mutation_id,pdb_residue_id,pdb_path,sequence\n"
        f"p1,1ABC,A,1,L,A,0.1,L1A,1,{pdb_path},L\n"
        f"p2,2ABC,A,1,M,V,0.2,M1V,1,{pdb_path},M\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(command, cwd, capture_output, text, check):
        calls.append(command)
        query_json = Path(command[command.index("--query-json") + 1])
        payload = json.loads(query_json.read_text(encoding="utf-8"))
        assert sorted(payload["queries"]) == ["p1__L1A", "p2__M1V"]
        output_dir = Path(command[command.index("--output-dir") + 1])
        rows = []
        for case_id, best_sample in (("p1__L1A", 2), ("p2__M1V", 1)):
            nested = output_dir / case_id / "seed_42"
            nested.mkdir(parents=True, exist_ok=True)
            for sample_index, score in ((1, 0.1), (2, 0.9)):
                prefix = nested / f"{case_id}_seed_42_sample_{sample_index}"
                Path(f"{prefix}_model.cif").write_text(f"{case_id}-sample-{sample_index}", encoding="utf-8")
                confidence_path = Path(f"{prefix}_confidences_aggregated.json")
                confidence_path.write_text(json.dumps({"sample_ranking_score": score}), encoding="utf-8")
                if sample_index == best_sample:
                    rows.append(
                        json.dumps(
                            {
                                "query_id": case_id,
                                "sample_index": sample_index,
                                "sample_ranking_score": score,
                                "aggregated_confidence_path": str(confidence_path),
                            }
                        )
                    )
        (output_dir / "summary.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("openfold3.experiment_pipeline.ddg_pipeline.server_run.subprocess.run", fake_run)

    results_path = run_openfold_shard(structure_csv, tmp_path / "openfold_shard", openfold_python="python-test")

    assert len(calls) == 1
    rows = list(csv.DictReader(results_path.open(encoding="utf-8")))
    assert [row["openfold3_status"] for row in rows] == ["success", "success"]
    for row in rows:
        assert Path(row["openfold3_path"]).exists()
