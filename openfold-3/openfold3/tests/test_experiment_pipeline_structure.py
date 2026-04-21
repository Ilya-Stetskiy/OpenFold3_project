import argparse
import csv
import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from openfold3.experiment_pipeline.structure import FoldXBackend, OpenFold3Backend
from openfold3.experiment_pipeline.structure import cli as structure_cli
from openfold3.experiment_pipeline.structure.cli import (
    build_config_dict,
    build_file_fingerprint,
    hash_config,
    load_case_sequences,
    load_cases,
    summarize_manifests,
)
from openfold3.experiment_pipeline.structure.cache import is_cached, validate_backend_artifacts
from openfold3.experiment_pipeline.structure.manifest import load_manifest
from openfold3.experiment_pipeline.structure.models import BackendResult, CaseManifest, MutationCase
from openfold3.experiment_pipeline.structure.runner import BackendContractError, StructureRunner
from openfold3.experiment_pipeline.structure.sequence import apply_mutation
from openfold3.tests.test_ddg_benchmark_harness import _write_minimal_pdb


TEST_CSV = Path(__file__).parent / "test_data" / "experiment_pipeline" / "structure" / "test.csv"


def _structure_cli_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "dataset": TEST_CSV,
        "output_dir": tmp_path / "run",
        "dry_run": False,
        "backend": "both",
        "foldx_binary": tmp_path / "foldx",
        "openfold_python": "python3",
        "openfold_runner_yaml": None,
        "openfold_inference_ckpt_path": None,
        "strict_config_hash": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@dataclass
class CountingBackend:
    backend_name: str = "foldx"
    calls: int = 0

    def run(self, case: MutationCase, case_dir: Path, config_hash: str) -> BackendResult:
        self.calls += 1
        output_dir = case_dir / self.backend_name
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / "mutant.pdb"
        artifact_path.write_text("MODEL\nEND\n", encoding="utf-8")
        return BackendResult(
            backend_name=self.backend_name,
            status="ok",
            output_dir=output_dir,
            artifact_paths=(artifact_path,),
            message=f"run-{self.calls}",
        )


@dataclass
class StableBackend:
    backend_name: str

    def run(self, case: MutationCase, case_dir: Path, config_hash: str, mutated_sequence: str) -> BackendResult:
        output_dir = case_dir / self.backend_name
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_name = "model.cif" if self.backend_name == "openfold3" else "mutant.pdb"
        artifact_path = output_dir / artifact_name
        artifact_path.write_text(mutated_sequence, encoding="utf-8")
        return BackendResult(
            backend_name=self.backend_name,
            status="ok",
            output_dir=output_dir,
            artifact_paths=(artifact_path,),
            message="ok",
        )


@dataclass
class FlakyBackend:
    backend_name: str
    failing_case_ids: tuple[str, ...]

    def run(self, case: MutationCase, case_dir: Path, config_hash: str, mutated_sequence: str) -> BackendResult:
        output_dir = case_dir / self.backend_name
        output_dir.mkdir(parents=True, exist_ok=True)
        if case.case_id in self.failing_case_ids:
            raise RuntimeError(f"planned failure for {case.case_id}")
        artifact_name = "model.cif" if self.backend_name == "openfold3" else "mutant.pdb"
        artifact_path = output_dir / artifact_name
        artifact_path.write_text(mutated_sequence, encoding="utf-8")
        return BackendResult(
            backend_name=self.backend_name,
            status="ok",
            output_dir=output_dir,
            artifact_paths=(artifact_path,),
            message="ok",
        )


@dataclass
class RecordingThreeArgBackend:
    backend_name: str = "foldx"
    seen_config_hash: str | None = None

    def run(self, case: MutationCase, case_dir: Path, config_hash: str) -> BackendResult:
        self.seen_config_hash = config_hash
        output_dir = case_dir / self.backend_name
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / "mutant.pdb"
        artifact_path.write_text(case.case_id, encoding="utf-8")
        return BackendResult(
            backend_name=self.backend_name,
            status="ok",
            output_dir=output_dir,
            artifact_paths=(artifact_path,),
            message="ok",
        )


@dataclass
class RecordingFourArgBackend:
    backend_name: str = "openfold3"
    seen_sequence: str | None = None

    def run(self, case: MutationCase, case_dir: Path, config_hash: str, mutated_sequence: str) -> BackendResult:
        self.seen_sequence = mutated_sequence
        output_dir = case_dir / self.backend_name
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / "model.cif"
        artifact_path.write_text(mutated_sequence, encoding="utf-8")
        return BackendResult(
            backend_name=self.backend_name,
            status="ok",
            output_dir=output_dir,
            artifact_paths=(artifact_path,),
            message="ok",
        )


@dataclass
class InternalTypeErrorBackend:
    backend_name: str = "openfold3"

    def run(self, case: MutationCase, case_dir: Path, config_hash: str, mutated_sequence: str) -> BackendResult:
        raise TypeError("internal backend type error")


class WrongSignatureBackend:
    backend_name: str = "openfold3"

    def run(self, case: MutationCase, case_dir: Path) -> BackendResult:
        raise AssertionError("should not be called")


def _fixed_mtime_ns() -> int:
    return 1_700_000_000_000_000_000


def _write_with_fixed_mtime(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    os.utime(path, ns=(_fixed_mtime_ns(), _fixed_mtime_ns()))


def _manifest_case(case: MutationCase, tmp_path: Path, statuses: tuple[str, ...]) -> CaseManifest:
    backend_names = ("openfold3", "foldx")
    backend_results = tuple(
        BackendResult(
            backend_name=backend_name,
            status=status,
            output_dir=tmp_path / case.case_id / backend_name,
            artifact_paths=(),
            message=status,
        )
        for backend_name, status in zip(backend_names, statuses, strict=True)
    )
    case_dir = tmp_path / case.case_id
    return CaseManifest(
        case=case,
        case_dir=case_dir,
        input_dir=case_dir / "input",
        case_json_path=case_dir / "input" / "case.json",
        mutant_fasta_path=case_dir / "input" / "mutant.fasta",
        wild_type_sequence="AAAA",
        mutated_sequence="AVAA",
        backend_results=backend_results,
    )


def test_apply_mutation_replaces_expected_residue():
    mutated = apply_mutation("MAG", 2, "A", "V")
    assert mutated == "MVG"


def test_mutation_case_from_row_loads_path_and_fields():
    cases = load_cases(TEST_CSV)

    assert len(cases) == 3
    first_case = cases[0]
    assert first_case.case_id == "protein-1__A10V"
    assert first_case.chain == "A"
    assert first_case.pdb_residue_id == "10A"
    assert first_case.pdb_path.is_absolute()


def test_build_file_fingerprint_for_missing_file(tmp_path):
    missing = tmp_path / "missing.yaml"

    fingerprint = build_file_fingerprint(missing, hash_contents=True)

    assert fingerprint == {
        "path": str(missing),
        "exists": False,
        "size": None,
        "mtime_ns": None,
        "sha256": None,
    }


def test_hash_config_is_stable_for_identical_parameters(tmp_path):
    yaml_path = tmp_path / "runner.yaml"
    _write_with_fixed_mtime(yaml_path, "runner: default\n")
    args_a = _structure_cli_args(tmp_path, openfold_runner_yaml=yaml_path)
    args_b = _structure_cli_args(tmp_path, openfold_runner_yaml=yaml_path)

    assert hash_config(build_config_dict(args_a)) == hash_config(build_config_dict(args_b))


def test_hash_config_changes_when_yaml_content_changes(tmp_path):
    yaml_path = tmp_path / "runner.yaml"
    _write_with_fixed_mtime(yaml_path, "runner: baseline\n")
    args = _structure_cli_args(tmp_path, openfold_runner_yaml=yaml_path)
    first_hash = hash_config(build_config_dict(args))

    _write_with_fixed_mtime(yaml_path, "runner: updated!\n")
    second_hash = hash_config(build_config_dict(args))

    assert first_hash != second_hash


def test_hash_config_changes_when_foldx_binary_file_changes(tmp_path):
    foldx_binary = tmp_path / "foldx"
    _write_with_fixed_mtime(foldx_binary, "#!/bin/sh\necho first\n")
    foldx_binary.chmod(0o755)
    args = _structure_cli_args(tmp_path, foldx_binary=foldx_binary)
    first_hash = hash_config(build_config_dict(args))

    _write_with_fixed_mtime(foldx_binary, "#!/bin/sh\necho second\n")
    foldx_binary.chmod(0o755)
    second_hash = hash_config(build_config_dict(args))

    assert first_hash != second_hash


def test_hash_config_resolves_path_based_foldx_binary(tmp_path, monkeypatch):
    real_binary = tmp_path / "bin" / "foldx-real"
    real_binary.parent.mkdir(parents=True, exist_ok=True)
    _write_with_fixed_mtime(real_binary, "#!/bin/sh\necho first\n")
    real_binary.chmod(real_binary.stat().st_mode | stat.S_IEXEC)

    monkeypatch.setattr(shutil, "which", lambda value: str(real_binary) if value == "foldx" else None)

    args = _structure_cli_args(tmp_path, foldx_binary="foldx")
    first_hash = hash_config(build_config_dict(args))

    _write_with_fixed_mtime(real_binary, "#!/bin/sh\necho second\n")
    real_binary.chmod(real_binary.stat().st_mode | stat.S_IEXEC)
    second_hash = hash_config(build_config_dict(args))

    assert first_hash != second_hash


def test_checkpoint_hash_without_strict_depends_on_metadata(tmp_path):
    ckpt_path = tmp_path / "checkpoint.bin"
    _write_with_fixed_mtime(ckpt_path, "AAAA")
    args = _structure_cli_args(tmp_path, openfold_inference_ckpt_path=ckpt_path, strict_config_hash=False)
    first_hash = hash_config(build_config_dict(args))

    _write_with_fixed_mtime(ckpt_path, "BBBB")
    second_hash = hash_config(build_config_dict(args))

    assert first_hash == second_hash


def test_checkpoint_hash_with_strict_depends_on_contents(tmp_path):
    ckpt_path = tmp_path / "checkpoint.bin"
    _write_with_fixed_mtime(ckpt_path, "AAAA")
    args = _structure_cli_args(tmp_path, openfold_inference_ckpt_path=ckpt_path, strict_config_hash=True)
    first_hash = hash_config(build_config_dict(args))

    _write_with_fixed_mtime(ckpt_path, "BBBB")
    second_hash = hash_config(build_config_dict(args))

    assert first_hash != second_hash


def test_is_cached_true_when_structure_path_exists(tmp_path):
    artifact_path = tmp_path / "model.cif"
    artifact_path.write_text("data", encoding="utf-8")
    entry = {
        "status": "success",
        "config_hash": "cfg",
        "structure_path": str(artifact_path),
    }

    assert validate_backend_artifacts(entry) is True
    assert is_cached({"openfold3": entry}, "openfold3", "cfg") is True


def test_is_cached_false_when_structure_path_missing(tmp_path, capsys):
    missing_path = tmp_path / "missing.cif"
    entry = {
        "status": "success",
        "config_hash": "cfg",
        "structure_path": str(missing_path),
    }

    assert validate_backend_artifacts(entry) is False
    assert is_cached({"openfold3": entry}, "openfold3", "cfg") is False
    assert "[CACHE INVALID] missing artifacts for backend=openfold3" in capsys.readouterr().out


def test_is_cached_false_when_config_hash_differs(tmp_path):
    artifact_path = tmp_path / "model.cif"
    artifact_path.write_text("data", encoding="utf-8")
    entry = {
        "status": "success",
        "config_hash": "cfg-a",
        "structure_path": str(artifact_path),
    }

    assert is_cached({"openfold3": entry}, "openfold3", "cfg-b") is False


def test_validate_backend_artifacts_false_for_corrupted_path():
    entry = {
        "status": "success",
        "config_hash": "cfg",
        "structure_path": "",
    }

    assert validate_backend_artifacts(entry) is False


def test_structure_runner_dry_run_writes_inputs(tmp_path):
    cases = load_cases(TEST_CSV)
    sequences_by_case_id = load_case_sequences(TEST_CSV)
    runner = StructureRunner(output_root=tmp_path, dry_run=True)

    manifests = runner.run_all(cases, sequences_by_case_id)

    assert len(manifests) == 3
    first_manifest = manifests[0]
    assert first_manifest.input_dir.exists()
    assert first_manifest.case_json_path.exists()
    assert first_manifest.mutant_fasta_path.exists()
    assert first_manifest.mutated_sequence == "CCCCCCCCCVQQ"
    assert first_manifest.mutant_fasta_path.read_text(encoding="utf-8") == (
        ">protein-1__A10V\nCCCCCCCCCVQQ\n"
    )

    payload = json.loads(first_manifest.case_json_path.read_text(encoding="utf-8"))
    assert payload["case"]["case_id"] == "protein-1__A10V"
    assert payload["mutated_sequence"] == "CCCCCCCCCVQQ"
    assert Path(payload["input_dir"]).exists()


def test_foldx_backend_builds_command_and_returns_success(tmp_path, monkeypatch):
    structure_path = tmp_path / "input.pdb"
    _write_minimal_pdb(structure_path)
    fake_binary = tmp_path / "foldx"
    fake_binary.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_binary.chmod(0o755)

    case = MutationCase(
        protein_id="protein-1",
        pdb_id="1ABC",
        chain="A",
        position=1,
        wt_residue="L",
        mut_residue="A",
        experimental_ddg=-1.5,
        mutation_id="L1A",
        pdb_residue_id="1",
        pdb_path=structure_path,
    )

    captured: dict[str, object] = {}

    def fake_run(command, cwd, capture_output, text, check):
        captured["command"] = command
        captured["cwd"] = cwd
        assert capture_output is True
        assert text is True
        assert check is False

        output_dir = Path(cwd) / "output"
        output_prefix = command[command.index("--output-file") + 1]
        pdb_name = f"{output_prefix}_1.pdb"
        (output_dir / pdb_name).write_text("MODEL\nEND\n", encoding="utf-8")
        (output_dir / f"PdbList_{output_prefix}_input.fxout").write_text(
            pdb_name + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    backend = FoldXBackend(binary_path=fake_binary)
    result = backend.run(case, tmp_path / "case-root", "cfg123")

    assert result.status == "ok"
    assert result.backend_name == "foldx"
    assert result.artifact_paths[0].name == "mutant.pdb"
    assert result.artifact_paths[0].exists()
    assert captured["command"] == [
        str(fake_binary.resolve()),
        "--command",
        "BuildModel",
        "--pdb",
        "input.pdb",
        "--mutant-file",
        "individual_list.txt",
        "--output-dir",
        "output",
        "--output-file",
        "protein-1__L1A_cfg123",
        "--numberOfRuns",
        "1",
        "--screen",
        "false",
    ]
    individual_list = (tmp_path / "case-root" / "foldx" / "individual_list.txt").read_text(encoding="utf-8")
    assert individual_list == "LA1A;\n"


def test_foldx_backend_cleans_output_dir_and_does_not_use_stale_pdb(tmp_path, monkeypatch, capsys):
    structure_path = tmp_path / "input.pdb"
    _write_minimal_pdb(structure_path)
    fake_binary = tmp_path / "foldx"
    fake_binary.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_binary.chmod(0o755)

    case = MutationCase(
        protein_id="protein-1",
        pdb_id="1ABC",
        chain="A",
        position=1,
        wt_residue="L",
        mut_residue="A",
        experimental_ddg=-1.5,
        mutation_id="L1A",
        pdb_residue_id="1",
        pdb_path=structure_path,
    )

    case_root = tmp_path / "case-root"
    stale_output_dir = case_root / "foldx" / "output"
    stale_output_dir.mkdir(parents=True, exist_ok=True)
    (stale_output_dir / "old.pdb").write_text("STALE", encoding="utf-8")

    def fake_run(command, cwd, capture_output, text, check):
        output_dir = Path(cwd) / "output"
        assert not (output_dir / "old.pdb").exists()
        output_prefix = command[command.index("--output-file") + 1]
        pdb_name = f"{output_prefix}_1.pdb"
        (output_dir / pdb_name).write_text("FRESH", encoding="utf-8")
        (output_dir / f"PdbList_{output_prefix}_input.fxout").write_text(
            pdb_name + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    backend = FoldXBackend(binary_path=fake_binary)
    result = backend.run(case, case_root, "cfg123")

    assert result.status == "ok"
    assert result.artifact_paths[0].read_text(encoding="utf-8") == "FRESH"
    assert not (stale_output_dir / "old.pdb").exists()
    assert "[FOLDX] cleaned output dir" in capsys.readouterr().out


def _openfold_test_case(tmp_path: Path) -> MutationCase:
    case = MutationCase(
        protein_id="protein-1",
        pdb_id="1ABC",
        chain="A",
        position=1,
        wt_residue="L",
        mut_residue="A",
        experimental_ddg=-1.5,
        mutation_id="L1A",
        pdb_residue_id="1",
        pdb_path=(tmp_path / "input.pdb").resolve(),
    )
    case.pdb_path.write_text("MODEL\nEND\n", encoding="utf-8")
    return case


def test_openfold_backend_builds_command_and_returns_success(tmp_path, monkeypatch, capsys):
    case = _openfold_test_case(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(command, cwd, capture_output, text, check):
        captured["command"] = command
        captured["cwd"] = cwd
        assert capture_output is True
        assert text is True
        assert check is False
        output_dir = Path(command[command.index("--output-dir") + 1])
        nested = output_dir / "prediction_001"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "sample.cif").write_text("data_test\n#\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    backend = OpenFold3Backend(python_executable="python-test")
    result = backend.run(case, tmp_path / "case-root", "cfg-of3", "A")

    assert result.status == "ok"
    assert result.backend_name == "openfold3"
    assert result.artifact_paths[0].name == "model.cif"
    assert result.artifact_paths[0].exists()
    assert captured["command"] == [
        "python-test",
        "-m",
        "openfold3.run_openfold",
        "predict",
        "--query-json",
        str((tmp_path / "case-root" / "openfold3" / "input" / "query.json").resolve()),
        "--output-dir",
        str((tmp_path / "case-root" / "openfold3" / "output").resolve()),
    ]
    payload = json.loads((tmp_path / "case-root" / "openfold3" / "input" / "query.json").read_text(encoding="utf-8"))
    assert payload["queries"]["protein-1__L1A"]["chains"][0]["sequence"] == "A"
    assert "[OF3] Found 1 CIF candidates" in capsys.readouterr().out


def test_openfold_backend_prefers_final_cif_over_intermediate(tmp_path, monkeypatch, capsys):
    case = _openfold_test_case(tmp_path)

    def fake_run(command, cwd, capture_output, text, check):
        output_dir = Path(command[command.index("--output-dir") + 1])
        nested = output_dir / "prediction_001"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "sample_intermediate.cif").write_text("intermediate", encoding="utf-8")
        (nested / "sample_final.cif").write_text("final", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    backend = OpenFold3Backend(python_executable="python-test")
    result = backend.run(case, tmp_path / "case-root", "cfg-of3", "A")

    assert result.status == "ok"
    assert result.artifact_paths[0].read_text(encoding="utf-8") == "final"
    assert "[OF3] Found 2 CIF candidates" in capsys.readouterr().out


def test_openfold_backend_raises_on_ambiguous_cif_outputs(tmp_path, monkeypatch, capsys):
    case = _openfold_test_case(tmp_path)

    def fake_run(command, cwd, capture_output, text, check):
        output_dir = Path(command[command.index("--output-dir") + 1])
        nested = output_dir / "prediction_001"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "candidate_a.cif").write_text("a", encoding="utf-8")
        (nested / "candidate_b.cif").write_text("b", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    backend = OpenFold3Backend(python_executable="python-test")
    with pytest.raises(RuntimeError, match="ambiguous CIF outputs"):
        backend.run(case, tmp_path / "case-root", "cfg-of3", "A")
    assert "[OF3] Found 2 CIF candidates" in capsys.readouterr().out


def test_runner_dispatches_three_argument_backend_signature(tmp_path):
    cases = load_cases(TEST_CSV)
    sequences_by_case_id = load_case_sequences(TEST_CSV)
    backend = RecordingThreeArgBackend()
    runner = StructureRunner(
        output_root=tmp_path,
        dry_run=False,
        config_hash="dispatch-3",
        backends=(backend,),
    )

    manifest = runner.run_case(cases[0], sequences_by_case_id[cases[0].case_id])

    assert manifest.backend_results[0].status == "ok"
    assert backend.seen_config_hash == "dispatch-3"


def test_runner_dispatches_four_argument_backend_signature(tmp_path):
    cases = load_cases(TEST_CSV)
    sequences_by_case_id = load_case_sequences(TEST_CSV)
    backend = RecordingFourArgBackend()
    runner = StructureRunner(
        output_root=tmp_path,
        dry_run=False,
        config_hash="dispatch-4",
        backends=(backend,),
    )

    manifest = runner.run_case(cases[0], sequences_by_case_id[cases[0].case_id])

    assert manifest.backend_results[0].status == "ok"
    assert backend.seen_sequence == "CCCCCCCCCVQQ"


def test_runner_raises_backend_contract_error_for_unexpected_signature(tmp_path, capsys):
    cases = load_cases(TEST_CSV)
    sequences_by_case_id = load_case_sequences(TEST_CSV)
    runner = StructureRunner(
        output_root=tmp_path,
        dry_run=False,
        config_hash="dispatch-contract-error",
        backends=(WrongSignatureBackend(),),
    )

    with pytest.raises(BackendContractError, match="Unsupported backend.run signature"):
        runner.run_case(cases[0], sequences_by_case_id[cases[0].case_id])
    assert "[CONTRACT ERROR] backend=openfold3" in capsys.readouterr().out


def test_runner_runtime_error_inside_backend_becomes_failed_status(tmp_path):
    cases = load_cases(TEST_CSV)
    sequences_by_case_id = load_case_sequences(TEST_CSV)
    runner = StructureRunner(
        output_root=tmp_path,
        dry_run=False,
        config_hash="dispatch-type-error",
        backends=(InternalTypeErrorBackend(),),
    )

    manifest = runner.run_case(cases[0], sequences_by_case_id[cases[0].case_id])

    assert manifest.backend_results[0].status == "failed"
    assert "TypeError: internal backend type error" in manifest.backend_results[0].message
    assert "missing 1 required positional argument" not in manifest.backend_results[0].message


def test_summarize_manifests_counts_case_and_backend_statuses(tmp_path):
    cases = load_cases(TEST_CSV)
    manifests = [
        _manifest_case(cases[0], tmp_path, ("ok", "ok")),
        _manifest_case(cases[1], tmp_path, ("ok", "failed")),
        _manifest_case(cases[2], tmp_path, ("cached", "cached")),
    ]

    summary = summarize_manifests(manifests)

    assert summary["case"] == {
        "total_cases": 3,
        "cases_all_success": 2,
        "cases_any_failed": 1,
        "cases_with_unavailable": 0,
        "cases_fully_cached": 1,
        "cases_partially_completed": 0,
    }
    assert summary["backend"] == {
        "total_backend_runs": 6,
        "backend_success": 3,
        "backend_failed": 1,
        "backend_cached": 2,
        "backend_unavailable": 0,
    }


def test_summarize_manifests_fully_cached_rerun_is_counted(tmp_path):
    cases = load_cases(TEST_CSV)
    manifests = [
        _manifest_case(cases[0], tmp_path, ("cached", "cached")),
        _manifest_case(cases[1], tmp_path, ("cached", "cached")),
        _manifest_case(cases[2], tmp_path, ("cached", "cached")),
    ]

    summary = summarize_manifests(manifests)

    assert summary["case"]["cases_fully_cached"] == 3
    assert summary["case"]["cases_all_success"] == 3
    assert summary["backend"]["backend_cached"] == 6


def test_summarize_manifests_counts_unavailable_backend(tmp_path):
    cases = load_cases(TEST_CSV)
    manifests = [
        _manifest_case(cases[0], tmp_path, ("unavailable", "ok")),
    ]

    summary = summarize_manifests(manifests)

    assert summary["case"]["cases_with_unavailable"] == 1
    assert summary["case"]["cases_any_failed"] == 0
    assert summary["backend"]["backend_unavailable"] == 1


def test_summarize_manifests_prioritizes_unavailable_over_partial(tmp_path):
    cases = load_cases(TEST_CSV)
    manifests = [
        _manifest_case(cases[0], tmp_path, ("ok", "unavailable")),
    ]

    summary = summarize_manifests(manifests)

    assert summary["case"]["cases_with_unavailable"] == 1
    assert summary["case"]["cases_partially_completed"] == 0


def test_summarize_manifests_mixed_ok_and_cached_stays_all_success(tmp_path):
    cases = load_cases(TEST_CSV)
    manifests = [
        _manifest_case(cases[0], tmp_path, ("ok", "cached")),
    ]

    summary = summarize_manifests(manifests)

    assert summary["case"]["cases_all_success"] == 1
    assert summary["case"]["cases_partially_completed"] == 0


def test_summarize_manifests_prioritizes_failed_over_unavailable(tmp_path):
    cases = load_cases(TEST_CSV)
    manifests = [
        _manifest_case(cases[0], tmp_path, ("failed", "unavailable")),
    ]

    summary = summarize_manifests(manifests)

    assert summary["case"]["cases_any_failed"] == 1
    assert summary["case"]["cases_with_unavailable"] == 0
    assert summary["backend"]["backend_failed"] == 1
    assert summary["backend"]["backend_unavailable"] == 1


def test_runner_uses_cache_and_manifest(tmp_path, capsys):
    cases = load_cases(TEST_CSV)
    sequences_by_case_id = load_case_sequences(TEST_CSV)
    backend = CountingBackend()
    runner = StructureRunner(
        output_root=tmp_path,
        dry_run=False,
        config_hash="cfg-cache",
        backends=(backend,),
    )

    first_manifest = runner.run_case(cases[0], sequences_by_case_id[cases[0].case_id])
    assert backend.calls == 1
    assert first_manifest.backend_results[0].status == "ok"

    manifest_path = tmp_path / "cases" / cases[0].case_id / "manifest.json"
    assert manifest_path.exists()
    payload = load_manifest(manifest_path)
    assert payload["openfold3"] == {}
    assert payload["foldx"]["status"] == "success"
    assert payload["foldx"]["config_hash"] == "cfg-cache"
    assert payload["foldx"]["structure_path"].endswith("mutant.pdb")

    second_manifest = runner.run_case(cases[0], sequences_by_case_id[cases[0].case_id])
    assert backend.calls == 1
    assert second_manifest.backend_results[0].status == "cached"
    captured = capsys.readouterr()
    assert "[SKIP] protein-1__A10V foldx cached" in captured.out


def test_runner_integrates_multiple_backends_and_isolates_failures(tmp_path):
    base_cases = load_cases(TEST_CSV)
    sequences_by_case_id = load_case_sequences(TEST_CSV)
    cases = [
        replace(base_cases[index % len(base_cases)], protein_id=f"protein-{index + 1}")
        for index in range(5)
    ]
    expanded_sequences = {
        case.case_id: sequences_by_case_id[base_cases[index % len(base_cases)].case_id]
        for index, case in enumerate(cases)
    }
    failing_case_ids = (cases[1].case_id, cases[3].case_id)
    runner = StructureRunner(
        output_root=tmp_path,
        dry_run=False,
        config_hash="multi-backend",
        backends=(
            StableBackend("openfold3"),
            FlakyBackend("foldx", failing_case_ids=failing_case_ids),
        ),
    )

    manifests = runner.run_all(cases, expanded_sequences)

    assert len(manifests) == 5
    foldx_statuses = []
    for manifest in manifests:
        case_dir = tmp_path / "cases" / manifest.case.case_id
        assert (case_dir / "manifest.json").exists()
        assert (case_dir / "openfold3").exists()
        assert (case_dir / "foldx").exists()

        by_backend = {result.backend_name: result for result in manifest.backend_results}
        assert by_backend["openfold3"].status == "ok"
        foldx_statuses.append(by_backend["foldx"].status)

        payload = load_manifest(case_dir / "manifest.json")
        assert payload["openfold3"]["status"] == "success"
        assert payload["foldx"]["config_hash"] == "multi-backend"

    assert foldx_statuses.count("ok") == 3
    assert foldx_statuses.count("failed") == 2

    results_csv = tmp_path / "results.csv"
    assert results_csv.exists()
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert [row["case_id"] for row in rows] == sorted(case.case_id for case in cases)
    row_by_case = {row["case_id"]: row for row in rows}
    for manifest in manifests:
        row = row_by_case[manifest.case.case_id]
        assert row["openfold3_status"] == "success"
        assert row["openfold3_path"].endswith("model.cif")
        assert row["foldx_status"] in {"success", "failed"}


def test_structure_stage_end_to_end_recomputes_on_config_change(tmp_path, capsys, monkeypatch):
    base_cases = load_cases(TEST_CSV)
    sequences_by_case_id = load_case_sequences(TEST_CSV)
    dataset_path = tmp_path / "e2e_dataset.csv"
    rows = []
    for case in base_cases:
        rows.append(
            {
                "protein_id": case.protein_id,
                "pdb_id": case.pdb_id,
                "chain": case.chain,
                "position": case.position,
                "wt_residue": case.wt_residue,
                "mut_residue": case.mut_residue,
                "experimental_ddg": case.experimental_ddg,
                "mutation_id": case.mutation_id,
                "pdb_residue_id": case.pdb_residue_id,
                "pdb_path": str(case.pdb_path),
                "sequence": sequences_by_case_id[case.case_id],
            }
        )
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    foldx_binary_a = tmp_path / "foldx-a"
    foldx_binary_b = tmp_path / "foldx-b"
    for binary in (foldx_binary_a, foldx_binary_b):
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o755)

    call_counts = {"openfold3": 0, "foldx": 0}

    def fake_run(command, cwd, capture_output, text, check):
        assert capture_output is True
        assert text is True
        assert check is False
        if command[:3] == ["python-test", "-m", "openfold3.run_openfold"]:
            call_counts["openfold3"] += 1
            output_dir = Path(command[command.index("--output-dir") + 1])
            nested = output_dir / f"prediction_{call_counts['openfold3']:03d}"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "sample_intermediate.cif").write_text(
                f"intermediate-{call_counts['openfold3']}", encoding="utf-8"
            )
            (nested / "sample_final_model.cif").write_text(
                f"final-{call_counts['openfold3']}", encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        call_counts["foldx"] += 1
        output_dir = Path(cwd) / "output"
        assert output_dir.exists()
        assert list(output_dir.iterdir()) == []
        output_prefix = command[command.index("--output-file") + 1]
        pdb_name = f"{output_prefix}_1.pdb"
        binary_name = Path(command[0]).name
        (output_dir / pdb_name).write_text(
            f"foldx-{binary_name}-{call_counts['foldx']}", encoding="utf-8"
        )
        (output_dir / f"PdbList_{output_prefix}_input.fxout").write_text(
            pdb_name + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    run_root = tmp_path / "run"
    exit_code = structure_cli.main(
        [
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(run_root),
            "--backend",
            "both",
            "--foldx-binary",
            str(foldx_binary_a),
            "--openfold-python",
            "python-test",
        ]
    )
    assert exit_code == 0

    first_output = capsys.readouterr().out
    assert "=== CASE SUMMARY ===" in first_output
    assert "All success: 3" in first_output
    assert "Any failed: 0" in first_output
    assert "Fully cached: 0" in first_output
    assert "With unavailable: 0" in first_output
    assert "=== BACKEND SUMMARY ===" in first_output
    assert "Success: 6" in first_output
    assert "Failed: 0" in first_output
    assert "Cached: 0" in first_output
    assert "Unavailable: 0" in first_output
    assert call_counts == {"openfold3": 3, "foldx": 3}

    first_case_id = base_cases[0].case_id
    first_case_dir = run_root / "cases" / first_case_id
    first_openfold_output = (first_case_dir / "openfold3" / "output" / "model.cif").read_text(encoding="utf-8")
    first_foldx_output = (first_case_dir / "foldx" / "output" / "mutant.pdb").read_text(encoding="utf-8")
    assert first_openfold_output == "final-1"
    assert first_foldx_output == "foldx-foldx-a-1"

    exit_code = structure_cli.main(
        [
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(run_root),
            "--backend",
            "both",
            "--foldx-binary",
            str(foldx_binary_b),
            "--openfold-python",
            "python-test",
        ]
    )
    assert exit_code == 0

    second_output = capsys.readouterr().out
    assert "All success: 3" in second_output
    assert "Any failed: 0" in second_output
    assert "With unavailable: 0" in second_output
    assert "Fully cached: 0" in second_output
    assert "Success: 6" in second_output
    assert "Failed: 0" in second_output
    assert "Cached: 0" in second_output
    assert "Unavailable: 0" in second_output
    assert "[SKIP]" not in second_output
    assert call_counts == {"openfold3": 6, "foldx": 6}

    second_openfold_output = (first_case_dir / "openfold3" / "output" / "model.cif").read_text(encoding="utf-8")
    second_foldx_output = (first_case_dir / "foldx" / "output" / "mutant.pdb").read_text(encoding="utf-8")
    assert second_openfold_output == "final-4"
    assert second_foldx_output == "foldx-foldx-b-4"
    assert second_openfold_output != first_openfold_output
    assert second_foldx_output != first_foldx_output

    results_csv = run_root / "results.csv"
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        results_rows = list(csv.DictReader(handle))
    assert len(results_rows) == 3
    assert {row["openfold3_status"] for row in results_rows} == {"success"}
    assert {row["foldx_status"] for row in results_rows} == {"success"}

    print("SUCCESS=6 FAILED=0 CACHED=0")


def test_pipeline_final_smoke_cli_runs_without_manual_edits(tmp_path, capsys, monkeypatch):
    base_cases = load_cases(TEST_CSV)
    sequences_by_case_id = load_case_sequences(TEST_CSV)
    cases = [
        replace(base_cases[index % len(base_cases)], protein_id=f"smoke-{index + 1}")
        for index in range(5)
    ]
    dataset_path = tmp_path / "smoke_dataset.csv"
    rows = []
    for index, case in enumerate(cases):
        source_case = base_cases[index % len(base_cases)]
        rows.append(
            {
                "protein_id": case.protein_id,
                "pdb_id": case.pdb_id,
                "chain": case.chain,
                "position": case.position,
                "wt_residue": case.wt_residue,
                "mut_residue": case.mut_residue,
                "experimental_ddg": case.experimental_ddg,
                "mutation_id": case.mutation_id,
                "pdb_residue_id": case.pdb_residue_id,
                "pdb_path": str(case.pdb_path),
                "sequence": sequences_by_case_id[source_case.case_id],
            }
        )
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    failing_case_ids = (cases[1].case_id, cases[4].case_id)

    def fake_build_backends(_args):
        return (
            StableBackend("openfold3"),
            FlakyBackend("foldx", failing_case_ids=failing_case_ids),
        )

    monkeypatch.setattr(structure_cli, "build_backends", fake_build_backends)

    exit_code = structure_cli.main([
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(tmp_path / "run"),
        "--backend",
        "both",
    ])
    assert exit_code == 0

    first_output = capsys.readouterr().out
    assert "Processed 5 cases" in first_output
    assert "=== CASE SUMMARY ===" in first_output
    assert "Total cases: 5" in first_output
    assert "All success: 3" in first_output
    assert "Any failed: 2" in first_output
    assert "With unavailable: 0" in first_output
    assert "Fully cached: 0" in first_output
    assert "Partially completed: 0" in first_output
    assert "=== BACKEND SUMMARY ===" in first_output
    assert "Total runs: 10" in first_output
    assert "Success: 8" in first_output
    assert "Failed: 2" in first_output
    assert "Cached: 0" in first_output
    assert "Unavailable: 0" in first_output

    run_root = tmp_path / "run"
    results_csv = run_root / "results.csv"
    assert results_csv.exists()
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5

    for case in cases:
        case_dir = run_root / "cases" / case.case_id
        assert case_dir.exists()
        assert (case_dir / "manifest.json").exists()
        assert (case_dir / "openfold3").exists()
        assert (case_dir / "foldx").exists()

    exit_code = structure_cli.main([
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(tmp_path / "run"),
        "--backend",
        "both",
    ])
    assert exit_code == 0

    second_output = capsys.readouterr().out
    assert "Processed 5 cases" in second_output
    assert "Total cases: 5" in second_output
    assert "All success: 3" in second_output
    assert "Any failed: 2" in second_output
    assert "With unavailable: 0" in second_output
    assert "Fully cached: 3" in second_output
    assert "Partially completed: 0" in second_output
    assert "Success: 0" in second_output
    assert "Failed: 2" in second_output
    assert "Cached: 8" in second_output
    assert "Unavailable: 0" in second_output
    assert f"[SKIP] {cases[0].case_id} openfold3 cached" in second_output

    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rerun_rows = list(csv.DictReader(handle))
    assert len(rerun_rows) == 5



def test_structure_cli_smoke(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openfold3.experiment_pipeline.structure.cli",
            "--dataset",
            str(TEST_CSV),
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--backend",
            "both",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[CONFIG HASH]" in result.stdout
    assert "Processed 3 cases" in result.stdout
    assert "=== CASE SUMMARY ===" in result.stdout
    assert "Total cases: 3" in result.stdout
    assert "With unavailable: 0" in result.stdout
    for case_id in ["protein-1__A10V", "protein-2__G25D", "protein-3__L7P"]:
        case_dir = tmp_path / "cases" / case_id / "input"
        assert case_dir.exists()
        assert (case_dir / "case.json").exists()
        assert (case_dir / "mutant.fasta").exists()


@pytest.mark.slow
def test_foldx_backend_real_run_if_available(tmp_path):
    resolved_foldx = os.environ.get("FOLDX_BINARY") or shutil.which("foldx")
    if resolved_foldx is None:
        pytest.skip("FoldX binary is unavailable")

    structure_path = tmp_path / "input.pdb"
    _write_minimal_pdb(structure_path)
    case = MutationCase(
        protein_id="protein-1",
        pdb_id="1ABC",
        chain="A",
        position=1,
        wt_residue="L",
        mut_residue="A",
        experimental_ddg=-1.5,
        mutation_id="L1A",
        pdb_residue_id="1",
        pdb_path=structure_path,
    )
    runner = StructureRunner(
        output_root=tmp_path / "runner-out",
        dry_run=False,
        config_hash="real",
        backends=(FoldXBackend(binary_path=Path(resolved_foldx)),),
    )

    manifest = runner.run_case(case, "L")

    assert len(manifest.backend_results) == 1
    result = manifest.backend_results[0]
    assert result.backend_name == "foldx"
    assert result.status == "ok"
    assert result.artifact_paths[0].name == "mutant.pdb"
    assert result.artifact_paths[0].exists()
