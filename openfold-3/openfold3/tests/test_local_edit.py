from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from openfold3.benchmark.cif_utils import parse_structure_records, summarize_structure
from openfold3.benchmark.harness import DdgBenchmarkHarness, MethodResult
from openfold3.benchmark.local_edit import run_local_mutation_case, run_local_mutation_suite
from openfold3.benchmark.methods import _resolve_executable_path
from openfold3.benchmark.models import BenchmarkCase, MutationInput
from openfold3.benchmark.structure_source import extract_protein_sequence

from .test_ddg_benchmark_harness import _write_minimal_pdb


class DummyFoldxMethod:
    name = "foldx"

    def __init__(
        self,
        output_root: Path,
        *,
        status: str = "ok",
        reason: str | None = None,
    ):
        self.output_root = output_root
        self.status = status
        self.reason = reason

    def run(self, context):
        if self.status != "ok":
            return MethodResult(
                method=self.name,
                status=self.status,
                details={"reason": self.reason or self.status},
            )

        mutation = context.case.mutations[0]
        case_root = self.output_root / context.case.case_id
        case_root.mkdir(parents=True, exist_ok=True)
        mutant_model_path = case_root / "mutant_model.pdb"
        if context.case.structure_path.suffix.lower() == ".pdb":
            shutil.copy2(context.case.structure_path, mutant_model_path)
        else:
            _write_minimal_pdb(mutant_model_path)

        score_map = {"A": -1.0, "V": -0.4, "G": 0.2}
        return MethodResult(
            method=self.name,
            status="ok",
            score=score_map.get(mutation.to_residue.upper(), 0.0),
            units="kcal/mol",
            details={
                "reason": None,
                "prepared_from_cif": context.case.structure_path.suffix.lower() == ".cif",
                "runtime_seconds": 0.01,
                "mutant_model_path": str(mutant_model_path),
            },
        )


def _write_single_chain_pdb(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ATOM      1  N   LEU A   1       0.000   0.000   0.000  1.00 90.00           N",
                "ATOM      2  CA  LEU A   1       1.458   0.000   0.000  1.00 90.00           C",
                "ATOM      3  C   LEU A   1       1.958   1.420   0.000  1.00 90.00           C",
                "ATOM      4  O   LEU A   1       1.200   2.360   0.000  1.00 90.00           O",
                "ATOM      5  CB  LEU A   1       1.958  -0.780  -1.220  1.00 80.00           C",
                "ATOM      6  N   GLU A   2       3.258   1.620   0.000  1.00 88.00           N",
                "ATOM      7  CA  GLU A   2       3.858   2.960   0.000  1.00 88.00           C",
                "ATOM      8  C   GLU A   2       5.378   2.940   0.000  1.00 88.00           C",
                "ATOM      9  O   GLU A   2       6.018   3.980   0.000  1.00 88.00           O",
                "ATOM     10  CB  GLU A   2       3.318   3.760  -1.220  1.00 77.00           C",
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _matched_coords(
    left_path: Path,
    right_path: Path,
    *,
    atom_names: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    def _index(path: Path) -> dict[tuple[str, str, str], np.ndarray]:
        index: dict[tuple[str, str, str], np.ndarray] = {}
        for atom in parse_structure_records(path):
            if atom.atom_name not in atom_names:
                continue
            index[(atom.chain_id, atom.residue_id, atom.atom_name)] = np.array(
                [atom.x, atom.y, atom.z],
                dtype=float,
            )
        return index

    left_index = _index(left_path)
    right_index = _index(right_path)
    common_keys = sorted(set(left_index) & set(right_index))
    if not common_keys:
        raise AssertionError("No matched atoms were found for RMSD calculation")
    left_coords = np.stack([left_index[key] for key in common_keys], axis=0)
    right_coords = np.stack([right_index[key] for key in common_keys], axis=0)
    return left_coords, right_coords


def _kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mobile_centroid = mobile.mean(axis=0)
    reference_centroid = reference.mean(axis=0)
    mobile_centered = mobile - mobile_centroid
    reference_centered = reference - reference_centroid
    covariance = mobile_centered.T @ reference_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    return mobile_centered @ rotation + reference_centroid


def _rmsd_after_superposition(
    left_path: Path,
    right_path: Path,
    *,
    atom_names: set[str],
) -> float:
    mobile, reference = _matched_coords(left_path, right_path, atom_names=atom_names)
    aligned = _kabsch_align(mobile, reference)
    deltas = aligned - reference
    return float(np.sqrt(np.mean(np.sum(deltas * deltas, axis=1))))


def _require_foldx() -> None:
    if _resolve_executable_path("foldx", env_var_name="FOLDX_BINARY") is None:
        pytest.skip("FoldX binary is unavailable")


def test_run_local_mutation_case_returns_mutant_structure_and_report(tmp_path: Path) -> None:
    structure_path = tmp_path / "mini.pdb"
    _write_single_chain_pdb(structure_path)
    harness = DdgBenchmarkHarness(methods=[DummyFoldxMethod(tmp_path / "dummy-foldx")])

    result = run_local_mutation_case(
        mutation=MutationInput("A", "L", 1, "A"),
        structure_path=structure_path,
        work_dir=tmp_path / "run",
        case_id="single_case",
        harness=harness,
    )

    assert result.local_edit_status == "ok"
    assert result.source_kind == "structure_path"
    assert result.mutant_structure_path is not None
    assert result.mutant_structure_path.exists()
    assert result.report_path.exists()
    assert (result.report_path.parent / "local_edit_result.json").exists()
    assert result.failure_reason is None
    assert result.mutant_structure_summary is not None
    assert result.mutant_structure_summary["chain_lengths"] == {"A": 2}


def test_run_local_mutation_case_rejects_noop_mutation(tmp_path: Path) -> None:
    structure_path = tmp_path / "mini.pdb"
    _write_single_chain_pdb(structure_path)

    with pytest.raises(ValueError, match="No-op mutation"):
        run_local_mutation_case(
            mutation=MutationInput("A", "L", 1, "L"),
            structure_path=structure_path,
            work_dir=tmp_path / "run",
        )


def test_run_local_mutation_case_propagates_failure_reason(tmp_path: Path) -> None:
    structure_path = tmp_path / "mini.pdb"
    _write_single_chain_pdb(structure_path)
    harness = DdgBenchmarkHarness(
        methods=[DummyFoldxMethod(tmp_path / "dummy-foldx", status="failed", reason="boom")]
    )

    result = run_local_mutation_case(
        mutation=MutationInput("A", "L", 1, "A"),
        structure_path=structure_path,
        work_dir=tmp_path / "run",
        case_id="failure_case",
        harness=harness,
    )

    assert result.local_edit_status == "failed"
    assert result.failure_reason == "boom"
    assert result.mutant_structure_path is None


def test_run_local_mutation_suite_writes_manifest_and_evaluation(tmp_path: Path) -> None:
    structure_path = tmp_path / "mini.pdb"
    _write_single_chain_pdb(structure_path)
    harness = DdgBenchmarkHarness(methods=[DummyFoldxMethod(tmp_path / "dummy-foldx")])
    cases = [
        BenchmarkCase(
            case_id="case_a",
            structure_path=structure_path,
            mutations=(MutationInput("A", "L", 1, "A"),),
            experimental_ddg=-1.2,
        ),
        BenchmarkCase(
            case_id="case_v",
            structure_path=structure_path,
            mutations=(MutationInput("A", "L", 1, "V"),),
            experimental_ddg=-0.6,
        ),
    ]

    suite = run_local_mutation_suite(
        output_root=tmp_path / "suite",
        cases=cases,
        harness=harness,
        dataset_kind="benchmark",
    )

    assert len(suite.results) == 2
    assert suite.evaluation_summary.total_cases == 2
    assert suite.evaluation_summary.benchmark_cases == 2
    assert suite.evaluation_summary_path.exists()
    assert suite.manifest_path.exists()
    manifest = json.loads(suite.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_kind"] == "benchmark"
    assert manifest["num_cases"] == 2
    assert manifest["successful_edits"] == 2
    method_summary = next(
        method
        for method in suite.evaluation_summary.methods
        if method.method == "foldx"
    )
    assert method_summary.spearman_vs_experimental is not None
    assert method_summary.sign_accuracy_vs_experimental == 1.0


def test_run_local_mutation_suite_accepts_cases_json_with_pdb_id(tmp_path: Path) -> None:
    harness = DdgBenchmarkHarness(methods=[DummyFoldxMethod(tmp_path / "dummy-foldx")])
    cases_json = tmp_path / "cases.json"
    cases_json.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "pdb_case",
                        "pdb_id": "1KD8",
                        "mutations": [
                            {
                                "chain_id": "A",
                                "from_residue": "L",
                                "position_1based": 1,
                                "to_residue": "A",
                            }
                        ],
                        "experimental_ddg": -1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cif_path = tmp_path / "downloaded.cif"
    from .test_ddg_benchmark_harness import _write_foldx_ready_cif

    _write_foldx_ready_cif(cif_path)

    class _Session:
        def get(self, url: str, timeout: int = 60):
            del url, timeout

            class _Response:
                text = cif_path.read_text(encoding="utf-8")

                @staticmethod
                def raise_for_status() -> None:
                    return None

            return _Response()

    suite = run_local_mutation_suite(
        output_root=tmp_path / "suite",
        cases_json=cases_json,
        harness=harness,
        dataset_kind="benchmark",
        cache_dir=tmp_path / "cache",
        session=_Session(),
    )

    assert len(suite.results) == 1
    assert suite.results[0].source_kind == "pdb_id"
    assert suite.results[0].source_path.exists()
    assert suite.results[0].local_edit_status == "ok"


@pytest.mark.slow
def test_run_local_mutation_case_real_foldx_for_complex_fixture(tmp_path: Path) -> None:
    _require_foldx()
    structure_path = (
        Path(__file__).resolve().parent / "test_data" / "mmcifs" / "1kd8.cif"
    )

    result = run_local_mutation_case(
        mutation=MutationInput("A", "L", 5, "A"),
        structure_path=structure_path,
        work_dir=tmp_path / "complex_run",
        case_id="1kd8_A_L5A",
    )

    assert result.local_edit_status == "ok"
    assert result.prepared_from_cif is True
    assert result.mutant_structure_path is not None
    assert result.mutant_structure_path.exists()


@pytest.mark.slow
def test_run_local_mutation_case_real_foldx_for_monomer_fixture(tmp_path: Path) -> None:
    _require_foldx()
    structure_path = (
        Path(__file__).resolve().parent / "test_data" / "mmcifs" / "4zey.cif"
    )

    result = run_local_mutation_case(
        mutation=MutationInput("A", "L", 8, "A"),
        structure_path=structure_path,
        work_dir=tmp_path / "monomer_run",
        case_id="4zey_A_L8A",
    )

    assert result.local_edit_status == "ok"
    assert result.mutant_structure_path is not None
    assert result.mutant_structure_path.exists()


@pytest.mark.slow
def test_cyclic_mutation_round_trip_preserves_sequence_and_structure(tmp_path: Path) -> None:
    _require_foldx()
    structure_path = (
        Path(__file__).resolve().parent / "test_data" / "mmcifs" / "1kd8.cif"
    )
    original_summary = summarize_structure(structure_path)

    first = run_local_mutation_case(
        mutation=MutationInput("A", "L", 5, "A"),
        structure_path=structure_path,
        work_dir=tmp_path / "cycle",
        case_id="cycle_step_1",
    )
    second = run_local_mutation_case(
        mutation=MutationInput("A", "A", 5, "V"),
        structure_path=first.mutant_structure_path,
        work_dir=tmp_path / "cycle",
        case_id="cycle_step_2",
    )
    third = run_local_mutation_case(
        mutation=MutationInput("A", "V", 5, "L"),
        structure_path=second.mutant_structure_path,
        work_dir=tmp_path / "cycle",
        case_id="cycle_step_3",
    )

    assert first.local_edit_status == "ok"
    assert second.local_edit_status == "ok"
    assert third.local_edit_status == "ok"
    assert third.mutant_structure_path is not None
    assert extract_protein_sequence(structure_path, "A") == extract_protein_sequence(
        third.mutant_structure_path,
        "A",
    )

    final_summary = summarize_structure(third.mutant_structure_path)
    assert len(final_summary.chain_ids) == len(original_summary.chain_ids)
    assert final_summary.residue_count == original_summary.residue_count

    ca_rmsd = _rmsd_after_superposition(
        third.mutant_structure_path,
        structure_path,
        atom_names={"CA"},
    )
    backbone_rmsd = _rmsd_after_superposition(
        third.mutant_structure_path,
        structure_path,
        atom_names={"N", "CA", "C", "O"},
    )
    assert ca_rmsd <= 1.0
    assert backbone_rmsd <= 0.75

    original_contacts = original_summary.interface_ca_contacts_8a
    final_contacts = final_summary.interface_ca_contacts_8a
    if original_contacts > 0:
        drift_fraction = abs(final_contacts - original_contacts) / original_contacts
        assert drift_fraction <= 0.10
