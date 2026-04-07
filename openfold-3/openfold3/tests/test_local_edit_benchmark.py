from __future__ import annotations

import json
from pathlib import Path

import pytest

from openfold3.benchmark.harness import DdgBenchmarkHarness, MethodResult
from openfold3.benchmark.local_edit_benchmark import (
    CyclicMutationCase,
    ReferenceMutationCase,
    benchmark_cases_for_preset,
    run_cyclic_mutation_case,
    run_local_edit_benchmark,
    run_reference_mutation_case,
)
from openfold3.benchmark.models import MutationInput


class _MappedFoldxMethod:
    name = "foldx"

    def __init__(self, mutant_by_case_id: dict[str, Path]):
        self.mutant_by_case_id = mutant_by_case_id

    def run(self, context):
        mutant_model_path = self.mutant_by_case_id[context.case.case_id]
        return MethodResult(
            method=self.name,
            status="ok",
            score=0.0,
            units="kcal/mol",
            details={
                "reason": None,
                "prepared_from_cif": False,
                "runtime_seconds": 0.01,
                "mutant_model_path": str(mutant_model_path),
            },
        )


def _write_two_chain_complex(
    path: Path,
    *,
    chain_a_res2: str = "LEU",
    chain_b_res2: str = "ASP",
) -> None:
    path.write_text(
        "\n".join(
            [
                "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 80.00           N",
                "ATOM      2  CA  GLY A   1       1.300   0.000   0.000  1.00 80.00           C",
                "ATOM      3  C   GLY A   1       1.900   1.300   0.000  1.00 80.00           C",
                "ATOM      4  O   GLY A   1       1.200   2.300   0.000  1.00 80.00           O",
                f"ATOM      5  N   {chain_a_res2:>3} A   2       3.200   1.300   0.000  1.00 80.00           N",
                f"ATOM      6  CA  {chain_a_res2:>3} A   2       3.800   2.500   0.000  1.00 80.00           C",
                f"ATOM      7  C   {chain_a_res2:>3} A   2       5.200   2.500   0.000  1.00 80.00           C",
                f"ATOM      8  O   {chain_a_res2:>3} A   2       5.900   3.500   0.000  1.00 80.00           O",
                "ATOM      9  N   GLY B   1       3.100   4.000   0.000  1.00 80.00           N",
                "ATOM     10  CA  GLY B   1       4.000   5.100   0.000  1.00 80.00           C",
                "ATOM     11  C   GLY B   1       5.300   4.700   0.000  1.00 80.00           C",
                "ATOM     12  O   GLY B   1       6.200   5.500   0.000  1.00 80.00           O",
                f"ATOM     13  N   {chain_b_res2:>3} B   2       5.400   3.400   0.000  1.00 80.00           N",
                f"ATOM     14  CA  {chain_b_res2:>3} B   2       6.600   2.700   0.000  1.00 80.00           C",
                f"ATOM     15  C   {chain_b_res2:>3} B   2       7.700   3.600   0.000  1.00 80.00           C",
                f"ATOM     16  O   {chain_b_res2:>3} B   2       8.900   3.300   0.000  1.00 80.00           O",
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_reference_mutation_case_scores_against_reference_structure(tmp_path: Path) -> None:
    wt_path = tmp_path / "wt.pdb"
    reference_path = tmp_path / "reference.pdb"
    _write_two_chain_complex(wt_path, chain_b_res2="ASP")
    _write_two_chain_complex(reference_path, chain_b_res2="ALA")
    harness = DdgBenchmarkHarness(
        methods=[_MappedFoldxMethod({"reference_case": reference_path})]
    )

    result = run_reference_mutation_case(
        ReferenceMutationCase(
            case_id="reference_case",
            source_structure_path=wt_path,
            reference_structure_path=reference_path,
            mutation=MutationInput("B", "D", 2, "A"),
            focus_chain_ids=("A", "B"),
        ),
        work_dir=tmp_path / "run",
        harness=harness,
    )

    assert result.row["quality_status"] == "ok"
    assert result.row["predicted_sequences_match_reference"] is True
    assert result.row["predicted_vs_reference_ca_rmsd_angstrom"] == pytest.approx(0.0)
    assert result.row["predicted_vs_reference_backbone_rmsd_angstrom"] == pytest.approx(0.0)


def test_run_cyclic_mutation_case_restores_original_sequence(tmp_path: Path) -> None:
    wt_path = tmp_path / "wt.pdb"
    step_1 = tmp_path / "step_1.pdb"
    step_2 = tmp_path / "step_2.pdb"
    step_3 = tmp_path / "step_3.pdb"
    _write_two_chain_complex(wt_path, chain_b_res2="ASP")
    _write_two_chain_complex(step_1, chain_b_res2="ALA")
    _write_two_chain_complex(step_2, chain_b_res2="ASN")
    _write_two_chain_complex(step_3, chain_b_res2="ASP")
    harness = DdgBenchmarkHarness(
        methods=[
            _MappedFoldxMethod(
                {
                    "cycle_case_step_1": step_1,
                    "cycle_case_step_2": step_2,
                    "cycle_case_step_3": step_3,
                }
            )
        ]
    )

    result = run_cyclic_mutation_case(
        CyclicMutationCase(
            case_id="cycle_case",
            source_structure_path=wt_path,
            steps=(
                MutationInput("B", "D", 2, "A"),
                MutationInput("B", "A", 2, "N"),
                MutationInput("B", "N", 2, "D"),
            ),
            focus_chain_ids=("A", "B"),
        ),
        work_dir=tmp_path / "cycle_run",
        harness=harness,
    )

    assert result.row["quality_status"] == "ok"
    assert result.row["final_sequences_match_source"] is True
    assert result.row["round_trip_ca_rmsd_angstrom"] == pytest.approx(0.0)
    assert result.row["round_trip_backbone_rmsd_angstrom"] == pytest.approx(0.0)


def test_run_local_edit_benchmark_writes_summary_for_custom_stub_cases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wt_path = tmp_path / "wt.pdb"
    mutant_path = tmp_path / "mutant.pdb"
    asn_path = tmp_path / "asn.pdb"
    _write_two_chain_complex(wt_path, chain_b_res2="ASP")
    _write_two_chain_complex(mutant_path, chain_b_res2="ALA")
    _write_two_chain_complex(asn_path, chain_b_res2="ASN")
    harness = DdgBenchmarkHarness(
        methods=[
            _MappedFoldxMethod(
                {
                    "reference_case": mutant_path,
                    "cycle_case_step_1": mutant_path,
                    "cycle_case_step_2": asn_path,
                    "cycle_case_step_3": wt_path,
                }
            )
        ]
    )

    monkeypatch.setattr(
        "openfold3.benchmark.local_edit_benchmark.benchmark_cases_for_preset",
        lambda preset: (
            (
                ReferenceMutationCase(
                    case_id="reference_case",
                    source_structure_path=wt_path,
                    reference_structure_path=mutant_path,
                    mutation=MutationInput("B", "D", 2, "A"),
                    focus_chain_ids=("A", "B"),
                ),
            ),
            (
                CyclicMutationCase(
                    case_id="cycle_case",
                    source_structure_path=wt_path,
                    steps=(
                        MutationInput("B", "D", 2, "A"),
                        MutationInput("B", "A", 2, "N"),
                        MutationInput("B", "N", 2, "D"),
                    ),
                    focus_chain_ids=("A", "B"),
                ),
            ),
        ),
    )

    suite = run_local_edit_benchmark(
        output_root=tmp_path / "suite",
        preset="smoke",
        harness=harness,
    )

    payload = json.loads(suite.summary_json_path.read_text(encoding="utf-8"))
    assert payload["aggregate"]["total_cases"] == 2
    assert suite.rows_csv_path.exists()


def test_benchmark_cases_for_preset_exposes_safe_default_cases() -> None:
    smoke_reference, smoke_cycles = benchmark_cases_for_preset("smoke")
    full_reference, full_cycles = benchmark_cases_for_preset("full")

    assert smoke_reference[0].source_pdb_id == "1L63"
    assert smoke_reference[0].reference_pdb_id == "181L"
    assert len(smoke_cycles) == 1
    assert len(full_reference) >= 1
    assert len(full_cycles) >= 2
