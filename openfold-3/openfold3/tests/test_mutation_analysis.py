from __future__ import annotations

import json
from pathlib import Path

import pytest

import openfold3.benchmark.mutation_analysis as mutation_analysis
from openfold3.benchmark.harness import MethodResult
from openfold3.benchmark.models import MutationInput
from openfold3.tests.test_ddg_benchmark_harness import _write_minimal_cif


class _MutantAwareMethod:
    name = "rosetta_score"

    def run(self, context):
        if context.case.case_id.endswith("_WT"):
            score = -12.0
        else:
            score = -10.5
        return MethodResult(
            method=self.name,
            status="ok",
            score=score,
            units="rosetta_energy",
            details={"case_id": context.case.case_id},
        )


class _MutationRequiredMethod:
    name = "saambe_3d"

    def run(self, context):
        if not context.case.mutations:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={"reason": "mutation_spec_missing"},
            )
        return MethodResult(
            method=self.name,
            status="ok",
            score=1.25,
            units="kcal/mol",
            details={"mutation_id": context.case.mutations[0].mutation_id},
        )


@pytest.fixture
def structure_pair(tmp_path: Path) -> tuple[Path, Path]:
    mutant = tmp_path / "mutant.cif"
    wt = tmp_path / "wt.cif"
    _write_minimal_cif(mutant)
    _write_minimal_cif(wt)
    return mutant, wt


def test_run_mutation_analysis_writes_json_and_csv(monkeypatch, tmp_path: Path, structure_pair: tuple[Path, Path]) -> None:
    mutant, _wt = structure_pair
    monkeypatch.setattr(
        mutation_analysis,
        "_method_factories",
        lambda: {
            "rosetta_score": _MutantAwareMethod,
            "saambe_3d": _MutationRequiredMethod,
        },
    )

    result = mutation_analysis.run_mutation_analysis(
        structure_path=mutant,
        mutation=MutationInput("A", "L", 1, "A"),
        case_id="demo_case",
        methods=["rosetta_score", "saambe_3d"],
        output_dir=tmp_path / "out",
        write_outputs=True,
    )

    assert result.case_id == "demo_case"
    assert result.max_workers == 2
    assert result.result_json_path is not None and result.result_json_path.exists()
    assert result.result_csv_path is not None and result.result_csv_path.exists()
    payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))
    assert payload["inputs"]["methods"] == ["rosetta_score", "saambe_3d"]
    assert payload["mutant"]["results"][0]["method"] == "rosetta_score"
    assert payload["flat_results"][1]["method"] == "saambe_3d"


def test_run_mutation_analysis_mutant_vs_wt_computes_delta(monkeypatch, tmp_path: Path, structure_pair: tuple[Path, Path]) -> None:
    mutant, wt = structure_pair
    monkeypatch.setattr(
        mutation_analysis,
        "_method_factories",
        lambda: {
            "rosetta_score": _MutantAwareMethod,
            "saambe_3d": _MutationRequiredMethod,
        },
    )

    result = mutation_analysis.run_mutation_analysis(
        structure_path=mutant,
        wt_structure_path=wt,
        mutation={"chain_id": "A", "from_residue": "L", "position_1based": 1, "to_residue": "A"},
        case_id="pair_case",
        methods=["rosetta_score", "saambe_3d"],
        mode="mutant_vs_wt",
        write_outputs=False,
    )

    comparison = result.report["comparison"]["delta_vs_wt"]
    assert comparison["rosetta_score"]["delta"] == pytest.approx(1.5)
    assert "saambe_3d" not in comparison
    flat_rows = result.report["flat_results"]
    rosetta_row = next(row for row in flat_rows if row["method"] == "rosetta_score")
    assert rosetta_row["delta_vs_wt"] == pytest.approx(1.5)


def test_run_mutation_analysis_requires_wt_for_pair_mode(tmp_path: Path) -> None:
    mutant = tmp_path / "mutant.cif"
    _write_minimal_cif(mutant)

    with pytest.raises(ValueError, match="wt_structure_path is required"):
        mutation_analysis.run_mutation_analysis(
            structure_path=mutant,
            mode="mutant_vs_wt",
            write_outputs=False,
        )


def test_run_mutation_analysis_batch_reuses_single_case_api(monkeypatch, tmp_path: Path, structure_pair: tuple[Path, Path]) -> None:
    mutant, wt = structure_pair
    monkeypatch.setattr(
        mutation_analysis,
        "_method_factories",
        lambda: {"rosetta_score": _MutantAwareMethod},
    )

    results = mutation_analysis.run_mutation_analysis_batch(
        cases=[
            mutation_analysis.MutationAnalysisInput(
                structure_path=mutant,
                case_id="case_one",
                methods=("rosetta_score",),
            ),
            mutation_analysis.MutationAnalysisInput(
                structure_path=mutant,
                wt_structure_path=wt,
                case_id="case_two",
                methods=("rosetta_score",),
                mode="mutant_vs_wt",
            ),
        ],
        output_dir=tmp_path / "batch_out",
        write_outputs=False,
    )

    assert [result.case_id for result in results] == ["case_one", "case_two"]
    assert results[1].report["comparison"]["delta_vs_wt"]["rosetta_score"]["delta"] == pytest.approx(1.5)
