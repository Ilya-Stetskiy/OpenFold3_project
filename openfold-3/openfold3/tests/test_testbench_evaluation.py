from openfold3.benchmark.harness import HarnessReport, MethodResult
from openfold3.testbench.evaluation import evaluate_reports


def _report(case_id: str, score: float, experimental_ddg: float | None) -> HarnessReport:
    return HarnessReport(
        case_id=case_id,
        structure_path=f"/tmp/{case_id}.pdb",
        confidence_path=None,
        structure_summary={
            "atom_count": 10,
            "residue_count": 2,
            "chain_ids": ["A", "B"],
            "chain_lengths": {"A": 1, "B": 1},
            "inferred_chain_groups": [["A"], ["B"]],
            "min_inter_chain_atom_distance": 3.5,
            "interface_atom_contacts_5a": 12,
            "interface_ca_contacts_8a": 1,
            "chain_pair_min_distances": {"A-B": 3.5},
        },
        results=(
            MethodResult(
                method="foldx",
                status="ok",
                score=score,
                units="kcal/mol",
                details={"runtime_seconds": 1.0},
            ),
            MethodResult(
                method="ml_stub",
                status="unavailable",
                score=None,
                units=None,
                details={"reason": "not_installed"},
            ),
        ),
        experimental_ddg=experimental_ddg,
        notes=None,
    )


def test_evaluate_reports_separates_benchmark_and_exploratory():
    summary = evaluate_reports(
        [
            _report("bench-1", -1.0, -1.5),
            _report("bench-2", -2.0, -2.5),
            _report("explore-1", -0.2, None),
        ]
    )

    assert summary.total_cases == 3
    assert summary.benchmark_cases == 2
    assert summary.exploratory_cases == 1
    foldx = next(item for item in summary.methods if item.method == "foldx")
    assert foldx.ok_count == 3
    assert foldx.failed_count == 0
    assert foldx.score_mean is not None
    assert foldx.pearson_vs_experimental is not None
    assert foldx.spearman_vs_experimental is not None
    assert foldx.kendall_vs_experimental is not None
    assert foldx.mae_vs_experimental is not None
    assert foldx.rmse_vs_experimental is not None
    assert foldx.sign_accuracy_vs_experimental == 1.0
    assert foldx.top_1_overlap_vs_experimental == 1.0
    ml_stub = next(item for item in summary.methods if item.method == "ml_stub")
    assert ml_stub.unavailable_count == 3
