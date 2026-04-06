import json
from pathlib import Path

from openfold3.benchmark.harness import HarnessReport, MethodResult
from openfold3.testbench.models import TestbenchConfig as TBConfig
from openfold3.testbench.registry import SQLiteRegistry


def _make_report(case_id: str, experimental_ddg: float | None = None) -> HarnessReport:
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
                score=-0.4,
                units="kcal/mol",
                details={"runtime_seconds": 1.2},
            ),
            MethodResult(
                method="openfold_confidence",
                status="ok",
                score=0.9,
                units="ranking_score",
                details={"avg_plddt": 90.0},
            ),
        ),
        experimental_ddg=experimental_ddg,
        notes="note",
    )


def test_registry_creates_run_and_persists_case_report(tmp_path):
    registry = SQLiteRegistry(tmp_path / "registry.sqlite")
    try:
        config = TBConfig(output_root=tmp_path, dataset_kind="benchmark").resolved()
        run_id = registry.create_run(config)
        case_pk = registry.insert_case_report(
            run_id=run_id,
            dataset_kind="benchmark",
            report=_make_report("case-1", experimental_ddg=-1.0),
            report_path=tmp_path / "report.json",
        )

        assert run_id > 0
        assert case_pk > 0
        run_row = registry.fetch_run(run_id)
        assert run_row is not None
        assert run_row["dataset_kind"] == "benchmark"
        assert list(registry.list_case_ids(run_id)) == ["case-1"]

        method_rows = registry.fetch_method_rows(run_id)
        assert len(method_rows) == 2
        assert method_rows[0]["case_id"] == "case-1"
        assert json.loads(method_rows[0]["details_json"]) != {}
        stage_rows = registry.fetch_stage_rows(run_id)
        assert len(stage_rows) == 1
        assert stage_rows[0]["case_id"] == "case-1"
        assert stage_rows[0]["stage_name"] == "method:foldx"
    finally:
        registry.close()
