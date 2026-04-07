import json
from pathlib import Path

import pytest

from openfold3.benchmark.harness import DdgBenchmarkHarness, MethodResult
from openfold3.benchmark.models import BenchmarkCase, MutationInput
from openfold3.testbench.models import TestbenchConfig as TBConfig
from openfold3.testbench.runner import TestbenchRunner as TBRunner, load_cases_from_json

from .test_ddg_benchmark_harness import _write_foldx_ready_cif, _write_minimal_pdb


class DummyMethod:
    name = "dummy"

    def run(self, context):
        return MethodResult(
            method=self.name,
            status="ok",
            score=float(len(context.case.mutations)),
            units="count",
            details={"runtime_seconds": 0.01},
        )


def test_load_cases_from_json_reads_case_list(tmp_path):
    structure_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(structure_path)
    cases_json = tmp_path / "cases.json"
    cases_json.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case-1",
                        "structure_path": str(structure_path),
                        "mutations": [
                            {
                                "chain_id": "A",
                                "from_residue": "L",
                                "position_1based": 1,
                                "to_residue": "A",
                            }
                        ],
                        "experimental_ddg": -1.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_cases_from_json(cases_json)

    assert len(cases) == 1
    assert cases[0].case_id == "case-1"
    assert cases[0].mutations[0].mutation_id == "A_L1A"


def test_load_cases_from_json_supports_pdb_id(tmp_path: Path):
    cases_json = tmp_path / "cases.json"
    cases_json.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case-pdb",
                        "pdb_id": "1KD8",
                        "mutations": [
                            {
                                "chain_id": "A",
                                "from_residue": "L",
                                "position_1based": 1,
                                "to_residue": "A",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cif_path = tmp_path / "1KD8.cif"
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

    cases = load_cases_from_json(
        cases_json,
        structure_cache_dir=tmp_path / "cache",
        session=_Session(),
    )

    assert len(cases) == 1
    assert cases[0].pdb_id == "1KD8"
    assert cases[0].structure_path.exists()


def test_load_cases_from_json_requires_exactly_one_structure_source(tmp_path: Path):
    cases_json = tmp_path / "cases.json"
    cases_json.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "broken",
                        "mutations": [
                            {
                                "chain_id": "A",
                                "from_residue": "L",
                                "position_1based": 1,
                                "to_residue": "A",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Provide exactly one of structure_path or pdb_id"):
        load_cases_from_json(cases_json)


def test_testbench_runner_writes_reports_registry_and_summary(tmp_path):
    structure_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(structure_path)
    cases = [
        BenchmarkCase(
            case_id="case-1",
            structure_path=structure_path,
            mutations=(MutationInput("A", "L", 1, "A"),),
            experimental_ddg=-1.0,
        ),
        BenchmarkCase(
            case_id="case-2",
            structure_path=structure_path,
            mutations=(MutationInput("A", "L", 1, "V"),),
            experimental_ddg=-0.8,
        ),
    ]
    harness = DdgBenchmarkHarness(methods=[DummyMethod()])
    runner = TBRunner(
        TBConfig(output_root=tmp_path, dataset_kind="benchmark"),
        harness=harness,
    )

    run_id, reports, summary = runner.run_cases(cases)

    assert run_id > 0
    assert len(reports) == 2
    assert summary.total_cases == 2
    assert (tmp_path / "registry.sqlite").exists()
    assert (tmp_path / "run_manifest.json").exists()
    assert (tmp_path / "evaluation_summary.json").exists()
    assert (tmp_path / "reports" / "case-1.json").exists()
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_kind"] == "benchmark"
    assert manifest["num_cases"] == 2
