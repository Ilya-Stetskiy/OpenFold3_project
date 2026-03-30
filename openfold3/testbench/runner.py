from __future__ import annotations

import json
from pathlib import Path

from openfold3.benchmark.harness import DdgBenchmarkHarness, HarnessReport
from openfold3.benchmark.methods import default_methods
from openfold3.benchmark.models import BenchmarkCase, MutationInput

from .evaluation import EvaluationSummary, evaluate_reports
from .models import DatasetKind, TestbenchConfig
from .registry import SQLiteRegistry


def _format_result_brief(report: HarnessReport) -> str:
    parts: list[str] = []
    for result in report.results:
        if result.score is None:
            parts.append(f"{result.method}={result.status}")
        else:
            parts.append(f"{result.method}={result.status}({result.score:.4f})")
    return " ".join(parts)


def _parse_mutation_payload(payload: dict[str, object]) -> MutationInput:
    return MutationInput(
        chain_id=str(payload["chain_id"]),
        from_residue=str(payload["from_residue"]),
        position_1based=int(payload["position_1based"]),
        to_residue=str(payload["to_residue"]),
    )


def load_cases_from_json(path: Path) -> list[BenchmarkCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload["cases"] if isinstance(payload, dict) and "cases" in payload else payload
    cases: list[BenchmarkCase] = []
    for item in raw_cases:
        mutations = tuple(
            _parse_mutation_payload(mutation_payload)
            for mutation_payload in item.get("mutations", [])
        )
        chain_groups = tuple(tuple(group) for group in item.get("chain_groups", []))
        cases.append(
            BenchmarkCase(
                case_id=item["case_id"],
                structure_path=Path(item["structure_path"]),
                confidence_path=(
                    None
                    if item.get("confidence_path") is None
                    else Path(item["confidence_path"])
                ),
                mutations=mutations,
                chain_groups=chain_groups,
                experimental_ddg=item.get("experimental_ddg"),
                notes=item.get("notes"),
            )
        )
    return cases


class TestbenchRunner:
    def __init__(self, config: TestbenchConfig, harness: DdgBenchmarkHarness | None = None):
        self.config = config.resolved()
        self.harness = harness or DdgBenchmarkHarness(methods=default_methods())

    def run_cases(self, cases: list[BenchmarkCase]) -> tuple[int, list[HarnessReport], EvaluationSummary]:
        output_root = self.config.output_root
        reports_dir = output_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        registry = SQLiteRegistry(self.config.registry_path or (output_root / "registry.sqlite"))
        try:
            run_id = registry.create_run(self.config)
            reports: list[HarnessReport] = []
            total_cases = len(cases)
            for index, case in enumerate(cases, start=1):
                print(f"[{index}/{total_cases}] {case.case_id}", flush=True)
                report = self.harness.run_case(case)
                report_path = reports_dir / f"{case.case_id}.json"
                self.harness.write_report(report, report_path)
                registry.insert_case_report(
                    run_id=run_id,
                    dataset_kind=self.config.dataset_kind,
                    report=report,
                    report_path=report_path,
                )
                reports.append(report)
                print(
                    f"{_format_result_brief(report)} report={report_path}",
                    flush=True,
                )
            summary = evaluate_reports(reports)
            summary_path = output_root / "evaluation_summary.json"
            summary_path.write_text(summary.to_json(), encoding="utf-8")
            manifest_path = output_root / "run_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "output_root": str(output_root),
                        "dataset_kind": self.config.dataset_kind,
                        "gpu_concurrency": self.config.gpu_concurrency,
                        "cpu_prep_workers": self.config.cpu_prep_workers,
                        "cpu_ddg_workers": self.config.cpu_ddg_workers,
                        "registry_path": str(self.config.registry_path),
                        "num_cases": len(cases),
                        "report_paths": [str(reports_dir / f"{case.case_id}.json") for case in cases],
                        "evaluation_summary_path": str(summary_path),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return run_id, reports, summary
        finally:
            registry.close()
