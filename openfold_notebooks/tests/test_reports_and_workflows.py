from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from of_notebook_lib import RuntimeConfig
from of_notebook_lib.analysis import (
    best_samples_by_metric,
    collect_samples,
    copy_best_artifacts,
    write_best_samples_report,
)
from of_notebook_lib.workflows import run_mutation_scan, run_single_case


def test_report_and_artifact_copy_use_tmp_path(big_ace_output_dir: Path, tmp_path: Path) -> None:
    samples = collect_samples(big_ace_output_dir)
    winners = best_samples_by_metric(samples)
    report_path = tmp_path / "best_samples_report.txt"
    summary_dir = tmp_path / "summary"

    write_best_samples_report(report_path, samples, winners)
    copy_best_artifacts(summary_dir, winners)

    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8")
    assert any(summary_dir.rglob("*_confidences_aggregated.json"))
    assert any(summary_dir.rglob("*_model.cif"))


@dataclass
class _FakeRunResult:
    experiment_name: str
    samples_df: pd.DataFrame
    elapsed_seconds: float = 1.0


def test_run_single_case_builds_payload_without_running_openfold(
    big_ace_molecules: list[dict],
    monkeypatch: object,
) -> None:
    captured: dict = {}

    def fake_run_prediction(runtime, payload, experiment_name, **kwargs):
        captured["runtime"] = runtime
        captured["payload"] = payload
        captured["experiment_name"] = experiment_name
        captured["kwargs"] = kwargs
        return _FakeRunResult(experiment_name=experiment_name, samples_df=pd.DataFrame())

    monkeypatch.setattr("of_notebook_lib.workflows.run_prediction", fake_run_prediction)

    runtime = RuntimeConfig(results_dir=Path("/tmp/results"))
    result = run_single_case(
        runtime=runtime,
        experiment_name="single_case",
        molecules=big_ace_molecules,
        mutation={"enabled": True, "chain_id": "B", "position_1based": 4, "new_residue": "G"},
    )

    assert result.experiment_name == "single_case"
    assert set(captured["payload"]["queries"]) == {"single_case"}
    chains = captured["payload"]["queries"]["single_case"]["chains"]
    assert chains[1]["sequence"][3] == "G"


def test_run_mutation_scan_uses_mocked_prediction_result(
    big_ace_molecules: list[dict],
    monkeypatch: object,
) -> None:
    fake_samples = pd.DataFrame(
        [
            {
                "query_name": "scan_case__WT",
                "mutation_label": "WT",
                "sample_name": "sample_wt",
                "sample_ranking_score": 0.9,
                "iptm": 0.8,
                "ptm": 0.7,
                "avg_plddt": 90.0,
                "gpde": 0.1,
                "has_clash": 0.0,
            },
            {
                "query_name": "scan_case__B_F4G",
                "mutation_label": "B_F4G",
                "sample_name": "sample_mut",
                "sample_ranking_score": 0.8,
                "iptm": 0.75,
                "ptm": 0.69,
                "avg_plddt": 88.0,
                "gpde": 0.2,
                "has_clash": 0.0,
            },
        ]
    )

    captured: dict = {}

    def fake_run_prediction(runtime, payload, experiment_name, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeRunResult(experiment_name=experiment_name, samples_df=fake_samples)

    monkeypatch.setattr("of_notebook_lib.workflows.run_prediction", fake_run_prediction)

    runtime = RuntimeConfig(results_dir=Path("/tmp/results"))
    result, mutation_summary, mutation_ranking = run_mutation_scan(
        runtime=runtime,
        experiment_name="scan_case",
        molecules=big_ace_molecules,
        mutation_chain_id="B",
        position_1based=4,
        amino_acids="AG",
        include_wt=True,
    )

    assert result.experiment_name == "scan_case"
    assert len(mutation_summary) == 2
    assert not mutation_ranking.empty
    assert set(mutation_summary["mutation_label"]) == {"WT", "B_F4G"}
    assert captured["kwargs"]["enable_monitoring"] is True
