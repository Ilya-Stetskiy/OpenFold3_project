from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pandas as pd

from of_notebook_lib.config import RuntimeConfig
from of_notebook_lib.runner import RunResult
from of_notebook_lib.screening import (
    BatchApproachComparison,
    ScreeningBatchResult,
    compare_mutation_batch_approaches,
    run_screened_mutation_scan,
    run_server_end_to_end_smoke,
)


def _fake_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "openfold-3"
    package_dir = repo_root / "openfold3"
    package_dir.mkdir(parents=True)
    (package_dir / "run_openfold.py").write_text("# fake\n", encoding="utf-8")
    return repo_root


def test_run_screened_mutation_scan_builds_job_and_parses_results(
    big_ace_molecules: list[dict],
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = RuntimeConfig(
        results_dir=tmp_path / "results",
        openfold_prefix=tmp_path / "prefix",
        openfold_repo_dir=_fake_repo(tmp_path),
    )

    def fake_run(cmd, *, env, cwd, log_path):
        del env
        del cwd
        job_json_path = Path(cmd[-1])
        job = json.loads(job_json_path.read_text(encoding="utf-8"))
        output_dir = Path(job["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "mutation_id": "WT",
                "query_id": "scan_case_WT",
                "sample_ranking_score": 0.9,
                "iptm": 0.8,
                "ptm": 0.7,
                "avg_plddt": 90.0,
                "gpde": 0.1,
                "has_clash": 0.0,
                "cpu_prep_seconds": 0.2,
                "gpu_inference_seconds": 0.4,
                "total_seconds": 0.6,
                "query_result_cache_hit": False,
            },
            {
                "mutation_id": "B_F4G",
                "query_id": "scan_case_B_F4G",
                "sample_ranking_score": 0.8,
                "iptm": 0.75,
                "ptm": 0.65,
                "avg_plddt": 88.0,
                "gpde": 0.2,
                "has_clash": 0.0,
                "cpu_prep_seconds": 0.1,
                "gpu_inference_seconds": 0.3,
                "total_seconds": 0.4,
                "query_result_cache_hit": True,
            },
        ]
        (output_dir / "results.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        log_path.write_text("ok\n", encoding="utf-8")
        return 0, 3.5

    monkeypatch.setattr("of_notebook_lib.screening._run_timed_cmd", fake_run)

    result = run_screened_mutation_scan(
        runtime=runtime,
        experiment_name="scan_case",
        molecules=big_ace_molecules,
        mutation_chain_id="B",
        position_1based=4,
        amino_acids="FG",
        include_wt=True,
        cache_query_results=False,
        subprocess_batch_size=4,
    )

    job = json.loads(result.job_json_path.read_text(encoding="utf-8"))
    assert result.elapsed_seconds == 3.5
    assert result.return_code == 0
    assert job["mutations"][0]["chain_id"] == "B"
    assert job["cache_query_results"] is False
    assert job["subprocess_batch_size"] == 4
    assert set(result.rows_df["mutation_label"]) == {"WT", "B_F4G"}
    assert not result.mutation_ranking.empty


def test_run_screened_mutation_scan_falls_back_to_installed_runner_without_repo_checkout(
    big_ace_molecules: list[dict],
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = RuntimeConfig(
        results_dir=tmp_path / "results",
        openfold_prefix=tmp_path / "prefix",
        openfold_repo_dir=tmp_path / "missing-openfold-3",
        project_dir=tmp_path / "project",
    )
    runtime.project_dir.mkdir(parents=True)
    runtime.openfold_runner.parent.mkdir(parents=True)
    runtime.openfold_runner.write_text("#!/bin/sh\n", encoding="utf-8")

    captured = {}

    def fake_run(cmd, *, env, cwd, log_path):
        del env
        job_json_path = Path(cmd[-1])
        job = json.loads(job_json_path.read_text(encoding="utf-8"))
        output_dir = Path(job["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.jsonl").write_text(
            json.dumps(
                {
                    "mutation_id": "WT",
                    "query_id": "scan_case_WT",
                    "sample_ranking_score": 0.9,
                    "iptm": 0.8,
                    "ptm": 0.7,
                    "avg_plddt": 90.0,
                    "gpde": 0.1,
                    "has_clash": 0.0,
                    "total_seconds": 0.6,
                    "query_result_cache_hit": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        log_path.write_text("ok\n", encoding="utf-8")
        captured["cwd"] = cwd
        return 0, 1.0

    monkeypatch.setattr("of_notebook_lib.screening._run_timed_cmd", fake_run)

    result = run_screened_mutation_scan(
        runtime=runtime,
        experiment_name="scan_case",
        molecules=big_ace_molecules,
        mutation_chain_id="B",
        position_1based=4,
        amino_acids="FG",
        include_wt=True,
    )

    assert result.return_code == 0
    assert captured["cwd"] == runtime.project_dir.resolve()


def test_compare_mutation_batch_approaches_writes_speedup_summary(
    big_ace_molecules: list[dict],
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = RuntimeConfig(results_dir=tmp_path / "results")
    fake_predict = RunResult(
        experiment_name="predict_case",
        run_dir=tmp_path / "predict_run",
        query_path=tmp_path / "query.json",
        output_dir=tmp_path / "predict_output",
        summary_dir=tmp_path / "predict_summary",
        log_path=tmp_path / "predict.log",
        samples_df=pd.DataFrame(),
        elapsed_seconds=12.0,
        return_code=0,
    )
    fake_screen = ScreeningBatchResult(
        experiment_name="screen_case",
        run_dir=tmp_path / "screen_run",
        job_json_path=tmp_path / "screen_job.json",
        output_dir=tmp_path / "screen_output",
        cache_dir=tmp_path / "screen_cache",
        log_path=tmp_path / "screen.log",
        summary_path=tmp_path / "screen_summary.json",
        rows_df=pd.DataFrame(
            [
                {
                    "query_result_cache_hit": True,
                    "total_seconds": 2.0,
                },
                {
                    "query_result_cache_hit": False,
                    "total_seconds": 3.0,
                },
            ]
        ),
        mutation_summary=pd.DataFrame(),
        mutation_ranking=pd.DataFrame(),
        elapsed_seconds=4.0,
        return_code=0,
    )

    monkeypatch.setattr("of_notebook_lib.screening.run_prediction", lambda *args, **kwargs: fake_predict)
    monkeypatch.setattr(
        "of_notebook_lib.screening.run_screened_mutation_scan",
        lambda *args, **kwargs: fake_screen,
    )

    result = compare_mutation_batch_approaches(
        runtime=runtime,
        experiment_name="compare_case",
        molecules=big_ace_molecules,
        mutation_chain_id="B",
        position_1based=4,
        amino_acids="FG",
        cache_query_results=False,
        subprocess_batch_size=8,
    )

    assert isinstance(result, BatchApproachComparison)
    assert result.comparison["speedup_ratio"] == 3.0
    assert result.comparison["time_saved_seconds"] == 8.0
    assert result.comparison["cache_query_results"] is False
    assert result.comparison["subprocess_batch_size"] == 8
    assert result.summary_path.exists()


def test_run_server_end_to_end_smoke_writes_summary(
    big_ace_molecules: list[dict],
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = RuntimeConfig(results_dir=tmp_path / "results")
    fake_predict = RunResult(
        experiment_name="single_case",
        run_dir=tmp_path / "single_run",
        query_path=tmp_path / "query.json",
        output_dir=tmp_path / "single_output",
        summary_dir=tmp_path / "single_summary",
        log_path=tmp_path / "single.log",
        samples_df=pd.DataFrame(),
        elapsed_seconds=5.0,
        return_code=0,
    )
    fake_screen = ScreeningBatchResult(
        experiment_name="screen_case",
        run_dir=tmp_path / "screen_run",
        job_json_path=tmp_path / "screen_job.json",
        output_dir=tmp_path / "screen_output",
        cache_dir=tmp_path / "screen_cache",
        log_path=tmp_path / "screen.log",
        summary_path=tmp_path / "screen_summary.json",
        rows_df=pd.DataFrame(),
        mutation_summary=pd.DataFrame(),
        mutation_ranking=pd.DataFrame(),
        elapsed_seconds=7.0,
        return_code=0,
    )

    monkeypatch.setattr("of_notebook_lib.screening._probe_gpu", lambda: {"available": True})
    monkeypatch.setattr("of_notebook_lib.screening.run_prediction", lambda *args, **kwargs: fake_predict)
    monkeypatch.setattr(
        "of_notebook_lib.screening.run_screened_mutation_scan",
        lambda *args, **kwargs: fake_screen,
    )

    result = run_server_end_to_end_smoke(
        runtime=runtime,
        experiment_name="server_case",
        molecules=big_ace_molecules,
        mutation_chain_id="B",
        position_1based=4,
        amino_acids="FG",
        cache_query_results=False,
        subprocess_batch_size=6,
    )

    assert result.gpu_probe["available"] is True
    assert result.summary_path.exists()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["single_elapsed_seconds"] == 5.0
    assert summary["screening_elapsed_seconds"] == 7.0
    assert summary["cache_query_results"] is False
    assert summary["subprocess_batch_size"] == 6


def test_run_compare_mutation_batch_script_invokes_workflow(
    big_ace_molecules: list[dict],
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    query_json = tmp_path / "query.json"
    query_json.write_text(
        json.dumps({"queries": {"mini_query": {"chains": big_ace_molecules}}}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeResult:
        summary_path = tmp_path / "comparison_summary.json"
        comparison = {"speedup_ratio": 2.0, "cache_query_results": False}

    monkeypatch.setattr("of_notebook_lib.RuntimeConfig", lambda: "runtime")

    def fake_compare(**kwargs):
        captured.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(
        "of_notebook_lib.workflows.compare_mutation_batch_case",
        fake_compare,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_compare_mutation_batch.py",
            "--query-json",
            str(query_json),
            "--query-id",
            "mini_query",
            "--mutation-chain-id",
            "B",
            "--position-1based",
            "4",
            "--amino-acids",
            "FG",
            "--subprocess-batch-size",
            "8",
            "--no-query-result-cache",
        ],
    )

    runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "server_smoke"
            / "run_compare_mutation_batch.py"
        ),
        run_name="__main__",
    )
    stdout = capsys.readouterr().out

    assert captured["runtime"] == "runtime"
    assert captured["mutation_chain_id"] == "B"
    assert captured["position_1based"] == 4
    assert captured["cache_query_results"] is False
    assert captured["subprocess_batch_size"] == 8
    assert "summary_path=" in stdout
