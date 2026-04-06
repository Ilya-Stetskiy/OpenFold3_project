import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from click.testing import CliRunner

from openfold3.mutation_runner import (
    MutationScreeningRunner,
    MutationSpec,
    ScreeningJob,
    ScreeningResultRow,
    run_screening_job_from_json,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import Query
from openfold3.run_openfold import cli
from scripts.dev import run_nightly_test_suite, run_overnight_screening


class CapturingBackend:
    def __init__(self):
        self.query_ids = []

    def run(self, prepared_job):
        self.query_ids.append(prepared_job.query_id)
        return ScreeningResultRow(
            mutation_id=prepared_job.mutation_id,
            query_id=prepared_job.query_id,
            query_hash=prepared_job.query_hash,
            sample_index=1,
            seed=7,
            sample_ranking_score=0.5,
            iptm=0.4,
            ptm=0.3,
            avg_plddt=0.2,
            gpde=0.1,
            has_clash=0.0,
            cache_hit=prepared_job.cache_hit,
            sequence_cache_hits=prepared_job.sequence_cache_hits,
            query_result_cache_hit=False,
            cpu_prep_seconds=prepared_job.cpu_prep_seconds,
            gpu_inference_seconds=0.0,
            total_seconds=prepared_job.cpu_prep_seconds,
            output_dir=str(prepared_job.output_dir),
            aggregated_confidence_path=None,
            mutation_spec=None
            if prepared_job.mutation_spec is None
            else {
                "chain_id": prepared_job.mutation_spec.chain_id,
                "position_1based": prepared_job.mutation_spec.position_1based,
                "from_residue": prepared_job.mutation_spec.from_residue,
                "to_residue": prepared_job.mutation_spec.to_residue,
            },
        )


def _write_screening_job_json(path: Path) -> None:
    payload = {
        "base_query": {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": "ACDE",
                }
            ]
        },
        "mutations": [
            {
                "chain_id": "A",
                "position_1based": 2,
                "from_residue": "C",
                "to_residue": "G",
            }
        ],
        "output_dir": str(path.parent / "screening"),
        "cache_dir": str(path.parent / "cache"),
        "query_prefix": "job",
        "include_wt": True,
        "run_baseline_first": True,
        "output_policy": "metrics_only",
        "resume": True,
        "num_cpu_workers": 1,
        "max_inflight_queries": 1,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_query_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "queries": {
                    "mini_query": {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "ACDE",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _write_mutations_csv(path: Path) -> None:
    path.write_text(
        "chain_id,position_1based,from_residue,to_residue\n"
        "A,2,C,G\n"
        "\n"
        "A,3,D,Y\n",
        encoding="utf-8",
    )


def test_run_screening_job_from_json_respects_output_override(tmp_path, monkeypatch):
    job_json = tmp_path / "job.json"
    _write_screening_job_json(job_json)
    captured = {}

    class FakeRunner:
        def run(self, job):
            captured["output_dir"] = job.output_dir
            return []

    monkeypatch.setattr("openfold3.mutation_runner.MutationScreeningRunner", FakeRunner)

    override = tmp_path / "override-output"
    rows = run_screening_job_from_json(job_json, output_dir=override)

    assert rows == []
    assert captured["output_dir"] == override


def test_run_openfold_screen_mutations_cli_invokes_json_runner(tmp_path, monkeypatch):
    job_json = tmp_path / "job.json"
    _write_screening_job_json(job_json)
    captured = {}

    def fake_run_screening_job_from_json(screening_job_json, output_dir=None):
        captured["screening_job_json"] = screening_job_json
        captured["output_dir"] = output_dir
        return []

    monkeypatch.setattr("openfold3.run_openfold._torch_gpu_setup", lambda: None)
    monkeypatch.setattr(
        "openfold3.mutation_runner.run_screening_job_from_json",
        fake_run_screening_job_from_json,
    )

    result = CliRunner().invoke(
        cli,
        [
            "screen-mutations",
            "--screening-job-json",
            str(job_json),
            "--output-dir",
            str(tmp_path / "override"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["screening_job_json"] == job_json
    assert captured["output_dir"] == tmp_path / "override"


def test_run_overnight_screening_build_screening_job_truncates_csv_and_writes_manifests(
    tmp_path,
):
    query_json = tmp_path / "query.json"
    mutations_csv = tmp_path / "mutations.csv"
    _write_query_json(query_json)
    _write_mutations_csv(mutations_csv)

    args = Namespace(
        base_query_json=query_json,
        mutations_csv=mutations_csv,
        output_root=tmp_path / "output",
        runner_yaml=None,
        query_id="mini_query",
        query_prefix=None,
        num_diffusion_samples=4,
        num_model_seeds=1,
        max_mutations=1,
        num_cpu_workers=2,
        max_inflight_queries=2,
        subprocess_batch_size=4,
        min_free_disk_gb=1.0,
        inference_ckpt_path=None,
        inference_ckpt_name=None,
        use_msa_server=False,
        use_templates=False,
        include_wt=True,
        keep_query_outputs=False,
        no_resume=False,
        no_query_result_cache=True,
    )

    job = run_overnight_screening.build_screening_job(args)
    launch_manifest = json.loads(
        (args.output_root / "launch_manifest.json").read_text(encoding="utf-8")
    )

    assert len(job.mutations) == 1
    assert job.mutations[0].mutation_id == "A_C2G"
    assert launch_manifest["mutation_count"] == 1
    assert launch_manifest["max_mutations"] == 1
    assert job.cache_query_results is False
    assert job.subprocess_batch_size == 4
    assert launch_manifest["cache_query_results"] is False
    assert launch_manifest["subprocess_batch_size"] == 4


def test_run_overnight_screening_build_screening_job_rejects_missing_query_id(tmp_path):
    query_json = tmp_path / "query.json"
    mutations_csv = tmp_path / "mutations.csv"
    _write_query_json(query_json)
    _write_mutations_csv(mutations_csv)

    args = Namespace(
        base_query_json=query_json,
        mutations_csv=mutations_csv,
        output_root=tmp_path / "output",
        runner_yaml=None,
        query_id="missing_query",
        query_prefix=None,
        num_diffusion_samples=1,
        num_model_seeds=1,
        max_mutations=1,
        num_cpu_workers=1,
        max_inflight_queries=1,
        min_free_disk_gb=1.0,
        inference_ckpt_path=None,
        inference_ckpt_name=None,
        use_msa_server=False,
        use_templates=False,
        include_wt=True,
        keep_query_outputs=False,
        no_resume=False,
    )

    try:
        run_overnight_screening.build_screening_job(args)
    except ValueError as exc:
        assert "Query id 'missing_query' was not found" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing query id")


def test_run_overnight_screening_build_screening_job_rejects_empty_mutation_csv(tmp_path):
    query_json = tmp_path / "query.json"
    mutations_csv = tmp_path / "mutations.csv"
    _write_query_json(query_json)
    mutations_csv.write_text(
        "chain_id,position_1based,from_residue,to_residue\n",
        encoding="utf-8",
    )

    args = Namespace(
        base_query_json=query_json,
        mutations_csv=mutations_csv,
        output_root=tmp_path / "output",
        runner_yaml=None,
        query_id="mini_query",
        query_prefix=None,
        num_diffusion_samples=1,
        num_model_seeds=1,
        max_mutations=None,
        num_cpu_workers=1,
        max_inflight_queries=1,
        min_free_disk_gb=1.0,
        inference_ckpt_path=None,
        inference_ckpt_name=None,
        use_msa_server=False,
        use_templates=False,
        include_wt=True,
        keep_query_outputs=False,
        no_resume=False,
    )

    try:
        run_overnight_screening.build_screening_job(args)
    except ValueError as exc:
        assert "did not produce any mutations" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty mutation CSV")


def test_run_overnight_screening_main_invokes_runner_with_built_job(
    tmp_path, monkeypatch
):
    query_json = tmp_path / "query.json"
    mutations_csv = tmp_path / "mutations.csv"
    _write_query_json(query_json)
    _write_mutations_csv(mutations_csv)
    captured = {}

    class FakeRunner:
        def run(self, job):
            captured["job"] = job
            return []

    monkeypatch.setattr(
        run_overnight_screening, "MutationScreeningRunner", lambda: FakeRunner()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_overnight_screening.py",
            "--base-query-json",
            str(query_json),
            "--mutations-csv",
            str(mutations_csv),
            "--output-root",
            str(tmp_path / "output"),
            "--query-id",
            "mini_query",
            "--max-mutations",
            "1",
        ],
    )

    run_overnight_screening.main()

    assert captured["job"].query_prefix == "mini_query"
    assert len(captured["job"].mutations) == 1


def test_run_nightly_stream_command_logs_success(tmp_path):
    log_path = tmp_path / "nightly.log"

    run_nightly_test_suite.stream_command(
        [sys.executable, "-c", "print('hello from test')"],
        cwd=tmp_path,
        log_path=log_path,
        step_name="echo",
    )

    text = log_path.read_text(encoding="utf-8")
    assert "[echo] START" in text
    assert "[echo] hello from test" in text
    assert "[echo] OK" in text


def test_run_nightly_stream_command_logs_failure(tmp_path):
    log_path = tmp_path / "nightly.log"

    try:
        run_nightly_test_suite.stream_command(
            [sys.executable, "-c", "import sys; sys.exit(3)"],
            cwd=tmp_path,
            log_path=log_path,
            step_name="fail",
        )
    except subprocess.CalledProcessError as exc:
        assert exc.returncode == 3
    else:
        raise AssertionError("Expected CalledProcessError")

    assert "FAIL exit_code=3" in log_path.read_text(encoding="utf-8")


def test_run_nightly_main_builds_expected_commands(tmp_path, monkeypatch):
    commands = []
    stages = []

    class FakeMonitor:
        def __init__(self, summary_dir):
            self.artifacts = Namespace(
                resource_csv_path=summary_dir / "resource_usage.csv",
                stage_marks_path=summary_dir / "stage_marks.csv",
                monitor_plot_path=summary_dir / "resource_usage.png",
            )

        def start(self):
            return None

        def stop(self):
            return self.artifacts

        def record_stage(self, stage, details=""):
            stages.append((stage, details))

    args = Namespace(
        output_root=tmp_path / "suite",
        comparison_query_json=tmp_path / "comparison.json",
        comparison_query_id="cmp",
        leucine_query_json=tmp_path / "leucine.json",
        leucine_query_id="leu",
        mutations_csv=tmp_path / "mutations.csv",
        runner_yaml=tmp_path / "runner.yml",
        python_bin="python",
        num_diffusion_samples=4,
        num_model_seeds=1,
        max_mutations=30,
        comparison_tolerance=0.1,
        num_cpu_workers=2,
        max_inflight_queries=2,
        min_free_disk_gb=2.0,
        inference_ckpt_path=None,
        inference_ckpt_name=None,
        include_wt=True,
        use_msa_server=False,
        use_templates=False,
    )

    monkeypatch.setattr(run_nightly_test_suite, "parse_args", lambda: args)
    monkeypatch.setattr(run_nightly_test_suite, "RunMonitor", FakeMonitor)
    monkeypatch.setattr(
        run_nightly_test_suite,
        "stream_command",
        lambda cmd, cwd, log_path, step_name: commands.append((step_name, cmd, cwd)),
    )

    run_nightly_test_suite.main()

    assert [step for step, _, _ in commands] == ["compare", "leucine"]
    assert "--max-mutations" in commands[1][1]
    assert "30" in commands[1][1]
    assert any(stage == "suite_finished" for stage, _ in stages)


def test_run_screening_job_from_json_with_real_runner_and_capturing_backend(tmp_path):
    job_json = tmp_path / "job.json"
    _write_screening_job_json(job_json)
    job = ScreeningJob.from_json_file(job_json)
    backend = CapturingBackend()

    rows = MutationScreeningRunner(backend=backend).run(job)

    assert [row.mutation_id for row in rows] == ["WT", "A_C2G"]
    assert backend.query_ids == ["job_WT", "job_A_C2G"]
