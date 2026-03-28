from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from of_notebook_lib.config import RuntimeConfig, _path_from_env
from of_notebook_lib.runner import RunResult, _slug_timestamp, ensure_msa_cache_link, run_cmd, run_prediction


def test_path_from_env_uses_default_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENFOLD_TEST_PATH", raising=False)
    assert _path_from_env("OPENFOLD_TEST_PATH", "~/demo").name == "demo"


def test_runtime_config_build_env_sets_expected_keys(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "base_path")
    monkeypatch.setenv("LD_LIBRARY_PATH", "base_ld")

    runtime = RuntimeConfig(
        openfold_prefix=Path("/opt/openfold"),
        triton_cache_dir=Path("/tmp/triton"),
        use_fused_attention=True,
        use_deepspeed=True,
    )
    env = runtime.build_env()

    assert runtime.openfold_runner == Path("/opt/openfold/bin/run_openfold")
    assert Path(env["CUDA_HOME"]) == Path("/opt/openfold")
    assert env["PATH"].startswith(f"{Path('/opt/openfold/bin')}:")
    assert env["LD_LIBRARY_PATH"].startswith(
        f"{Path('/opt/openfold/lib')}:{Path('/opt/openfold/lib64')}:"
    )
    assert Path(env["TRITON_CACHE_DIR"]) == Path("/tmp/triton")
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["OPENFOLD_USE_FUSED_ATTENTION"] == "1"
    assert env["OPENFOLD_USE_DEEPSPEED"] == "1"


def test_slug_timestamp_sanitizes_name() -> None:
    value = _slug_timestamp("name with spaces/and*chars")
    assert value.startswith("name_with_spaces_and_chars_")


def test_ensure_msa_cache_link_reuses_existing_symlink(tmp_path: Path) -> None:
    source = tmp_path / "msa_cache"
    source.mkdir()
    target = tmp_path / "tmp" / "msa_link"
    target.parent.mkdir(parents=True)
    target.symlink_to(source, target_is_directory=True)

    runtime = RuntimeConfig(msa_cache_dir=source, fixed_msa_tmp_dir=target)
    ensure_msa_cache_link(runtime)

    assert target.is_symlink()
    assert target.exists()


def test_ensure_msa_cache_link_replaces_directory(tmp_path: Path) -> None:
    source = tmp_path / "msa_cache"
    source.mkdir()
    target = tmp_path / "tmp" / "msa_link"
    target.mkdir(parents=True)
    (target / "stale.txt").write_text("stale", encoding="utf-8")

    runtime = RuntimeConfig(msa_cache_dir=source, fixed_msa_tmp_dir=target)
    ensure_msa_cache_link(runtime)

    assert target.is_symlink()


def test_run_cmd_writes_log(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"

    class FakeProcess:
        def __init__(self):
            self.stdout = iter(["line one\n", "line two\n"])

        def wait(self):
            return 0

    monkeypatch.setattr("of_notebook_lib.runner.subprocess.Popen", lambda *args, **kwargs: FakeProcess())

    return_code = run_cmd(["cmd"], env={"X": "1"}, log_path=log_path)

    assert return_code == 0
    assert "line one" in log_path.read_text(encoding="utf-8")


def test_run_prediction_full_mocked_flow(monkeypatch, tmp_path: Path) -> None:
    runtime = RuntimeConfig(
        results_dir=tmp_path / "results",
        openfold_prefix=tmp_path / "prefix",
        msa_cache_dir=tmp_path / "msa_cache",
        fixed_msa_tmp_dir=tmp_path / "msa_link",
        triton_cache_dir=tmp_path / "triton",
    )
    runtime.msa_cache_dir.mkdir(parents=True)

    payload = {"queries": {"demo_case": {"chains": [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": "AAAA"}]}}}
    fake_samples = pd.DataFrame(
        [
            {
                "query_name": "demo_case",
                "sample_name": "sample_1",
                "sample_ranking_score": 0.9,
                "iptm": 0.8,
                "ptm": 0.7,
                "avg_plddt": 91.0,
                "gpde": 0.2,
                "has_clash": 0.0,
                "model_path": str(tmp_path / "model.cif"),
            }
        ]
    )

    captured = {}

    def fake_ensure(runtime_arg):
        captured["ensure_called"] = runtime_arg.fixed_msa_tmp_dir

    def fake_run_cmd(cmd, env, log_path):
        captured["cmd"] = cmd
        captured["env"] = env
        log_path.write_text("fake log\n", encoding="utf-8")
        return 0

    fake_sample_objects = ["sample-object"]
    fake_winners = {"sample_ranking_score": "winner"}
    stage_calls: list[tuple[str, str]] = []

    class FakeMonitor:
        def __init__(self, summary_dir: Path, *, sample_interval_seconds: float = 1.0) -> None:
            self.summary_dir = summary_dir
            self.resource_csv_path = summary_dir / "resource_usage.csv"
            self.stage_marks_path = summary_dir / "stage_marks.csv"
            self.monitor_plot_path = summary_dir / "resource_usage.png"

        def start(self) -> None:
            self.summary_dir.mkdir(parents=True, exist_ok=True)

        def record_stage(self, stage: str, details: str = "") -> None:
            stage_calls.append((stage, details))

        def stop(self):
            self.resource_csv_path.write_text("timestamp_utc\n", encoding="utf-8")
            self.stage_marks_path.write_text("timestamp_utc,elapsed_seconds,stage,details\n", encoding="utf-8")
            self.monitor_plot_path.write_bytes(b"\x89PNG\r\n\x1a\n")

            class _Artifacts:
                resource_csv_path = self.resource_csv_path
                stage_marks_path = self.stage_marks_path
                monitor_plot_path = self.monitor_plot_path

            return _Artifacts()

    monkeypatch.setattr("of_notebook_lib.runner.ensure_msa_cache_link", fake_ensure)
    monkeypatch.setattr("of_notebook_lib.runner.run_cmd", fake_run_cmd)
    monkeypatch.setattr("of_notebook_lib.runner.collect_samples", lambda output_dir: fake_sample_objects)
    monkeypatch.setattr("of_notebook_lib.runner.best_samples_by_metric", lambda samples: fake_winners)
    monkeypatch.setattr("of_notebook_lib.runner.write_best_samples_report", lambda *args: captured.setdefault("report_called", True))
    monkeypatch.setattr("of_notebook_lib.runner.copy_best_artifacts", lambda *args: captured.setdefault("copy_called", True))
    monkeypatch.setattr("of_notebook_lib.runner.samples_to_dataframe", lambda samples: fake_samples)
    monkeypatch.setattr("of_notebook_lib.runner.RunMonitor", FakeMonitor)

    result = run_prediction(
        runtime=runtime,
        payload=payload,
        experiment_name="demo case",
        use_templates=False,
        use_msa_server=False,
        num_diffusion_samples=3,
        num_model_seeds=4,
        runner_yaml=tmp_path / "runner.yml",
        inference_ckpt_path=tmp_path / "ckpt",
        inference_ckpt_name="demo_ckpt",
        enable_monitoring=True,
    )

    assert isinstance(result, RunResult)
    assert result.return_code == 0
    assert result.query_path.exists()
    assert json.loads(result.query_path.read_text(encoding="utf-8")) == payload
    assert result.log_path.exists()
    assert result.summary_dir.exists()
    assert (result.summary_dir / "run_params.json").exists()
    assert "--use_templates=false" in captured["cmd"]
    assert "--use_msa_server=false" in captured["cmd"]
    assert any(str(item).startswith("--runner_yaml=") for item in captured["cmd"])
    assert any(str(item).startswith("--inference_ckpt_path=") for item in captured["cmd"])
    assert "--inference_ckpt_name=demo_ckpt" in captured["cmd"]
    assert captured["report_called"] is True
    assert captured["copy_called"] is True
    assert result.resource_csv_path is not None and result.resource_csv_path.exists()
    assert result.stage_marks_path is not None and result.stage_marks_path.exists()
    assert result.monitor_plot_path is not None and result.monitor_plot_path.exists()
    assert [name for name, _ in stage_calls] == [
        "query_written",
        "runtime_prepared",
        "openfold_started",
        "openfold_finished",
        "samples_collected",
        "summary_written",
    ]
