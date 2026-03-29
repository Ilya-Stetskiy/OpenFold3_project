from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from of_notebook_lib.config import RuntimeConfig, _path_from_env
from of_notebook_lib.runner import RunResult, _slug_timestamp, ensure_msa_cache_link, run_cmd, run_prediction


def test_path_from_env_uses_default_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENFOLD_TEST_PATH", raising=False)
    assert _path_from_env("OPENFOLD_TEST_PATH", "~/demo").name == "demo"


def test_runtime_config_defaults_follow_current_project_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for env_name in (
        "OPENFOLD_PROJECT_DIR",
        "OPENFOLD_REPO_DIR",
        "OPENFOLD_PREFIX",
        "OPENFOLD_RESULTS_DIR",
        "OPENFOLD_MSA_CACHE_DIR",
        "OPENFOLD_TRITON_CACHE_DIR",
        "OPENFOLD_FIXED_MSA_TMP_DIR",
    ):
        monkeypatch.delenv(env_name, raising=False)

    runtime = RuntimeConfig()

    assert runtime.project_dir == tmp_path.resolve()
    assert runtime.openfold_repo_dir == (tmp_path / "openfold-3").resolve()
    assert runtime.openfold_prefix == (tmp_path / ".venv").resolve()
    assert runtime.results_dir == (tmp_path / "results").resolve()
    assert runtime.msa_cache_dir == (
        tmp_path / "msa_cache" / "colabfold_msas"
    ).resolve()
    assert runtime.triton_cache_dir == (tmp_path / ".runtime" / "triton_cache").resolve()
    assert runtime.fixed_msa_tmp_dir == (
        tmp_path / ".runtime" / "of3_colabfold_msas"
    ).resolve()


def test_runtime_config_build_env_sets_expected_keys(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "base_path")
    monkeypatch.setenv("LD_LIBRARY_PATH", "base_ld")
    prefix = Path("/tmp/openfold_prefix_test")
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    (prefix / "lib").mkdir(parents=True, exist_ok=True)
    (prefix / "lib64").mkdir(parents=True, exist_ok=True)

    runtime = RuntimeConfig(
        openfold_prefix=prefix,
        triton_cache_dir=Path("/tmp/triton"),
        use_fused_attention=True,
        use_deepspeed=True,
    )
    env = runtime.build_env()

    assert runtime.openfold_runner == prefix / "bin" / "run_openfold"
    assert Path(env["CUDA_HOME"]) == prefix
    assert env["PATH"].startswith(f"{prefix / 'bin'}:")
    assert env["LD_LIBRARY_PATH"].startswith(
        f"{prefix / 'lib'}:{prefix / 'lib64'}:"
    )
    assert Path(env["TRITON_CACHE_DIR"]) == Path("/tmp/triton")
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["OPENFOLD_USE_FUSED_ATTENTION"] == "1"
    assert env["OPENFOLD_USE_DEEPSPEED"] == "1"


def test_runtime_config_openfold_runner_falls_back_to_active_env(monkeypatch, tmp_path: Path) -> None:
    active_python = tmp_path / "env" / "bin" / "python"
    active_runner = tmp_path / "env" / "bin" / "run_openfold"
    active_python.parent.mkdir(parents=True)
    active_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    active_runner.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    monkeypatch.setattr("of_notebook_lib.config.sys.executable", str(active_python))

    runtime = RuntimeConfig(openfold_prefix=tmp_path / "missing-prefix")

    assert runtime.openfold_runner == active_runner.resolve()


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


def test_ensure_msa_cache_link_rejects_non_temp_directory(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "msa_cache"
    source.mkdir()
    target = tmp_path / "persistent" / "msa_link"
    target.mkdir(parents=True)

    runtime = RuntimeConfig(msa_cache_dir=source, fixed_msa_tmp_dir=target)
    monkeypatch.setattr(
        "of_notebook_lib.runner._is_safe_temp_target",
        lambda path, *, extra_roots=(): False,
    )

    try:
        ensure_msa_cache_link(runtime)
    except RuntimeError as exc:
        assert "Refusing to replace non-temporary directory" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for unsafe MSA link replacement")


def test_ensure_msa_cache_link_directory_mode_uses_plain_directory(tmp_path: Path) -> None:
    source = tmp_path / "msa_cache"
    source.mkdir()
    target = tmp_path / "tmp" / "msa_link"

    runtime = RuntimeConfig(
        msa_cache_dir=source,
        fixed_msa_tmp_dir=target,
        msa_tmp_mode="directory",
    )
    ensure_msa_cache_link(runtime)

    assert target.exists()
    assert target.is_dir()
    assert not target.is_symlink()


def test_ensure_msa_cache_link_directory_mode_allows_project_runtime_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "msa_cache"
    source.mkdir()
    project_dir = tmp_path / "project"
    target = project_dir / ".runtime" / "of3_colabfold_msas"
    target.mkdir(parents=True)
    (target / "stale.txt").write_text("stale", encoding="utf-8")

    runtime = RuntimeConfig(
        project_dir=project_dir,
        msa_cache_dir=source,
        fixed_msa_tmp_dir=target,
        msa_tmp_mode="directory",
    )
    ensure_msa_cache_link(runtime)

    assert target.exists()
    assert target.is_dir()
    assert not target.is_symlink()


def test_run_cmd_writes_log(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"

    class FakeProcess:
        def __init__(self):
            self.stdout = iter(["line one\n", "line two\n"])

        def wait(self):
            return 0

    monkeypatch.setattr("of_notebook_lib.runner.subprocess.Popen", lambda *args, **kwargs: FakeProcess())

    return_code = run_cmd(["cmd"], env={"X": "1"}, log_path=log_path, cwd=tmp_path)

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

    def fake_run_cmd(cmd, env, log_path, *, cwd=None):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["cwd"] = cwd
        log_path.write_text("fake log\n", encoding="utf-8")
        return 0

    fake_sample_objects = ["sample-object"]
    fake_winners = {"sample_ranking_score": "winner"}

    monkeypatch.setattr("of_notebook_lib.runner.ensure_msa_cache_link", fake_ensure)
    monkeypatch.setattr("of_notebook_lib.runner.run_cmd", fake_run_cmd)
    monkeypatch.setattr("of_notebook_lib.runner.collect_samples", lambda output_dir: fake_sample_objects)
    monkeypatch.setattr("of_notebook_lib.runner.best_samples_by_metric", lambda samples: fake_winners)
    monkeypatch.setattr("of_notebook_lib.runner.write_best_samples_report", lambda *args: captured.setdefault("report_called", True))
    monkeypatch.setattr("of_notebook_lib.runner.copy_best_artifacts", lambda *args: captured.setdefault("copy_called", True))
    monkeypatch.setattr("of_notebook_lib.runner.samples_to_dataframe", lambda samples: fake_samples)

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
    )

    assert isinstance(result, RunResult)
    assert result.return_code == 0
    assert result.elapsed_seconds >= 0.0
    assert result.query_path.exists()
    assert json.loads(result.query_path.read_text(encoding="utf-8")) == payload
    assert result.log_path.exists()
    assert result.summary_dir.exists()
    assert (result.summary_dir / "run_params.json").exists()
    assert captured["cwd"] == runtime.project_dir.resolve()
    assert "--use_templates=false" in captured["cmd"]
    assert "--use_msa_server=false" in captured["cmd"]
    assert any(str(item).startswith("--runner_yaml=") for item in captured["cmd"])
    assert any(str(item).startswith("--inference_ckpt_path=") for item in captured["cmd"])
    assert "--inference_ckpt_name=demo_ckpt" in captured["cmd"]
    assert captured["report_called"] is True
    assert captured["copy_called"] is True


def test_run_prediction_raises_on_nonzero_return_code(monkeypatch, tmp_path: Path) -> None:
    runtime = RuntimeConfig(
        results_dir=tmp_path / "results",
        openfold_prefix=tmp_path / "prefix",
        msa_cache_dir=tmp_path / "msa_cache",
        fixed_msa_tmp_dir=Path("/tmp/of3_colabfold_msas_test"),
    )
    runtime.msa_cache_dir.mkdir(parents=True)

    monkeypatch.setattr("of_notebook_lib.runner.ensure_msa_cache_link", lambda runtime_arg: None)
    monkeypatch.setattr(
        "of_notebook_lib.runner.run_cmd",
        lambda cmd, env, log_path, *, cwd=None: 17,
    )

    try:
        run_prediction(
            runtime=runtime,
            payload={"queries": {}},
            experiment_name="broken case",
        )
    except subprocess.CalledProcessError as exc:
        assert exc.returncode == 17
    else:
        raise AssertionError("Expected CalledProcessError for failed run_openfold call")
