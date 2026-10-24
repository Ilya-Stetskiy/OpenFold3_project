import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check_env = load_module("scripts/run/check_server_environment.py", "check_server_environment")


def test_check_server_environment_uses_effective_server_paths(tmp_path, monkeypatch):
    project_root = tmp_path / "server_project"
    data_root = tmp_path / "server_data"
    results_root = tmp_path / "server_results"
    logs_root = tmp_path / "server_logs"
    reports_root = tmp_path / "server_reports"
    for path in [project_root, data_root, results_root, logs_root, reports_root]:
        path.mkdir(parents=True)

    repo_root = tmp_path / "repo"
    (repo_root / "configs").mkdir(parents=True)
    (repo_root / "data/processed/current_internal").mkdir(parents=True)
    (repo_root / "configs/datasets.yaml").write_text("datasets: {}\n", encoding="utf-8")
    (repo_root / "configs/run_stages.yaml").write_text("stage_groups: {}\n", encoding="utf-8")
    (repo_root / "data/processed/current_internal/current_internal_recalc_jobs_resolved.csv").write_text(
        "method,structure_source,input_ready,input_status\n", encoding="utf-8"
    )

    monkeypatch.setattr(check_env, "ROOT", repo_root)
    monkeypatch.setattr(check_env, "REPORT_DIR", repo_root / "reports/server_environment")

    server_config = {
        "server": {
            "project_root": str(project_root),
            "data_root": str(data_root),
            "results_root": str(results_root),
            "logs_root": str(logs_root),
            "reports_root": str(reports_root),
        }
    }
    meta = {"server_config_exists": True, "server_config_is_example_fallback": False}

    checks, summary = check_env.check_stage_group("all", server_config, meta)

    assert summary["effective_project_root"] == str(project_root)
    assert summary["effective_data_root"] == str(data_root)
    assert summary["effective_results_root"] == str(results_root)
    assert summary["effective_logs_root"] == str(logs_root)
    assert summary["effective_reports_root"] == str(reports_root)
    checked_paths = {row["path"] for row in checks}
    assert str(results_root) in checked_paths
    assert str(logs_root) in checked_paths
    assert str(reports_root) in checked_paths


def test_notebook_contains_repo_root_detection_and_safe_defaults():
    nb = json.loads((ROOT / "notebooks/server_run_control_current_internal.ipynb").read_text(encoding="utf-8"))
    text = "\n".join(
        cell.get("source", "") if isinstance(cell.get("source", ""), str) else "".join(cell.get("source", []))
        for cell in nb["cells"]
    )
    required = [
        "find_repo_root",
        "configs/datasets.yaml",
        "PYTHON_BIN = None",
        "MAX_PARALLEL_OPENFOLD_JOBS = None",
        "CONFIRM_WRITE_SERVER_CONFIG = False",
        "RUN_FOLDX_FULL = False",
        "RUN_OPENFOLD = False",
    ]
    for needle in required:
        assert needle in text


def test_build_run_bundle_include_server_config_uses_requested_path(tmp_path):
    custom_config = tmp_path / "custom_server_paths.yaml"
    custom_config.write_text(
        "server:\n"
        "  project_root: /tmp/custom_project\n"
        "  results_root: /tmp/custom_results\n"
        "  logs_root: /tmp/custom_logs\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run/build_run_bundle.py",
            "--dataset",
            "current_internal",
            "--stage-group",
            "cpu_heavy",
            "--dry-run",
            "--server-config",
            str(custom_config),
            "--include-server-config",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert str(custom_config) in proc.stdout
    assert "'server_config_exists': True" in proc.stdout
    assert "WARNING: including machine-specific" in proc.stdout


def test_build_run_bundle_include_missing_server_config_fails(tmp_path):
    missing_config = tmp_path / "missing_server_paths.yaml"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run/build_run_bundle.py",
            "--dataset",
            "current_internal",
            "--stage-group",
            "cpu_heavy",
            "--dry-run",
            "--server-config",
            str(missing_config),
            "--include-server-config",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "--include-server-config was set" in proc.stderr
