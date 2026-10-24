import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "run"))
sys.path.insert(0, str(ROOT / "scripts" / "executors"))


def load_module(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


execution_utils = load_module("scripts/run/execution_utils.py", "execution_utils")
foldx_adapter = load_module("scripts/executors/foldx_adapter.py", "foldx_adapter")
runner = load_module("scripts/executors/run_foldx_experimental.py", "run_foldx_experimental")


def make_repo(root: Path) -> Path:
    (root / "data/processed/current_internal").mkdir(parents=True)
    (root / "data/structures/chain/current_internal").mkdir(parents=True)
    (root / "configs/tools").mkdir(parents=True)
    (root / "data/structures/chain/current_internal/1ABC_A.pdb").write_text(
        "ATOM      1  CA  ASP A 105      0.000   0.000   0.000  1.00 10.00           C\n",
        encoding="utf-8",
    )
    (root / "results/current_internal/openfold/wt/seq_1").mkdir(parents=True)
    (root / "results/current_internal/openfold/wt/seq_1/model.pdb").write_text(
        "ATOM      1  CA  ASP A 105      0.000   0.000   0.000  1.00 10.00           C\n",
        encoding="utf-8",
    )
    (root / "data/processed/current_internal/current_internal_recalc_jobs_resolved.csv").write_text(
        "dataset_name,job_id,mutation_record_id,method,structure_source,input_ready,chain_id,mutation,experimental_structure_path\n"
        "current_internal,job_foldx_ready,mut_001,foldx,experimental,true,A,D105A,data/structures/chain/current_internal/1ABC_A.pdb\n"
        "current_internal,job_foldx_blocked,mut_002,foldx,experimental,false,A,D106A,data/structures/chain/current_internal/1ABC_A.pdb\n"
        "current_internal,job_rosetta_ready,mut_003,rosetta,experimental,true,A,D107A,data/structures/chain/current_internal/1ABC_A.pdb\n"
        "current_internal,job_foldx_openfold,mut_004,foldx,openfold,true,A,D108A,data/structures/chain/current_internal/1ABC_A.pdb\n",
        encoding="utf-8",
    )
    (root / "data/processed/current_internal/current_internal_recalc_jobs_openfold_ready.csv").write_text(
        "dataset_name,job_id,mutation_record_id,method,structure_source,input_ready,chain_id,mutation,openfold_structure_path\n"
        "current_internal,job_foldx_openfold_ready,mut_004,foldx,openfold,true,A,D108A,results/current_internal/openfold/wt/seq_1/model.pdb\n"
        "current_internal,job_foldx_openfold_blocked,mut_005,foldx,openfold,false,A,D109A,results/current_internal/openfold/wt/seq_1/model.pdb\n",
        encoding="utf-8",
    )
    config = root / "configs/tools/foldx.yaml"
    config.write_text(
        """
foldx:
  binary_path: /missing/foldx
  number_of_runs: 3
  timeout_seconds: 600
paths:
  structure_root: data/structures/chain/current_internal
  results_root: results/current_internal/foldx
  logs_root: logs/foldx_runs
""".lstrip(),
        encoding="utf-8",
    )
    return config


def make_fake_foldx(root: Path) -> Path:
    binary = root / "fake_foldx.py"
    binary.write_text(
        """#!/usr/bin/env python3
import sys
from pathlib import Path

args = " ".join(sys.argv[1:])
if "--command=RepairPDB" in args:
    Path("RepairPDB_input_structure.pdb").write_text("REPAIRED\\n", encoding="utf-8")
elif "--command=BuildModel" in args:
    Path("Dif_fake.fxout").write_text("Pdb\\tddG\\nmodel_1\\t-1.23\\n", encoding="utf-8")
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


def patch_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(execution_utils, "ROOT", tmp_path)


def test_dry_run_selects_only_ready_foldx_experimental_jobs(tmp_path, monkeypatch):
    make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    jobs = runner.select_jobs("current_internal")

    assert [job["job_id"] for job in jobs] == ["job_foldx_ready"]


def test_structure_source_openfold_selects_openfold_ready_jobs(tmp_path, monkeypatch):
    make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    jobs = runner.select_jobs("current_internal", structure_source="openfold")

    assert [job["job_id"] for job in jobs] == ["job_foldx_openfold_ready"]
    assert runner.job_dir("current_internal", jobs[0]) == tmp_path / "results/current_internal/foldx/openfold/mut_004"


def test_materialize_plan_creates_pending_manifest_and_individual_list(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    assert (
        runner.main(
            [
                "--dataset",
                "current_internal",
                "--config",
                str(config),
                "--materialize-plan",
                "--force",
            ]
        )
        == 0
    )

    job_dir = tmp_path / "results/current_internal/foldx/experimental/mut_001"
    manifest = json.loads((job_dir / "job_manifest.json").read_text(encoding="utf-8"))
    assert manifest["job_status"] == "pending"
    assert (job_dir / "individual_list.txt").read_text(encoding="utf-8") == "DA105A;\n"
    assert not (job_dir / "parsed_result.json").exists()


def test_materialize_plan_refuses_existing_stage_manifest_without_force(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)
    manifest_dir = tmp_path / "results/current_internal/_manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "foldx_experimental_jobs.csv").write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="Existing stage manifest.*Use --force"):
        runner.main(["--dataset", "current_internal", "--config", str(config), "--materialize-plan", "--max-jobs", "1"])

    assert not (tmp_path / "results/current_internal/foldx/experimental/mut_001/job_manifest.json").exists()
    assert (manifest_dir / "foldx_experimental_jobs.csv").read_text(encoding="utf-8") == "sentinel\n"


def test_materialize_plan_overwrites_existing_stage_manifest_with_force(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)
    manifest_dir = tmp_path / "results/current_internal/_manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "foldx_experimental_jobs.csv").write_text("sentinel\n", encoding="utf-8")

    assert runner.main(["--dataset", "current_internal", "--config", str(config), "--materialize-plan", "--max-jobs", "1", "--force"]) == 0

    assert "sentinel" not in (manifest_dir / "foldx_experimental_jobs.csv").read_text(encoding="utf-8")


def test_materialize_openfold_uses_openfold_output_layout(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    assert (
        runner.main(
            [
                "--dataset",
                "current_internal",
                "--structure-source",
                "openfold",
                "--config",
                str(config),
                "--materialize-plan",
                "--force",
            ]
        )
        == 0
    )

    job_dir = tmp_path / "results/current_internal/foldx/openfold/mut_004"
    manifest = json.loads((job_dir / "job_manifest.json").read_text(encoding="utf-8"))
    assert manifest["job_status"] == "pending"
    assert manifest["stage_name"] == "foldx_openfold"
    assert manifest["stage_group"] == "post_gpu_cpu_heavy"


def test_preflight_rows_include_structure_source_and_stage_name(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)
    jobs = runner.select_jobs("current_internal", structure_source="openfold")
    foldx_config, _ = runner.load_config_for_mode(config, allow_example_fallback=False)

    rows, _ = runner.preflight_jobs("current_internal", jobs, foldx_config, force=False)

    assert rows[0]["method"] == "foldx"
    assert rows[0]["structure_source"] == "openfold"
    assert rows[0]["stage_name"] == "foldx_openfold"


def test_openfold_preflight_writes_openfold_report_name_and_single_none(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    assert runner.main(["--dataset", "current_internal", "--structure-source", "openfold", "--config", str(config), "--preflight"]) == 0

    report = tmp_path / "reports/current_internal/foldx_openfold_preflight.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert text.count("- none") == 1
    assert not (tmp_path / "reports/current_internal/foldx_experimental_preflight.md").exists()


def test_no_completion_marker_for_dry_run_preflight_or_materialize(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    assert runner.main(["--dataset", "current_internal", "--config", str(config), "--dry-run"]) == 0
    assert runner.main(["--dataset", "current_internal", "--config", str(config), "--preflight"]) == 0
    assert runner.main(["--dataset", "current_internal", "--config", str(config), "--materialize-plan", "--force"]) == 0

    assert not (tmp_path / "data/run_state/current_internal/cpu_heavy.foldx_experimental.completed.json").exists()
    assert not (tmp_path / "data/run_state/current_internal/cpu_heavy.completed.json").exists()


def test_real_run_with_fake_foldx_creates_parsed_result_and_stage_marker(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    fake_foldx = make_fake_foldx(tmp_path)
    config.write_text(
        f"""
foldx:
  binary_path: {fake_foldx}
  number_of_runs: 3
  timeout_seconds: 30
paths:
  structure_root: data/structures/chain/current_internal
  results_root: results/current_internal/foldx
  logs_root: logs/foldx_runs
""".lstrip(),
        encoding="utf-8",
    )
    patch_roots(monkeypatch, tmp_path)

    assert (
        runner.main(
            [
                "--dataset",
                "current_internal",
                "--config",
                str(config),
                "--run",
                "--workers",
                "1",
                "--max-jobs",
                "1",
            ]
        )
        == 0
    )

    job_dir = tmp_path / "results/current_internal/foldx/experimental/mut_001"
    parsed = json.loads((job_dir / "parsed_result.json").read_text(encoding="utf-8"))
    manifest = json.loads((job_dir / "job_manifest.json").read_text(encoding="utf-8"))
    assert parsed["parser_status"] == "success"
    assert parsed["foldx_ddg"] == -1.23
    assert manifest["job_status"] == "success"
    assert (tmp_path / "data/run_state/current_internal/cpu_heavy.foldx_experimental.completed.json").exists()
    assert not (tmp_path / "data/run_state/current_internal/cpu_heavy.completed.json").exists()


def test_existing_success_is_skipped_without_force(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    fake_foldx = make_fake_foldx(tmp_path)
    config.write_text(
        f"foldx:\n  binary_path: {fake_foldx}\n  number_of_runs: 3\n  timeout_seconds: 30\n",
        encoding="utf-8",
    )
    patch_roots(monkeypatch, tmp_path)

    runner.main(["--dataset", "current_internal", "--config", str(config), "--run", "--max-jobs", "1"])
    runner.main(["--dataset", "current_internal", "--config", str(config), "--run", "--max-jobs", "1"])

    manifest = json.loads((tmp_path / "results/current_internal/foldx/experimental/mut_001/job_manifest.json").read_text(encoding="utf-8"))
    assert manifest["job_status"] == "skipped"
    assert "successful parsed_result already exists" in manifest["notes"]
