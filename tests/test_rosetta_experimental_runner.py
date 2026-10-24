import importlib.util
import json
import sys
from pathlib import Path


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
runner = load_module("scripts/executors/run_rosetta_experimental.py", "run_rosetta_experimental")


def make_repo(root: Path) -> Path:
    (root / "data/processed/current_internal").mkdir(parents=True)
    (root / "data/structures/chain/current_internal").mkdir(parents=True)
    (root / "configs/tools").mkdir(parents=True)
    (root / "data/structures/chain/current_internal/1ABC_A.pdb").write_text(
        "ATOM      1  CA  ASP A 105      0.000   0.000   0.000  1.00 10.00           C\n",
        encoding="utf-8",
    )
    (root / "data/processed/current_internal/current_internal_recalc_jobs_resolved.csv").write_text(
        "dataset_name,job_id,mutation_record_id,method,structure_source,input_ready,chain_id,mutation,experimental_structure_path\n"
        "current_internal,job_rosetta_ready,mut_001,rosetta,experimental,true,A,D105A,data/structures/chain/current_internal/1ABC_A.pdb\n"
        "current_internal,job_rosetta_blocked,mut_002,rosetta,experimental,false,A,D106A,data/structures/chain/current_internal/1ABC_A.pdb\n"
        "current_internal,job_foldx_ready,mut_003,foldx,experimental,true,A,D107A,data/structures/chain/current_internal/1ABC_A.pdb\n"
        "current_internal,job_rosetta_openfold,mut_004,rosetta,openfold,true,A,D108A,data/structures/chain/current_internal/1ABC_A.pdb\n",
        encoding="utf-8",
    )
    config = root / "configs/tools/rosetta.yaml"
    config.write_text(
        """
rosetta:
  binary_path: /missing/rosetta
  number_of_runs: 3
  timeout_seconds: 600
  extra_args: []
paths:
  structure_root: data/structures/chain/current_internal
  results_root: results/current_internal/rosetta
  logs_root: logs/rosetta_runs
""".lstrip(),
        encoding="utf-8",
    )
    return config


def make_fake_rosetta(root: Path) -> Path:
    binary = root / "fake_rosetta.py"
    binary.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
Path("score.sc").write_text("SCORE: total_score ddg description\\nSCORE: -10.0 -1.23 model_1\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


def patch_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(execution_utils, "ROOT", tmp_path)


def test_dry_run_selects_only_ready_rosetta_experimental_jobs(tmp_path, monkeypatch):
    make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    jobs = runner.select_jobs("current_internal")

    assert [job["job_id"] for job in jobs] == ["job_rosetta_ready"]


def test_materialize_plan_creates_pending_manifest_and_mutation_file(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    assert runner.main(["--dataset", "current_internal", "--config", str(config), "--materialize-plan", "--force"]) == 0

    job_dir = tmp_path / "results/current_internal/rosetta/experimental/mut_001"
    manifest = json.loads((job_dir / "job_manifest.json").read_text(encoding="utf-8"))
    assert manifest["job_status"] == "pending"
    assert (job_dir / "mutation_file.txt").read_text(encoding="utf-8") == "total 1\n1\nD 105 A\n"
    assert not (job_dir / "parsed_result.json").exists()


def test_no_completion_marker_for_dry_run_preflight_or_materialize(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    assert runner.main(["--dataset", "current_internal", "--config", str(config), "--dry-run"]) == 0
    assert runner.main(["--dataset", "current_internal", "--config", str(config), "--preflight"]) == 0
    assert runner.main(["--dataset", "current_internal", "--config", str(config), "--materialize-plan", "--force"]) == 0

    assert not (tmp_path / "data/run_state/current_internal/cpu_heavy.rosetta_experimental.completed.json").exists()
    assert not (tmp_path / "data/run_state/current_internal/cpu_heavy.completed.json").exists()


def test_real_run_with_fake_rosetta_creates_parsed_result_and_stage_marker(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    fake_rosetta = make_fake_rosetta(tmp_path)
    config.write_text(
        f"""
rosetta:
  binary_path: {fake_rosetta}
  number_of_runs: 3
  timeout_seconds: 30
  extra_args: []
""".lstrip(),
        encoding="utf-8",
    )
    patch_roots(monkeypatch, tmp_path)

    assert runner.main(["--dataset", "current_internal", "--config", str(config), "--run", "--workers", "1", "--max-jobs", "1"]) == 0

    job_dir = tmp_path / "results/current_internal/rosetta/experimental/mut_001"
    parsed = json.loads((job_dir / "parsed_result.json").read_text(encoding="utf-8"))
    manifest = json.loads((job_dir / "job_manifest.json").read_text(encoding="utf-8"))
    assert parsed["parser_status"] == "success"
    assert parsed["rosetta_ddg"] == -1.23
    assert manifest["job_status"] == "success"
    assert (tmp_path / "data/run_state/current_internal/cpu_heavy.rosetta_experimental.completed.json").exists()
    assert not (tmp_path / "data/run_state/current_internal/cpu_heavy.completed.json").exists()

