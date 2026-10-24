import importlib.util
import json
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


execution_utils = load_module("scripts/run/execution_utils.py", "execution_utils")
run_jobs_skeleton = load_module("scripts/run/run_jobs_skeleton.py", "run_jobs_skeleton")


def test_build_job_output_dir_creates_expected_path(tmp_path):
    path = execution_utils.build_job_output_dir("current_internal", "foldx", "experimental", "mut_001", root=tmp_path)

    assert path == tmp_path / "results/current_internal/foldx/experimental/mut_001"


def test_job_manifest_contains_required_fields():
    contract = execution_utils.load_yaml(ROOT / "configs/execution_contract.yaml")
    job = {
        "job_id": "job_1",
        "mutation_record_id": "mut_001",
        "method": "foldx",
        "structure_source": "experimental",
        "input_ready": "true",
    }

    manifest = execution_utils.build_job_manifest(
        dataset_name="current_internal",
        stage_group="cpu_heavy",
        stage_name="foldx_experimental",
        job=job,
    )

    for field in contract["required_job_manifest_fields"]:
        assert field in manifest


def test_filter_foldx_experimental_returns_only_ready_foldx_experimental_jobs():
    rows = [
        {"method": "foldx", "structure_source": "experimental", "input_ready": "true"},
        {"method": "foldx", "structure_source": "experimental", "input_ready": "false"},
        {"method": "rosetta", "structure_source": "experimental", "input_ready": "true"},
        {"method": "foldx", "structure_source": "openfold", "input_ready": "true"},
    ]

    filtered = execution_utils.filter_jobs_for_stage(rows, "foldx_experimental")

    assert filtered == [{"method": "foldx", "structure_source": "experimental", "input_ready": "true"}]


def make_minimal_repo(root: Path):
    (root / "configs").mkdir(parents=True)
    (root / "data/processed/current_internal").mkdir(parents=True)
    (root / "configs/run_stages.yaml").write_text(
        """
stages:
  - stage_name: foldx_experimental
    stage_group: cpu_heavy
""".lstrip(),
        encoding="utf-8",
    )
    (root / "configs/execution_contract.yaml").write_text(
        """
required_job_manifest_fields:
  - dataset_name
""".lstrip(),
        encoding="utf-8",
    )
    (root / "data/processed/current_internal/current_internal_recalc_jobs_resolved.csv").write_text(
        "dataset_name,job_id,mutation_record_id,method,structure_source,input_ready,experimental_structure_path,sequence_input_fasta_path\n"
        "current_internal,job_1,mut_001,foldx,experimental,true,data/structures/chain/current_internal/1ABC_A.pdb,\n",
        encoding="utf-8",
    )


def test_dry_run_skeleton_does_not_create_completion_marker(tmp_path, monkeypatch):
    make_minimal_repo(tmp_path)
    monkeypatch.setattr(run_jobs_skeleton, "ROOT", tmp_path)
    monkeypatch.setattr(execution_utils, "ROOT", tmp_path)

    assert run_jobs_skeleton.main(["--dataset", "current_internal", "--stage-name", "foldx_experimental", "--dry-run"]) == 0

    assert not (tmp_path / "data/run_state/current_internal/cpu_heavy.completed.json").exists()


def test_materialize_plan_creates_pending_manifest_without_parsed_result(tmp_path, monkeypatch):
    make_minimal_repo(tmp_path)
    monkeypatch.setattr(run_jobs_skeleton, "ROOT", tmp_path)
    monkeypatch.setattr(execution_utils, "ROOT", tmp_path)

    assert (
        run_jobs_skeleton.main(
            [
                "--dataset",
                "current_internal",
                "--stage-name",
                "foldx_experimental",
                "--dry-run",
                "--materialize-plan",
            ]
        )
        == 0
    )

    job_dir = tmp_path / "results/current_internal/foldx/experimental/mut_001"
    manifest = json.loads((job_dir / "job_manifest.json").read_text(encoding="utf-8"))
    assert manifest["job_status"] == "pending"
    assert (job_dir / "command.txt").exists()
    assert not (job_dir / "parsed_result.json").exists()

