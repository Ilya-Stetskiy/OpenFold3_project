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
runner = load_module("scripts/executors/run_esm2_gpu.py", "run_esm2_gpu")


def make_repo(root: Path) -> Path:
    (root / "data/processed/current_internal").mkdir(parents=True)
    (root / "data/sequences/current_internal").mkdir(parents=True)
    (root / "configs/tools").mkdir(parents=True)
    (root / "data/sequences/current_internal/current_internal_esm2_inputs.fasta").write_text(
        ">current_internal__O74035__1IO2_A\nMKIAGIDEA\n",
        encoding="utf-8",
    )
    (root / "data/processed/current_internal/current_internal_recalc_jobs_resolved.csv").write_text(
        "dataset_name,job_id,mutation_record_id,protein_id,uniprot_id,pdb_id,chain_id,mutation,experimental_ddg,method,structure_source,input_ready,sequence_input_fasta_path\n"
        "current_internal,job_esm2_ready,mut_001,p,O74035,1IO2,A,I3V,-1.0,esm2,sequence_only,true,data/sequences/current_internal/current_internal_esm2_inputs.fasta\n"
        "current_internal,job_esm2_blocked,mut_002,p,O74035,1IO2,A,I4V,-1.0,esm2,sequence_only,false,data/sequences/current_internal/current_internal_esm2_inputs.fasta\n"
        "current_internal,job_foldx_ready,mut_003,p,O74035,1IO2,A,I5V,-1.0,foldx,experimental,true,data/sequences/current_internal/current_internal_esm2_inputs.fasta\n",
        encoding="utf-8",
    )
    config = root / "configs/tools/esm2.yaml"
    config.write_text(
        f"""
esm2:
  model_name: fake/esm2
  python_bin: {sys.executable}
  device: cpu
  scoring_mode: masked_marginal
  timeout_seconds: 30
paths:
  sequence_root: data/sequences/current_internal
  results_root: results/current_internal/esm2
  logs_root: logs/esm2_runs
""".lstrip(),
        encoding="utf-8",
    )
    return config


def make_fake_scorer(root: Path) -> Path:
    scorer = root / "fake_esm2_scorer.py"
    scorer.write_text(
        """#!/usr/bin/env python3
import argparse, json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--input-json", required=True)
parser.add_argument("--output-json", required=True)
parser.add_argument("--model-name")
parser.add_argument("--device")
parser.add_argument("--scoring-mode")
args = parser.parse_args()
payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
out = {
    "dataset_name": payload["dataset_name"],
    "method": "esm2",
    "structure_source": "sequence_only",
    "mutation_record_id": payload["mutation_record_id"],
    "job_id": payload["job_id"],
    "mutation": payload["mutation"],
    "chain_id": payload["chain_id"],
    "esm2_score": -0.42,
    "esm2_score_unit": "log_probability_difference",
    "parser_status": "success",
    "error_message": "",
}
Path(args.output_json).write_text(json.dumps(out), encoding="utf-8")
""",
        encoding="utf-8",
    )
    scorer.chmod(0o755)
    return scorer


def patch_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(execution_utils, "ROOT", tmp_path)


def test_dry_run_selects_only_ready_esm2_jobs(tmp_path, monkeypatch):
    make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    jobs = runner.select_jobs("current_internal")

    assert [job["job_id"] for job in jobs] == ["job_esm2_ready"]


def test_materialize_plan_creates_pending_manifest_and_job_input(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    assert runner.main(["--dataset", "current_internal", "--config", str(config), "--materialize-plan", "--force"]) == 0

    job_dir = tmp_path / "results/current_internal/esm2/sequence_only/mut_001"
    manifest = json.loads((job_dir / "job_manifest.json").read_text(encoding="utf-8"))
    job_input = json.loads((job_dir / "job_input.json").read_text(encoding="utf-8"))
    assert manifest["job_status"] == "pending"
    assert job_input["sequence_id"] == "current_internal__O74035__1IO2_A"
    assert not (job_dir / "parsed_result.json").exists()


def test_no_completion_marker_for_dry_run_preflight_or_materialize(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    assert runner.main(["--dataset", "current_internal", "--config", str(config), "--dry-run"]) == 0
    assert runner.main(["--dataset", "current_internal", "--config", str(config), "--preflight"]) == 0
    assert runner.main(["--dataset", "current_internal", "--config", str(config), "--materialize-plan", "--force"]) == 0

    assert not (tmp_path / "data/run_state/current_internal/gpu_heavy.esm2_gpu_optional.completed.json").exists()
    assert not (tmp_path / "data/run_state/current_internal/gpu_heavy.completed.json").exists()


def test_real_run_with_fake_scorer_creates_parsed_result_and_stage_marker(tmp_path, monkeypatch):
    config = make_repo(tmp_path)
    fake_scorer = make_fake_scorer(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    assert runner.main(
        [
            "--dataset",
            "current_internal",
            "--config",
            str(config),
            "--scorer-script",
            str(fake_scorer),
            "--run",
            "--workers",
            "1",
            "--max-jobs",
            "1",
        ]
    ) == 0

    job_dir = tmp_path / "results/current_internal/esm2/sequence_only/mut_001"
    parsed = json.loads((job_dir / "parsed_result.json").read_text(encoding="utf-8"))
    manifest = json.loads((job_dir / "job_manifest.json").read_text(encoding="utf-8"))
    assert parsed["parser_status"] == "success"
    assert parsed["esm2_score"] == -0.42
    assert manifest["job_status"] == "success"
    assert (tmp_path / "data/run_state/current_internal/gpu_heavy.esm2_gpu_optional.completed.json").exists()
    assert not (tmp_path / "data/run_state/current_internal/gpu_heavy.completed.json").exists()
