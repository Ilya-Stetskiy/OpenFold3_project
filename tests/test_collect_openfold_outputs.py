import importlib.util
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "run"))


def load_module(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


collector = load_module("scripts/executors/collect_openfold_outputs.py", "collect_openfold_outputs")


def test_collect_sequence_output_extracts_mean_plddt(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "ROOT", tmp_path)
    out_dir = tmp_path / "results/current_internal/openfold/wt/seq_1"
    out_dir.mkdir(parents=True)
    (out_dir / "model.cif").write_text("data_seq_1\n", encoding="utf-8")
    (out_dir / "model.pdb").write_text("ATOM      1  CA  ASP A 105      0.000   0.000   0.000  1.00 88.00           C\n", encoding="utf-8")
    (out_dir / "confidence.json").write_text(json.dumps({"plddt": [80.0, 90.0, 100.0]}), encoding="utf-8")
    (out_dir / "openfold_result.json").write_text(json.dumps({"summary_row": {"sample_ranking_score": 0.7}}), encoding="utf-8")
    (out_dir / "job_manifest.json").write_text(json.dumps({"job_status": "success"}), encoding="utf-8")

    row = collector.collect_sequence_output("current_internal", {"sequence_id": "seq_1", "pdb_id": "1abc", "chain_id": "A"})

    assert row["output_status"] == "ready"
    assert row["mean_plddt"] == "90.0"
    assert row["plddt_source_field"] == "plddt"
    assert row["sample_ranking_score"] == "0.7"


def test_update_openfold_jobs_sets_ready_only_for_openfold_structure_jobs():
    sequences = [{"sequence_id": "seq_1", "pdb_id": "1ABC", "chain_id": "A"}]
    outputs = [{"sequence_id": "seq_1", "output_status": "ready", "model_path": "model.cif", "model_pdb_path": "model.pdb", "confidence_path": "confidence.json"}]
    jobs = [
        {"method": "foldx", "structure_source": "openfold", "pdb_id": "1ABC", "chain_id": "A", "input_ready": "false", "blocking_reason": "pending_openfold_structure"},
        {"method": "rosetta", "structure_source": "openfold", "pdb_id": "1ABC", "chain_id": "A", "input_ready": "false", "blocking_reason": "pending_openfold_structure"},
        {"method": "foldx", "structure_source": "experimental", "pdb_id": "1ABC", "chain_id": "A", "input_ready": "true", "blocking_reason": ""},
    ]

    updated = collector.update_openfold_jobs("current_internal", jobs, outputs, sequences)

    assert updated[0]["input_ready"] == "true"
    assert updated[0]["openfold_structure_path"] == "model.pdb"
    assert updated[1]["input_ready"] == "true"
    assert updated[2]["structure_source"] == "experimental"
    assert updated[2]["input_ready"] == "true"


def test_stage_group_matches_run_stages_config():
    config = yaml.safe_load((ROOT / "configs/run_stages.yaml").read_text(encoding="utf-8"))
    stage = next(stage for stage in config["stages"] if stage["stage_name"] == "collect_openfold_outputs")

    assert collector.STAGE_GROUP == stage["stage_group"] == "post_gpu_cpu_heavy"
