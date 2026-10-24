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
runner = load_module("scripts/executors/run_openfold_wt.py", "run_openfold_wt")


def make_repo(root: Path) -> Path:
    (root / "configs/tools").mkdir(parents=True)
    (root / "data/manifests").mkdir(parents=True)
    (root / "openfold-3").mkdir(parents=True)
    (root / "data/manifests/current_internal_sequence_manifest.csv").write_text(
        "dataset_name,sequence_id,uniprot_id,protein_id,pdb_id,chain_id,sequence_source,sequence,sequence_length,fasta_path,n_mutations,quality_flags,notes\n"
        "current_internal,seq_1,U1,p1,1ABC,A,pdb_chain,ACDE,4,fasta,1,,\n"
        "current_internal,seq_2,U2,p2,2ABC,B,pdb_chain,FGHI,4,fasta,1,,\n"
        "other,seq_3,U3,p3,3ABC,C,pdb_chain,KLMN,4,fasta,1,,\n",
        encoding="utf-8",
    )
    config = root / "configs/tools/openfold.yaml"
    config.write_text(
        f"""
openfold:
  python_bin: {sys.executable}
  openfold_repo_dir: openfold-3
  runner_yaml: null
  inference_ckpt_path: null
  inference_ckpt_name: null
  num_diffusion_samples: 1
  num_model_seeds: 1
  use_msa_server: false
  use_templates: false
  timeout_seconds: 30
  batch_size: 2
paths:
  sequence_manifest: data/manifests/current_internal_sequence_manifest.csv
  sequence_fasta: data/sequences/current_internal/current_internal_openfold_inputs.fasta
  results_root: results/current_internal/openfold/wt
  logs_root: logs/openfold_runs
""".lstrip(),
        encoding="utf-8",
    )
    return config


def make_fake_openfold_python(root: Path) -> Path:
    fake = root / "fake_openfold_python.py"
    fake.write_text(
        r'''#!/usr/bin/env python3
import json
import sys
from pathlib import Path

if "-c" in sys.argv:
    print("openfold3")
    raise SystemExit(0)

query_json = Path(sys.argv[sys.argv.index("--query_json") + 1])
output_dir = Path(sys.argv[sys.argv.index("--output_dir") + 1])
payload = json.loads(query_json.read_text(encoding="utf-8"))
summary = output_dir / "summary.jsonl"
output_dir.mkdir(parents=True, exist_ok=True)
lines = []
for query_id in payload["queries"]:
    seed_dir = output_dir / query_id / "seed_1"
    seed_dir.mkdir(parents=True, exist_ok=True)
    confidence = seed_dir / f"{query_id}_seed_1_sample_1_confidences_aggregated.json"
    model = seed_dir / f"{query_id}_seed_1_sample_1_model.cif"
    confidence.write_text('{"mean_plddt": 88.0}\n', encoding="utf-8")
    model.write_text(f"""data_{query_id}
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_atom_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N ASP A 105 0.0 0.0 0.0 1.0 88.0 N ASP A 105 1
ATOM 2 CA C ASP A 105 1.0 0.0 0.0 1.0 88.0 CA ASP A 105 1
#
""", encoding="utf-8")
    lines.append(json.dumps({
        "query_id": query_id,
        "sample_index": 1,
        "sample_ranking_score": 0.88,
        "aggregated_confidence_path": str(confidence),
    }))
summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
''',
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | 0o111)
    return fake


def patch_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(execution_utils, "ROOT", tmp_path)


def test_select_jobs_reads_unique_current_internal_sequences(tmp_path, monkeypatch):
    config_path = make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)
    config, _ = runner.load_config_for_mode(config_path, allow_example_fallback=False)

    jobs = runner.select_jobs("current_internal", config)

    assert [job["sequence_id"] for job in jobs] == ["seq_1", "seq_2"]


def test_materialize_plan_creates_batch_and_per_sequence_manifests(tmp_path, monkeypatch):
    config_path = make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    assert runner.main(["--dataset", "current_internal", "--config", str(config_path), "--materialize-plan", "--force"]) == 0

    batch_query = tmp_path / "results/current_internal/openfold/wt/_batches/batch_0001/query.json"
    seq_manifest = tmp_path / "results/current_internal/openfold/wt/seq_1/job_manifest.json"
    stage_manifest = tmp_path / "results/current_internal/_manifests/openfold_wt_generation_jobs.csv"
    payload = json.loads(batch_query.read_text(encoding="utf-8"))
    manifest = json.loads(seq_manifest.read_text(encoding="utf-8"))

    assert list(payload["queries"]) == ["seq_1", "seq_2"]
    assert manifest["job_status"] == "pending"
    assert manifest["sequence_id"] == "seq_1"
    assert stage_manifest.exists()
    assert not (tmp_path / "data/run_state/current_internal/gpu_heavy.openfold_wt_generation.completed.json").exists()
    assert not (tmp_path / "results/current_internal/openfold/wt/seq_1/model.cif").exists()


def test_preflight_writes_report_without_completion_marker(tmp_path, monkeypatch):
    config_path = make_repo(tmp_path)
    patch_roots(monkeypatch, tmp_path)

    assert runner.main(["--dataset", "current_internal", "--config", str(config_path), "--preflight"]) == 0

    report = tmp_path / "reports/current_internal/openfold_wt_preflight.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["summary"]["total_selected_sequences"] == 2
    assert data["summary"]["ready_for_execution"] == 2
    assert not (tmp_path / "data/run_state/current_internal/gpu_heavy.openfold_wt_generation.completed.json").exists()


def test_smoke_run_with_fake_openfold_normalizes_outputs_and_creates_stage_marker(tmp_path, monkeypatch):
    config_path = make_repo(tmp_path)
    fake_python = make_fake_openfold_python(tmp_path)
    config_text = config_path.read_text(encoding="utf-8")
    config_path.write_text(config_text.replace(f"python_bin: {sys.executable}", f"python_bin: {fake_python}"), encoding="utf-8")
    patch_roots(monkeypatch, tmp_path)

    assert (
        runner.main(
            [
                "--dataset",
                "current_internal",
                "--config",
                str(config_path),
                "--smoke-run",
                "--max-sequences",
                "1",
                "--force",
            ]
        )
        == 0
    )

    result_dir = tmp_path / "results/current_internal/openfold/wt/seq_1"
    manifest = json.loads((result_dir / "job_manifest.json").read_text(encoding="utf-8"))
    result = json.loads((result_dir / "openfold_result.json").read_text(encoding="utf-8"))

    assert manifest["job_status"] == "success"
    assert result["parser_status"] == "success"
    assert (result_dir / "model.cif").exists()
    assert (result_dir / "model.pdb").exists()
    assert (result_dir / "confidence.json").exists()
    assert (tmp_path / "data/run_state/current_internal/gpu_heavy.openfold_wt_generation.completed.json").exists()
    assert not (tmp_path / "data/run_state/current_internal/gpu_heavy.completed.json").exists()
