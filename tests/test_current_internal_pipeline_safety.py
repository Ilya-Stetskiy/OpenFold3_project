import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module("scripts/data/prepare_current_internal_inputs.py", "prepare_current_internal_inputs")
evaluation = load_module("scripts/evaluation/build_evaluation_tasks.py", "build_evaluation_tasks")


def make_paths(root: Path):
    return prepare.Paths(
        root=root,
        mutations_csv=root / "data/processed/current_internal/current_internal_mutations.csv",
        recalc_jobs_csv=root / "data/processed/current_internal/current_internal_recalc_jobs.csv",
        recalc_jobs_with_inputs_csv=root / "data/processed/current_internal/current_internal_recalc_jobs_with_inputs.csv",
        raw_structures_dir=root / "data/structures/raw/current_internal",
        clean_structures_dir=root / "data/structures/clean/current_internal",
        chain_structures_dir=root / "data/structures/chain/current_internal",
        sequences_dir=root / "data/sequences/current_internal",
        manifests_dir=root / "data/manifests",
        reports_dir=root / "reports/current_internal",
        structure_download_log_dir=root / "logs/structure_downloads/current_internal",
        structure_preparation_log_dir=root / "logs/structure_preparation/current_internal",
    )


def test_write_csv_refuses_existing_output_without_force(tmp_path):
    output = tmp_path / "out.csv"
    output.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        prepare.write_csv(output, [{"a": "new"}], ["a"], force=False)

    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_download_only_mode_has_no_protected_derived_outputs(tmp_path):
    paths = make_paths(tmp_path)

    outputs = prepare.output_paths_for_mode(paths, prepare_structures=False, prepare_sequences=False, full_pipeline=False)

    assert outputs == []


def test_existing_chain_pdb_is_revalidated_and_corrupt_file_fails(tmp_path):
    paths = make_paths(tmp_path)
    chain_dir = paths.chain_structures_dir
    chain_dir.mkdir(parents=True)
    (chain_dir / "1ABC_A.pdb").write_text("not a pdb\n", encoding="utf-8")

    result, residues = prepare.load_existing_chain("1ABC", "A", paths)

    assert residues == []
    assert result["status"] == "failed"
    assert result["error"].startswith("chain_file_validation_failed")


def test_confidence_blocking_distinguishes_missing_structure_and_missing_confidence(tmp_path):
    model_path = "results/current_internal/openfold/wt/seq/model.cif"
    confidence_path = "results/current_internal/openfold/wt/seq/confidence.json"

    assert evaluation.openfold_confidence_blocking_reason(tmp_path, model_path, confidence_path) == "pending_openfold_structure"

    model_file = tmp_path / model_path
    model_file.parent.mkdir(parents=True)
    model_file.write_text("dummy model\n", encoding="utf-8")
    assert evaluation.openfold_confidence_blocking_reason(tmp_path, model_path, confidence_path) == "pending_confidence_file"

    (tmp_path / confidence_path).write_text("{}\n", encoding="utf-8")
    assert evaluation.openfold_confidence_blocking_reason(tmp_path, model_path, confidence_path) == ""
