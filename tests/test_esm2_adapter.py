import importlib.util
import json
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


esm2_adapter = load_module("scripts/executors/esm2_adapter.py", "esm2_adapter")


def test_parse_single_site_mutation():
    assert esm2_adapter.parse_single_site_mutation("D105A") == ("D", 105, "A")
    assert esm2_adapter.parse_single_site_mutation("g30f") == ("G", 30, "F")


def test_invalid_mutation_raises():
    with pytest.raises(ValueError):
        esm2_adapter.parse_single_site_mutation("D105A/G30F")


def test_find_sequence_for_job_and_write_input(tmp_path):
    fasta = tmp_path / "inputs.fasta"
    fasta.write_text(">current_internal__O74035__1IO2_A\nMKIAGIDEA\n", encoding="utf-8")
    job = {
        "dataset_name": "current_internal",
        "job_id": "job_001",
        "mutation_record_id": "mut_001",
        "uniprot_id": "O74035",
        "pdb_id": "1IO2",
        "chain_id": "A",
        "mutation": "I3V",
    }

    sequence_id, sequence = esm2_adapter.find_sequence_for_job(job, fasta)
    input_json = esm2_adapter.write_job_input_json(tmp_path, job, fasta)
    payload = json.loads(input_json.read_text(encoding="utf-8"))

    assert sequence_id == "current_internal__O74035__1IO2_A"
    assert sequence == "MKIAGIDEA"
    assert payload["sequence_wt_matches_mutation"] is True


def test_parse_fake_esm2_result(tmp_path):
    (tmp_path / "parsed_result.json").write_text(
        json.dumps({"parser_status": "success", "esm2_score": -0.75, "esm2_score_unit": "log_probability_difference"}),
        encoding="utf-8",
    )

    parsed = esm2_adapter.parse_esm2_result(tmp_path)

    assert parsed["parser_status"] == "success"
    assert parsed["esm2_score"] == -0.75
