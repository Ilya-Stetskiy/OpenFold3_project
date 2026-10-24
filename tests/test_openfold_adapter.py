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


openfold_adapter = load_module("scripts/executors/openfold_adapter.py", "openfold_adapter")


def minimal_model_cif() -> str:
    return """data_seq_1
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
ATOM 1 N N ASP A 105 0.0 0.0 0.0 1.0 90.0 N ASP A 105 1
ATOM 2 CA C ASP A 105 1.0 0.0 0.0 1.0 90.0 CA ASP A 105 1
#
"""


def test_build_query_payload_uses_query_level_msa_flags():
    rows = [
        {
            "dataset_name": "current_internal",
            "sequence_id": "current_internal__O74035__1IO2_A",
            "pdb_id": "1IO2",
            "chain_id": "A",
            "sequence": "MKIAGIDEA",
        }
    ]

    payload = openfold_adapter.build_query_payload(rows, use_msas=False, use_templates=False)
    query = payload["queries"]["current_internal__O74035__1IO2_A"]

    assert query["use_msas"] is False
    assert query["use_main_msas"] is False
    assert query["use_paired_msas"] is False
    assert query["chains"][0]["molecule_type"] == "protein"
    assert "use_msas" not in query["chains"][0]


def test_batch_sequence_rows():
    rows = [{"sequence_id": f"s{i}"} for i in range(5)]

    batches = openfold_adapter.batch_sequence_rows(rows, 2)

    assert [[row["sequence_id"] for row in batch] for batch in batches] == [["s0", "s1"], ["s2", "s3"], ["s4"]]


def test_write_query_json(tmp_path):
    rows = [
        {
            "dataset_name": "current_internal",
            "sequence_id": "seq1",
            "pdb_id": "1ABC",
            "chain_id": "A",
            "sequence": "ACDE",
        }
    ]
    path = openfold_adapter.write_query_json(tmp_path / "query.json", rows, use_msas=True, use_templates=True)

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert list(payload["queries"]) == ["seq1"]
    assert payload["queries"]["seq1"]["chains"][0]["sequence"] == "ACDE"


def test_select_sequence_rows_deduplicates_and_filters():
    rows = [
        {"dataset_name": "current_internal", "sequence_id": "s1", "pdb_id": "1ABC"},
        {"dataset_name": "current_internal", "sequence_id": "s1", "pdb_id": "1ABC"},
        {"dataset_name": "other", "sequence_id": "s2", "pdb_id": "2ABC"},
    ]

    selected = openfold_adapter.select_sequence_rows(rows, dataset="current_internal")

    assert [row["sequence_id"] for row in selected] == ["s1"]


def test_normalize_openfold_outputs_selects_ranked_model_and_confidence(tmp_path):
    output_dir = tmp_path / "openfold_output"
    seed_dir = output_dir / "seq_1" / "seed_1"
    seed_dir.mkdir(parents=True)
    confidence = seed_dir / "seq_1_seed_1_sample_1_confidences_aggregated.json"
    model = seed_dir / "seq_1_seed_1_sample_1_model.cif"
    confidence.write_text('{"plddt": 90.0}\n', encoding="utf-8")
    model.write_text(minimal_model_cif(), encoding="utf-8")
    (output_dir / "summary.jsonl").write_text(
        json.dumps(
            {
                "query_id": "seq_1",
                "sample_index": 1,
                "sample_ranking_score": 0.95,
                "aggregated_confidence_path": str(confidence),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = openfold_adapter.normalize_openfold_outputs(output_dir, "seq_1", tmp_path / "target")

    assert result["parser_status"] == "success"
    assert "data_seq_1" in (tmp_path / "target/model.cif").read_text(encoding="utf-8")
    assert (tmp_path / "target/model.pdb").exists()
    assert json.loads((tmp_path / "target/confidence.json").read_text(encoding="utf-8"))["plddt"] == 90.0
    assert (tmp_path / "target/openfold_result.json").exists()


def test_convert_cif_to_pdb_writes_foldx_friendly_pdb(tmp_path):
    cif = tmp_path / "model.cif"
    pdb = tmp_path / "model.pdb"
    cif.write_text(minimal_model_cif(), encoding="utf-8")

    result = openfold_adapter.convert_cif_to_pdb(cif, pdb)

    text = pdb.read_text(encoding="utf-8")
    assert result["status"] == "success"
    assert result["atom_count"] == 2
    assert "ATOM" in text
    assert " ASP A 105" in text


def test_validate_cif_to_pdb_conversion_reports_success(tmp_path):
    cif = tmp_path / "model.cif"
    pdb = tmp_path / "model.pdb"
    cif.write_text(minimal_model_cif(), encoding="utf-8")

    result = openfold_adapter.validate_cif_to_pdb_conversion(cif, pdb)

    assert result["status"] == "success"
    assert result["atom_count"] == 2
    assert result["residue_count"] == 1
    assert result["error_message"] == ""


def test_validate_cif_to_pdb_conversion_reports_controlled_failure(tmp_path):
    cif = tmp_path / "bad.cif"
    pdb = tmp_path / "bad.pdb"
    cif.write_text("data_bad\n#\n", encoding="utf-8")

    result = openfold_adapter.validate_cif_to_pdb_conversion(cif, pdb)

    assert result["status"] == "failed"
    assert result["atom_count"] == 0
    assert result["residue_count"] == 0
    assert "No first-model protein atoms" in result["error_message"]
