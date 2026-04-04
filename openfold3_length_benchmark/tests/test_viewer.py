from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from openfold3_length_benchmark.viewer import (
    _build_pdb_index,
    _render_reference_comparison_html,
)


def test_reference_browser_renders_bank_reference_and_prediction(
    monkeypatch,
    tmp_path,
):
    reference_path = tmp_path / "reference.cif"
    model_selected_path = tmp_path / "sample_selected_model.cif"
    model_oracle_path = tmp_path / "sample_oracle_model.cif"
    reference_path.write_text("data_reference\n#\n", encoding="utf-8")
    model_selected_path.write_text("data_selected\n#\n", encoding="utf-8")
    model_oracle_path.write_text("data_oracle\n#\n", encoding="utf-8")

    predict_run_dir = tmp_path / "predict_run"
    (predict_run_dir / "output").mkdir(parents=True)

    results_df = pd.DataFrame(
        [
            {
                "pdb_id": "1UBQ",
                "status": "ok",
                "predict_run_dir": str(predict_run_dir),
                "total_protein_length": 76,
                "reference_path": str(reference_path),
                "model_selected_sample": "sample_selected",
                "oracle_sample": "sample_oracle",
            }
        ]
    )
    sample_points_df = pd.DataFrame(
        [
            {
                "pdb_id": "1UBQ",
                "sample": "sample_selected",
                "rmsd_after_superposition": 1.23,
            },
            {
                "pdb_id": "1UBQ",
                "sample": "sample_oracle",
                "rmsd_after_superposition": 0.98,
            },
        ]
    )
    result = SimpleNamespace(results_df=results_df, sample_points_df=sample_points_df)

    monkeypatch.setattr(
        "openfold3_length_benchmark.viewer._sample_records_by_name",
        lambda _output_dir: {
            "sample_selected": SimpleNamespace(
                seed_name="seed_101",
                model_path=model_selected_path,
                sample_ranking_score=0.91,
                avg_plddt=87.4,
            ),
            "sample_oracle": SimpleNamespace(
                seed_name="seed_202",
                model_path=model_oracle_path,
                sample_ranking_score=0.77,
                avg_plddt=83.2,
            ),
        },
    )
    monkeypatch.setattr(
        "openfold3_length_benchmark.viewer._load_py3dmol",
        lambda: None,
    )

    entry = _build_pdb_index(result)["1UBQ"]
    html = _render_reference_comparison_html(
        entry,
        model_choice="oracle-best",
        width=480,
        height=320,
    )

    assert "Bank reference" in html
    assert "Chosen prediction" in html
    assert "Current view compares the chosen prediction to the bank reference." in html
    assert "RMSD vs bank reference: 0.98" in html
    assert str(reference_path) in html
    assert str(model_oracle_path) in html
