import pandas as pd

from pipeline.s669_analysis import build_s669_prediction_table, metric_table


def test_build_s669_prediction_table_uses_imported_direct_and_inverse_cases():
    processed = pd.DataFrame(
        [
            {
                "benchmark": "S669",
                "direction": "direct",
                "protein_id": "1abcA_direct",
                "pdb_id": "1ABC",
                "chain": "A",
                "position": 7,
                "pdb_residue_id": "11",
                "mutation_id": "S7A",
                "experimental_ddg": -1.0,
            },
            {
                "benchmark": "S669",
                "direction": "inverse",
                "protein_id": "1abcA_inverse",
                "pdb_id": "1ABC",
                "chain": "A",
                "position": 7,
                "pdb_residue_id": "7",
                "mutation_id": "A7S",
                "experimental_ddg": 1.0,
            },
        ]
    )
    raw = pd.DataFrame(
        [
            {
                "Protein": "1abcA",
                "Seq_Mut": "S7A",
                "FoldX_dir": -0.8,
                "FoldX_inv": 0.9,
                "ACDC-NN-Seq_dir": -0.7,
                "ACDC-NN-Seq_inv": 0.8,
                "Experimental_DDG_dir": -1.0,
                "Experimental_DDG_inv": 1.0,
            }
        ]
    )

    table = build_s669_prediction_table(processed, raw)

    assert len(table) == 4
    assert set(table["method"]) == {"FoldX", "ACDC-NN-Seq"}
    assert set(table["representation"]) == {"structure", "sequence"}
    assert table.loc[
        (table["method"] == "FoldX") & (table["direction"] == "inverse"),
        "ddg",
    ].item() == 0.9


def test_metric_table_computes_sign_accuracy():
    predictions = pd.DataFrame(
        [
            {"method": "m", "representation": "structure", "ddg": -1.0, "experimental_ddg": -2.0},
            {"method": "m", "representation": "structure", "ddg": 1.0, "experimental_ddg": 2.0},
            {"method": "m", "representation": "structure", "ddg": -1.0, "experimental_ddg": 2.0},
        ]
    )

    metrics = metric_table(predictions)

    assert metrics.loc[0, "n"] == 3
    assert metrics.loc[0, "sign_accuracy"] == 2 / 3
