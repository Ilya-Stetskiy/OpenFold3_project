from __future__ import annotations

import pandas as pd

from of_notebook_lib import (
    format_mutation_ranking,
    format_sample_table,
    preview_molecules,
    summarize_best_result,
    validate_molecules,
)


def test_preview_molecules_handles_short_and_missing_sequence() -> None:
    molecules = [
        {"type": "protein", "id": "A", "sequence": "ABCDE"},
        {"molecule_type": "ligand", "chain_ids": ["L1"]},
    ]

    preview = preview_molecules(molecules)

    assert len(preview) == 2
    assert preview.iloc[0]["sequence_preview"] == "ABCDE"
    assert pd.isna(preview.iloc[1]["length"])


def test_validate_molecules_reports_multiple_input_issues() -> None:
    issues = validate_molecules(
        [
            {"chain_ids": ["A"]},
            {"molecule_type": "protein", "chain_ids": ["A"]},
            {"molecule_type": "protein"},
        ]
    )

    assert "Molecule #1 is missing molecule_type." in issues
    assert "Chain id 'A' is duplicated." in issues
    assert "Molecule #2 requires a sequence." in issues
    assert "Molecule #3 is missing chain_ids." in issues
    assert "Molecule #3 requires a sequence." in issues


def test_validate_molecules_handles_empty_list() -> None:
    assert validate_molecules([]) == ["No molecules were provided."]


def test_format_helpers_return_empty_copy_for_empty_input() -> None:
    empty = pd.DataFrame()
    assert format_sample_table(empty).empty
    assert format_mutation_ranking(empty).empty
    assert summarize_best_result(empty) == {"status": "No samples found."}


def test_summarize_best_result_covers_other_thresholds() -> None:
    weak = pd.DataFrame(
        [{"sample_name": "s1", "query_name": "q1", "sample_ranking_score": 1.0, "iptm": 0.4, "avg_plddt": 60.0}]
    )
    medium = pd.DataFrame(
        [{"sample_name": "s2", "query_name": "q2", "sample_ranking_score": 1.0, "iptm": 0.6, "avg_plddt": 75.0}]
    )
    missing = pd.DataFrame(
        [{"sample_name": "s3", "query_name": "q3", "sample_ranking_score": 1.0, "iptm": None, "avg_plddt": None}]
    )

    weak_summary = summarize_best_result(weak)
    medium_summary = summarize_best_result(medium)
    missing_summary = summarize_best_result(missing)

    assert weak_summary["interface_note"] == "interface signal is weak"
    assert weak_summary["confidence_note"] == "local confidence is limited"
    assert medium_summary["interface_note"] == "interface may be plausible"
    assert medium_summary["confidence_note"] == "local confidence is decent"
    assert missing_summary["interface_note"] == "interface score is missing"
    assert missing_summary["confidence_note"] == "local confidence is missing"
