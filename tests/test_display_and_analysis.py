from __future__ import annotations

from of_notebook_lib import (
    format_mutation_ranking,
    format_sample_table,
    preview_molecules,
    summarize_best_result,
    validate_molecules,
)
from of_notebook_lib.analysis import (
    best_samples_by_metric,
    collect_samples,
    rank_mutations,
    samples_to_dataframe,
    summarize_mutation_batch,
)


def test_preview_and_validate_molecules(big_ace_molecules: list[dict]) -> None:
    preview_df = preview_molecules(big_ace_molecules)
    issues = validate_molecules(big_ace_molecules)

    assert len(preview_df) == 2
    assert issues == []
    assert set(preview_df.columns) == {
        "molecule_type",
        "chain_ids",
        "length",
        "sequence_preview",
    }


def test_collect_samples_and_sample_table(big_ace_output_dir) -> None:
    samples = collect_samples(big_ace_output_dir)
    sample_df = samples_to_dataframe(samples)
    formatted = format_sample_table(sample_df)
    winners = best_samples_by_metric(samples)
    summary = summarize_best_result(sample_df)

    assert len(samples) == 3
    assert not sample_df.empty
    assert not formatted.empty
    assert winners["sample_ranking_score"] is not None
    assert summary["best_query"] == "complex_AB"


def test_mutation_summary_and_ranking_from_real_samples(big_ace_output_dir) -> None:
    sample_df = samples_to_dataframe(collect_samples(big_ace_output_dir))
    mutation_like_df = sample_df.copy()
    mutation_like_df["query_name"] = [
        "scan_case__WT" if i % 2 == 0 else "scan_case__B_F4G"
        for i in range(len(mutation_like_df))
    ]
    mutation_like_df["mutation_label"] = (
        mutation_like_df["query_name"].str.split("__").str[1]
    )

    summary_df = summarize_mutation_batch(mutation_like_df)
    ranked_df = rank_mutations(summary_df, top_n=len(summary_df))
    formatted = format_mutation_ranking(ranked_df)

    assert not summary_df.empty
    assert not ranked_df.empty
    assert not formatted.empty
    assert "mutation_label" in formatted.columns
