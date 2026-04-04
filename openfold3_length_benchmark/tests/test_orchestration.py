from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from openfold3_length_benchmark.composition import compositions_to_dataframe, extract_entry_composition
from openfold3_length_benchmark.interop import RuntimeConfig
from openfold3_length_benchmark.models import EntryComposition
from openfold3_length_benchmark.orchestration import run_length_benchmark


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2].parent
    / "openfold-3"
    / "openfold3"
    / "tests"
    / "test_data"
    / "mmcifs"
)


def test_run_length_benchmark_writes_outputs_and_keeps_partial_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    success_composition = extract_entry_composition(FIXTURE_ROOT / "2crb.cif")
    failed_composition = EntryComposition.failed("4ZEY", "Synthetic composition failure")
    preview_df = compositions_to_dataframe([success_composition, failed_composition])

    monkeypatch.setattr(
        "openfold3_length_benchmark.orchestration.collect_entry_compositions",
        lambda *args, **kwargs: [success_composition, failed_composition],
    )
    monkeypatch.setattr(
        "openfold3_length_benchmark.orchestration.preview_entries",
        lambda *args, **kwargs: preview_df,
    )

    def fake_run_prediction(runtime, payload, experiment_name, **kwargs):
        run_dir = tmp_path / "openfold_runs" / experiment_name
        output_dir = run_dir / "output"
        summary_dir = run_dir / "summary"
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)
        query_path = run_dir / "query.json"
        query_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(
            query_path=query_path,
            run_dir=run_dir,
            summary_dir=summary_dir,
            output_dir=output_dir,
        )

    def fake_run_rmsd_benchmark(*, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        row_a = {
            "query": "2CRB",
            "sample": "sample_1",
            "rmsd_before_superposition": 2.8,
            "rmsd_after_superposition": 1.4,
            "coverage": {
                "matched_atom_count": 97,
                "pred_filtered_atom_count": 97,
                "ref_filtered_atom_count": 97,
            },
            "aggregated_confidence": {
                "avg_plddt": 88.0,
                "sample_ranking_score": 0.91,
            },
        }
        row_b = {
            "query": "2CRB",
            "sample": "sample_2",
            "rmsd_before_superposition": 2.1,
            "rmsd_after_superposition": 1.1,
            "coverage": {
                "matched_atom_count": 97,
                "pred_filtered_atom_count": 97,
                "ref_filtered_atom_count": 97,
            },
            "aggregated_confidence": {
                "avg_plddt": 82.0,
                "sample_ranking_score": 0.73,
            },
        }
        with (output_dir / "rmsd_rows.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(row_a) + "\n")
            handle.write(json.dumps(row_b) + "\n")
        (output_dir / "rmsd_summary.json").write_text("{}", encoding="utf-8")
        return output_dir

    monkeypatch.setattr(
        "openfold3_length_benchmark.orchestration.run_prediction",
        fake_run_prediction,
    )
    monkeypatch.setattr(
        "openfold3_length_benchmark.orchestration._run_rmsd_benchmark",
        fake_run_rmsd_benchmark,
    )

    runtime = RuntimeConfig(
        project_dir=tmp_path / "project",
        openfold_repo_dir=tmp_path / "repo",
        openfold_prefix=tmp_path / "prefix",
        results_dir=tmp_path / "unused_results",
        msa_cache_dir=tmp_path / "msa_cache",
        triton_cache_dir=tmp_path / "triton_cache",
        fixed_msa_tmp_dir=tmp_path / "fixed_msa_tmp",
    )

    result = run_length_benchmark(
        runtime=runtime,
        pdb_ids="2crb 4zey",
        output_root=tmp_path / "benchmark_runs",
    )

    assert result.results_csv_path.exists()
    assert result.results_json_path.exists()
    assert result.failures_csv_path.exists()
    assert result.summary_path.exists()
    assert result.plot_paths["scatter_svg"].exists()
    assert result.plot_paths["binned_svg"].exists()

    rows = result.results_df.set_index("pdb_id")
    assert rows.loc["2CRB", "status"] == "ok"
    assert rows.loc["2CRB", "model_selected_rmsd"] == 1.4
    assert rows.loc["2CRB", "oracle_best_rmsd"] == 1.1
    assert rows.loc["4ZEY", "status"] == "failed"
    assert "Synthetic composition failure" in rows.loc["4ZEY", "failure_reason"]

    submitted_query = Path(rows.loc["2CRB", "submitted_query_path"])
    assert submitted_query.exists()
    assert (result.run_root / "refs" / "2CRB.cif").exists()
