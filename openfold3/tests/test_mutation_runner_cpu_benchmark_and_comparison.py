import json
import sys
from pathlib import Path

from scripts.dev import benchmark_mutation_runner_cpu, run_single_protein_comparison


def _write_query_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "queries": {
                    "mini_query": {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "ACDE",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_compare_metric_dicts_flags_tolerance_breach():
    baseline = {"avg_plddt": 1.0, "iptm": 0.5, "text": "ignored"}
    screening = {"avg_plddt": 1.3, "iptm": 0.55, "extra": 9}

    comparison = run_single_protein_comparison.compare_metric_dicts(
        baseline, screening, tolerance=0.1
    )

    assert comparison["numeric_keys"] == ["avg_plddt", "iptm"]
    assert abs(comparison["differences"]["avg_plddt"] - 0.3) < 1e-9
    assert comparison["within_tolerance"] is False


def test_find_first_aggregated_confidence_json_returns_sorted_first(tmp_path):
    first = tmp_path / "a" / "first_confidences_aggregated.json"
    second = tmp_path / "b" / "second_confidences_aggregated.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")

    found = run_single_protein_comparison.find_first_aggregated_confidence_json(tmp_path)

    assert found == first


def test_build_screening_job_includes_optional_checkpoint_fields(tmp_path):
    job = run_single_protein_comparison.build_screening_job(
        query_id="mini_query",
        query_payload={"chains": [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": "ACDE"}]},
        output_root=tmp_path,
        runner_yaml=tmp_path / "runner.yml",
        num_diffusion_samples=4,
        num_model_seeds=1,
        use_msa_server=True,
        use_templates=False,
        inference_ckpt_path="checkpoint.pt",
        inference_ckpt_name="ckpt-name",
    )

    assert job["include_wt"] is True
    assert job["num_diffusion_samples"] == 4
    assert job["inference_ckpt_path"] == "checkpoint.pt"
    assert job["inference_ckpt_name"] == "ckpt-name"


def test_run_single_protein_comparison_main_writes_summary(tmp_path, monkeypatch):
    query_json = tmp_path / "query.json"
    runner_yaml = tmp_path / "runner.yml"
    output_root = tmp_path / "output"
    query_json.write_text(
        json.dumps(
            {
                "queries": {
                    "mini_query": {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "ACDE",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    runner_yaml.write_text("{}", encoding="utf-8")

    screening_dir = output_root / "screening"
    screening_dir.mkdir(parents=True)
    (screening_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "mutation_id": "WT",
                    "query_id": "mini_query_WT",
                    "sample_ranking_score": 0.94,
                    "iptm": 0.69,
                    "ptm": 0.79,
                    "avg_plddt": 89.55,
                    "gpde": 0.15,
                    "has_clash": 0.0,
                    "total_seconds": 2.0,
                    "cpu_prep_seconds": 0.5,
                "gpu_inference_seconds": 1.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_agg = output_root / "baseline_predict" / "mini_query" / "seed_1"
    baseline_agg.mkdir(parents=True)
    baseline_path = baseline_agg / "mini_query_seed_1_sample_1_confidences_aggregated.json"
    baseline_path.write_text(
        json.dumps(
            {
                "sample_ranking_score": 0.9,
                "iptm": 0.65,
                "ptm": 0.75,
                "avg_plddt": 89.5,
                "gpde": 0.2,
                "has_clash": 0.0,
            }
        ),
        encoding="utf-8",
    )

    recorded_commands = []

    def fake_run_cmd(cmd, cwd):
        recorded_commands.append((cmd, cwd))
        return 3.0 if "predict" in cmd else 2.5

    monkeypatch.setattr(run_single_protein_comparison, "run_cmd", fake_run_cmd)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_single_protein_comparison.py",
            "--query-json",
            str(query_json),
            "--runner-yaml",
            str(runner_yaml),
            "--output-root",
            str(output_root),
            "--query-id",
            "mini_query",
            "--num-diffusion-samples",
            "4",
            "--num-model-seeds",
            "1",
        ],
    )

    run_single_protein_comparison.main()
    summary = json.loads((output_root / "comparison_summary.json").read_text(encoding="utf-8"))

    assert len(recorded_commands) == 2
    assert summary["query_id"] == "mini_query"
    assert summary["comparison"]["within_tolerance"] is True
    assert summary["timing"]["baseline_elapsed_seconds"] == 3.0
    assert summary["timing"]["screening_elapsed_seconds"] == 2.5


def test_benchmark_build_job_creates_expected_number_of_mutations(tmp_path):
    job = benchmark_mutation_runner_cpu.build_job(tmp_path, num_mutations=6, include_wt=True)

    assert len(job.mutations) == 6
    assert job.include_wt is True
    assert job.base_query.chains[1].main_msa_file_paths is not None


def test_benchmark_summarize_returns_expected_keys():
    summary = benchmark_mutation_runner_cpu.summarize("wall_seconds", [1.0, 2.0, 3.0])

    assert summary == {
        "wall_seconds_min": 1.0,
        "wall_seconds_median": 2.0,
        "wall_seconds_mean": 2.0,
        "wall_seconds_max": 3.0,
    }


def test_benchmark_main_writes_summary_json(tmp_path, monkeypatch):
    output_root = tmp_path / "bench"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_mutation_runner_cpu.py",
            "--output-root",
            str(output_root),
            "--repeats",
            "2",
            "--num-mutations",
            "4",
            "--include-wt",
        ],
    )

    benchmark_mutation_runner_cpu.main()
    summary = json.loads((output_root / "benchmark_summary.json").read_text(encoding="utf-8"))

    assert summary["repeats"] == 2
    assert summary["num_mutations"] == 4
    assert summary["include_wt"] is True
    assert len(summary["warm_cached_rows_per_run"]) == 2
    assert "cold_wall_seconds_median" in summary
    assert "warm_cpu_prep_seconds_median" in summary
