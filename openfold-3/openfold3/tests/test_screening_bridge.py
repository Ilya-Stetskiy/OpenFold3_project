import json
import sys

from scripts.dev import summarize_screening_results

from openfold3.testbench.screening_bridge import load_screening_rows, summarize_screening_rows


def test_screening_bridge_load_and_summarize(tmp_path):
    results_jsonl = tmp_path / "results.jsonl"
    results_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "mutation_id": "WT",
                        "query_id": "screen_WT",
                        "query_hash": "hash-wt",
                        "sample_index": 1,
                        "seed": 1,
                        "sample_ranking_score": 0.8,
                        "iptm": 0.1,
                        "ptm": 0.2,
                        "avg_plddt": 80.0,
                        "gpde": 0.3,
                        "has_clash": 0.0,
                        "cache_hit": True,
                        "sequence_cache_hits": 1,
                        "query_result_cache_hit": False,
                        "cpu_prep_seconds": 1.0,
                        "gpu_inference_seconds": 2.0,
                        "total_seconds": 3.0,
                        "output_dir": "/tmp/wt",
                        "mutation_spec": None,
                        "aggregated_confidence_path": None,
                        "derived_interface_metrics": {},
                        "query_output_cleaned": False,
                    }
                ),
                json.dumps(
                    {
                        "mutation_id": "A_L1A",
                        "query_id": "screen_A_L1A",
                        "query_hash": "hash-a",
                        "sample_index": 1,
                        "seed": 1,
                        "sample_ranking_score": 1.2,
                        "iptm": 0.2,
                        "ptm": 0.3,
                        "avg_plddt": 90.0,
                        "gpde": 0.1,
                        "has_clash": 0.0,
                        "cache_hit": False,
                        "sequence_cache_hits": 0,
                        "query_result_cache_hit": True,
                        "cpu_prep_seconds": 2.0,
                        "gpu_inference_seconds": 4.0,
                        "total_seconds": 6.0,
                        "output_dir": "/tmp/mut",
                        "mutation_spec": {
                            "chain_id": "A",
                            "position_1based": 1,
                            "from_residue": "L",
                            "to_residue": "A",
                        },
                        "aggregated_confidence_path": None,
                        "derived_interface_metrics": {},
                        "query_output_cleaned": False,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_screening_rows(results_jsonl)
    summary = summarize_screening_rows(rows, top_k=1)

    assert len(rows) == 2
    assert summary.total_rows == 2
    assert summary.wt_rows == 1
    assert summary.mutated_rows == 1
    assert summary.query_result_cache_hits == 1
    assert summary.top_candidates_by_ranking_score[0]["mutation_id"] == "A_L1A"


def test_summarize_screening_results_cli(tmp_path, monkeypatch, capsys):
    results_jsonl = tmp_path / "results.jsonl"
    results_jsonl.write_text(
        json.dumps(
            {
                "mutation_id": "WT",
                "query_id": "screen_WT",
                "query_hash": "hash-wt",
                "sample_index": 1,
                "seed": 1,
                "sample_ranking_score": 0.8,
                "iptm": 0.1,
                "ptm": 0.2,
                "avg_plddt": 80.0,
                "gpde": 0.3,
                "has_clash": 0.0,
                "cache_hit": True,
                "sequence_cache_hits": 1,
                "query_result_cache_hit": False,
                "cpu_prep_seconds": 1.0,
                "gpu_inference_seconds": 2.0,
                "total_seconds": 3.0,
                "output_dir": "/tmp/wt",
                "mutation_spec": None,
                "aggregated_confidence_path": None,
                "derived_interface_metrics": {},
                "query_output_cleaned": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_screening_results.py",
            "--results-jsonl",
            str(results_jsonl),
            "--output",
            str(output_path),
        ],
    )

    summarize_screening_results.main()
    stdout = capsys.readouterr().out

    assert '"total_rows": 1' in stdout
    assert output_path.exists()
