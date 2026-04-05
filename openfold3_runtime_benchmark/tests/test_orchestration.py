from __future__ import annotations

import json
from pathlib import Path

from openfold3_length_benchmark.composition import compositions_to_dataframe, extract_entry_composition
from openfold3_length_benchmark.models import EntryComposition
from openfold3_runtime_benchmark.interop import RuntimeConfig
from openfold3_runtime_benchmark.orchestration import run_runtime_benchmark


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2].parent
    / "openfold-3"
    / "openfold3"
    / "tests"
    / "test_data"
    / "mmcifs"
)


def test_run_runtime_benchmark_writes_outputs_and_cpu_only_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    success_composition = extract_entry_composition(FIXTURE_ROOT / "2crb.cif")
    failed_composition = EntryComposition.failed("4ZEY", "Synthetic runtime failure")
    preview_df = compositions_to_dataframe([success_composition, failed_composition])

    monkeypatch.setattr(
        "openfold3_runtime_benchmark.orchestration.collect_entry_compositions",
        lambda *args, **kwargs: [success_composition, failed_composition],
    )
    monkeypatch.setattr(
        "openfold3_runtime_benchmark.orchestration.compositions_to_dataframe",
        lambda *args, **kwargs: preview_df.copy(),
    )

    def fake_run_case(
        *,
        run_id,
        run_root,
        runtime,
        composition,
        benchmark_mode,
        run_mode,
        sampling_interval_seconds,
        runner_yaml,
        use_templates,
        use_msa_server,
    ):
        mode_root = run_root / "cases" / f"{benchmark_mode}__{composition.pdb_id}" / "runs" / run_mode
        output_dir = mode_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        query_path = mode_root / "query.json"
        query_path.write_text("{}", encoding="utf-8")
        log_path = mode_root / "run_openfold.log"
        log_path.write_text("synthetic log\n", encoding="utf-8")
        timeline_path = mode_root / "timeline.svg"
        timeline_path.write_text("<svg></svg>", encoding="utf-8")
        summary_path = mode_root / "case_summary.md"
        summary_path.write_text("# summary\n", encoding="utf-8")

        row = {
            "case_id": f"{benchmark_mode}:{composition.pdb_id}",
            "benchmark_mode": benchmark_mode,
            "run_mode": run_mode,
            "status": "ok",
            "failure_reason": None,
            "pdb_id": composition.pdb_id,
            "chain_group": "single_chain",
            "total_protein_length": composition.total_protein_length,
            "wall_seconds": 20.0 if run_mode == "cold" else 12.0,
            "checkpoint_load_seconds": 8.0 if run_mode == "cold" else 2.0,
            "predict_seconds": 10.0 if run_mode == "cold" else 8.5,
            "forward_seconds": 7.0,
            "confidence_seconds": 1.0,
            "peak_rss_gb": 3.2,
            "peak_cpu_percent": 180.0,
            "peak_gpu_memory_gb": None,
            "mean_gpu_util_percent": None,
            "max_gpu_util_percent": None,
            "process_count_peak": 2,
            "gpu_metrics_available": False,
            "gpu_error": "GPU unavailable in synthetic test",
            "local_cache_state": "local_cold" if run_mode == "cold" else "warm_reuse",
            "reference_path": str(success_composition.source_path),
            "query_path": str(query_path),
            "run_dir": str(mode_root),
            "output_dir": str(output_dir),
            "log_path": str(log_path),
            "timeline_svg_path": str(timeline_path),
            "case_summary_path": str(summary_path),
            "gpu_probe_available": False,
        }
        event_rows = [
            {
                "run_id": run_id,
                "case_id": row["case_id"],
                "benchmark_mode": benchmark_mode,
                "run_mode": run_mode,
                "stage": "process",
                "event": "exit",
                "timestamp": 1.0,
                "relative_seconds": row["wall_seconds"],
                "pid": 123,
            }
        ]
        sample_rows = [
            {
                "run_id": run_id,
                "case_id": row["case_id"],
                "benchmark_mode": benchmark_mode,
                "run_mode": run_mode,
                "sample_seq": 1,
                "timestamp": 1.0,
                "relative_seconds": 1.0,
                "root_pid": 123,
                "process_count": 2,
                "child_count": 1,
                "tree_total_cpu_percent": 180.0,
                "tree_total_rss_bytes": int(3.2 * 1024**3),
                "tree_total_vms_bytes": int(4.5 * 1024**3),
                "tree_total_read_bytes": 100,
                "tree_total_write_bytes": 200,
                "gpu_metrics_available": False,
                "gpu_error": "GPU unavailable in synthetic test",
                "gpu_device_count": 0,
                "gpu_device_names": [],
                "gpu_util_percent_max": None,
                "gpu_util_percent_mean": None,
                "gpu_memory_used_total_mb": None,
                "gpu_memory_total_mb_total": None,
                "pid": 123,
                "parent_pid": 1,
                "is_root": True,
                "cmdline_label": "python",
                "cmdline": "python run_openfold",
                "state": "R",
                "cpu_percent": 180.0,
                "rss_bytes": int(3.2 * 1024**3),
                "vms_bytes": int(4.5 * 1024**3),
                "thread_count": 6,
                "read_bytes": 100,
                "write_bytes": 200,
                "process_gpu_memory_mb": None,
            }
        ]
        return row, event_rows, sample_rows

    monkeypatch.setattr(
        "openfold3_runtime_benchmark.orchestration._run_case",
        fake_run_case,
    )

    runtime = RuntimeConfig(
        project_dir=tmp_path / "project",
        openfold_repo_dir=tmp_path / "repo",
        openfold_prefix=tmp_path / "prefix",
        results_dir=tmp_path / "results",
        msa_cache_dir=tmp_path / "msa_cache",
        triton_cache_dir=tmp_path / "triton",
        fixed_msa_tmp_dir=tmp_path / "fixed_msa",
    )

    result = run_runtime_benchmark(
        runtime=runtime,
        pdb_ids="2crb 4zey",
        output_root=tmp_path / "runtime_runs",
        pipeline_trace_pdb_ids="2crb",
    )

    assert result.manifest_path.exists()
    assert result.summary_path.exists()
    assert result.case_results_csv_path.exists()
    assert result.case_results_json_path.exists()
    assert result.events_path.exists()
    assert result.samples_path.exists()
    assert result.plot_paths["length_vs_wall_seconds_svg"].exists()
    assert result.plot_paths["length_vs_peak_rss_gb_svg"].exists()

    rows = result.case_results_df.set_index(["benchmark_mode", "pdb_id", "run_mode"])
    assert rows.loc[("size_sweep_core", "2CRB", "cold"), "status"] == "ok"
    assert rows.loc[("size_sweep_core", "2CRB", "warm"), "wall_seconds"] == 12.0
    assert rows.loc[("size_sweep_core", "2CRB", "cold"), "gpu_metrics_available"] == False
    assert rows.loc[("pipeline_trace", "2CRB", "cold"), "status"] == "ok"
    assert rows.loc[("size_sweep_core", "4ZEY", "cold"), "status"] == "failed"
    assert "Synthetic runtime failure" in rows.loc[("size_sweep_core", "4ZEY", "cold"), "failure_reason"]

    records = json.loads(result.case_results_json_path.read_text(encoding="utf-8"))
    assert records["summary"]["n_successful_case_runs"] == 4
    assert records["summary"]["n_failed_case_runs"] == 2

    events = [json.loads(line) for line in result.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = [json.loads(line) for line in result.samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(events) == 4
    assert len(samples) == 4
