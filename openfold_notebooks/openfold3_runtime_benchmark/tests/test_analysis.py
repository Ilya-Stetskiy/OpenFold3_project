from __future__ import annotations

from openfold3_runtime_benchmark.analysis import (
    derive_events_from_log_line,
    event_rows_to_dataframe,
    sample_rows_to_dataframe,
    summarize_case_metrics,
)


def test_derive_events_from_structured_profile_line() -> None:
    line = 'INFO OF3_PROFILE_EVENT {"stage": "checkpoint_load", "event": "end", "duration_seconds": 12.5}'
    rows = derive_events_from_log_line(
        line,
        run_id="run_1",
        case_id="size_sweep_core:1UBQ",
        benchmark_mode="size_sweep_core",
        run_mode="cold",
        pid=123,
        timestamp=10.0,
        relative_seconds=3.0,
    )

    assert len(rows) == 1
    assert rows[0]["stage"] == "checkpoint_load"
    assert rows[0]["event"] == "end"
    assert rows[0]["duration_seconds"] == 12.5


def test_derive_events_from_legacy_predict_timing_line() -> None:
    line = "Predict timings for query_id(s) 1UBQ: batch_size=1 forward=8.25s confidence=1.50s"
    rows = derive_events_from_log_line(
        line,
        run_id="run_1",
        case_id="size_sweep_core:1UBQ",
        benchmark_mode="size_sweep_core",
        run_mode="cold",
        pid=123,
        timestamp=10.0,
        relative_seconds=3.0,
    )

    assert [row["stage"] for row in rows] == ["forward", "confidence"]
    assert rows[0]["duration_seconds"] == 8.25
    assert rows[1]["duration_seconds"] == 1.50


def test_summarize_case_metrics_aggregates_tree_and_stage_durations() -> None:
    samples_df = sample_rows_to_dataframe(
        [
            {
                "run_id": "run_1",
                "case_id": "size_sweep_core:1UBQ",
                "benchmark_mode": "size_sweep_core",
                "run_mode": "cold",
                "sample_seq": 1,
                "pid": 10,
                "timestamp": 1.0,
                "relative_seconds": 0.5,
                "process_count": 1,
                "child_count": 0,
                "tree_total_cpu_percent": 55.0,
                "tree_total_rss_bytes": 2 * 1024**3,
                "tree_total_vms_bytes": 3 * 1024**3,
                "tree_total_read_bytes": 100,
                "tree_total_write_bytes": 200,
                "gpu_metrics_available": True,
                "gpu_error": None,
                "gpu_device_count": 1,
                "gpu_device_names": ["RTX"],
                "gpu_util_percent_max": 72.0,
                "gpu_util_percent_mean": 70.0,
                "gpu_memory_used_total_mb": 8192.0,
                "gpu_memory_total_mb_total": 16384.0,
                "root_pid": 10,
                "parent_pid": 1,
                "is_root": True,
                "cmdline_label": "python",
                "cmdline": "python run_openfold",
                "state": "R",
                "cpu_percent": 55.0,
                "rss_bytes": 2 * 1024**3,
                "vms_bytes": 3 * 1024**3,
                "thread_count": 8,
                "read_bytes": 100,
                "write_bytes": 200,
                "process_gpu_memory_mb": 4096.0,
            },
            {
                "run_id": "run_1",
                "case_id": "size_sweep_core:1UBQ",
                "benchmark_mode": "size_sweep_core",
                "run_mode": "cold",
                "sample_seq": 2,
                "pid": 10,
                "timestamp": 2.0,
                "relative_seconds": 1.5,
                "process_count": 2,
                "child_count": 1,
                "tree_total_cpu_percent": 75.0,
                "tree_total_rss_bytes": 3 * 1024**3,
                "tree_total_vms_bytes": 4 * 1024**3,
                "tree_total_read_bytes": 300,
                "tree_total_write_bytes": 500,
                "gpu_metrics_available": True,
                "gpu_error": None,
                "gpu_device_count": 1,
                "gpu_device_names": ["RTX"],
                "gpu_util_percent_max": 88.0,
                "gpu_util_percent_mean": 80.0,
                "gpu_memory_used_total_mb": 10240.0,
                "gpu_memory_total_mb_total": 16384.0,
                "root_pid": 10,
                "parent_pid": 1,
                "is_root": True,
                "cmdline_label": "python",
                "cmdline": "python run_openfold",
                "state": "R",
                "cpu_percent": 75.0,
                "rss_bytes": 3 * 1024**3,
                "vms_bytes": 4 * 1024**3,
                "thread_count": 8,
                "read_bytes": 300,
                "write_bytes": 500,
                "process_gpu_memory_mb": 5120.0,
            },
        ]
    )
    events_df = event_rows_to_dataframe(
        [
            {"stage": "checkpoint_load", "event": "end", "duration_seconds": 12.0},
            {"stage": "predict_total", "event": "end", "duration_seconds": 18.0},
            {"stage": "forward", "event": "end", "duration_seconds": 8.0},
            {"stage": "confidence", "event": "end", "duration_seconds": 1.5},
        ]
    )

    summary = summarize_case_metrics(
        samples_df,
        events_df,
        wall_seconds=25.0,
    )

    assert summary["wall_seconds"] == 25.0
    assert summary["checkpoint_load_seconds"] == 12.0
    assert summary["predict_seconds"] == 18.0
    assert summary["forward_seconds"] == 8.0
    assert summary["confidence_seconds"] == 1.5
    assert summary["peak_rss_gb"] == 3.0
    assert summary["peak_cpu_percent"] == 75.0
    assert summary["peak_gpu_memory_gb"] == 10.0
    assert summary["mean_gpu_util_percent"] == 80.0
    assert summary["max_gpu_util_percent"] == 88.0
    assert summary["process_count_peak"] == 2
    assert summary["gpu_metrics_available"] is True


def test_event_rows_deduplicate_duplicate_profile_lines_and_keep_stage_duration() -> None:
    events_df = event_rows_to_dataframe(
        [
            {
                "run_id": "run_1",
                "case_id": "size_sweep_core:1UBQ",
                "benchmark_mode": "size_sweep_core",
                "run_mode": "cold",
                "stage": "forward",
                "event": "end",
                "pid": 123,
                "duration_seconds": 8.0,
                "relative_seconds": 10.0,
                "raw_line": 'WARNING:... OF3_PROFILE_EVENT {"stage":"forward","event":"end","duration_seconds":8.0}',
            },
            {
                "run_id": "run_1",
                "case_id": "size_sweep_core:1UBQ",
                "benchmark_mode": "size_sweep_core",
                "run_mode": "cold",
                "stage": "forward",
                "event": "end",
                "pid": 123,
                "duration_seconds": 8.0,
                "relative_seconds": 10.0,
                "raw_line": 'OF3_PROFILE_EVENT {"stage":"forward","event":"end","duration_seconds":8.0}',
            },
        ]
    )

    summary = summarize_case_metrics(
        sample_rows_to_dataframe([]),
        events_df,
        wall_seconds=12.0,
    )

    assert len(events_df) == 2
    assert summary["forward_seconds"] == 8.0
