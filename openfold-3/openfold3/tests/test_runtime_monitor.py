import csv
import sys
from pathlib import Path

from scripts.dev import benchmark_mutation_runner_cpu
from scripts.dev.runtime_monitor import RunMonitor, _read_stage_elapsed


def test_run_monitor_start_stop_writes_artifacts(tmp_path):
    monitor = RunMonitor(tmp_path, sample_interval_seconds=0.01)
    monitor.start()
    monitor.record_stage("custom_stage", "details")
    artifacts = monitor.stop()

    assert artifacts.resource_csv_path.exists()
    assert artifacts.stage_marks_path.exists()
    assert artifacts.monitor_plot_path.exists()
    assert artifacts.monitor_plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    with artifacts.resource_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "cpu_percent" in rows[0]

    stage_values = _read_stage_elapsed(artifacts.stage_marks_path)
    assert len(stage_values) >= 2


def test_read_stage_elapsed_skips_invalid_rows(tmp_path):
    stage_marks = tmp_path / "stage_marks.csv"
    stage_marks.write_text(
        "timestamp_utc,elapsed_seconds,stage,details\n"
        "2026-01-01T00:00:00Z,0.1,start,\n"
        "2026-01-01T00:00:01Z,not-a-float,bad,\n"
        "2026-01-01T00:00:02Z,0.3,done,\n",
        encoding="utf-8",
    )

    assert _read_stage_elapsed(stage_marks) == [0.1, 0.3]


def test_benchmark_main_rejects_nonpositive_repeats(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_mutation_runner_cpu.py",
            "--output-root",
            str(tmp_path / "bench"),
            "--repeats",
            "0",
        ],
    )

    try:
        benchmark_mutation_runner_cpu.main()
    except ValueError as exc:
        assert "--repeats must be at least 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for nonpositive repeats")


def test_benchmark_main_rejects_nonpositive_num_mutations(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_mutation_runner_cpu.py",
            "--output-root",
            str(tmp_path / "bench"),
            "--num-mutations",
            "0",
        ],
    )

    try:
        benchmark_mutation_runner_cpu.main()
    except ValueError as exc:
        assert "--num-mutations must be at least 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for nonpositive num-mutations")
