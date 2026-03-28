from __future__ import annotations

from pathlib import Path

from of_notebook_lib.monitoring import RunMonitor, _read_stage_elapsed


def test_run_monitor_writes_artifacts(tmp_path: Path) -> None:
    monitor = RunMonitor(tmp_path, sample_interval_seconds=0.01)
    monitor.start()
    monitor.record_stage("midpoint", "demo")
    artifacts = monitor.stop()

    assert artifacts.resource_csv_path.exists()
    assert artifacts.stage_marks_path.exists()
    assert artifacts.monitor_plot_path.exists()
    assert artifacts.monitor_plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    resource_rows = artifacts.resource_csv_path.read_text(encoding="utf-8").splitlines()
    stage_rows = artifacts.stage_marks_path.read_text(encoding="utf-8").splitlines()

    assert len(resource_rows) >= 2
    assert len(stage_rows) >= 3


def test_read_stage_elapsed_skips_malformed_rows(tmp_path: Path) -> None:
    stage_marks = tmp_path / "stage_marks.csv"
    stage_marks.write_text(
        "timestamp_utc,elapsed_seconds,stage,details\n"
        "2026-03-29T00:00:00Z,0.100,start,\n"
        "2026-03-29T00:00:01Z,not-a-number,bad,\n"
        "2026-03-29T00:00:02Z,0.250,end,\n",
        encoding="utf-8",
    )

    assert _read_stage_elapsed(stage_marks) == [0.1, 0.25]
