from __future__ import annotations

import subprocess

from openfold3_runtime_benchmark.telemetry import (
    NvidiaSmiCollector,
    descendant_pids,
    parse_nvidia_smi_compute_apps_csv,
    parse_nvidia_smi_device_csv,
    parse_proc_io_text,
    parse_proc_stat_line,
    parse_proc_status_text,
)


def test_parse_proc_stat_line_extracts_core_fields() -> None:
    line = "1234 (python worker) S 4321 0 0 0 0 0 0 0 0 0 10 20 0 0 0 0 7 0 0 0"
    parsed = parse_proc_stat_line(line)

    assert parsed["pid"] == 1234
    assert parsed["comm"] == "python worker"
    assert parsed["state"] == "S"
    assert parsed["ppid"] == 4321
    assert parsed["utime_ticks"] == 10
    assert parsed["stime_ticks"] == 20
    assert parsed["num_threads"] == 7


def test_parse_proc_status_and_io_text() -> None:
    status = "VmRSS:\t2048 kB\nVmSize:\t4096 kB\nThreads:\t5\n"
    io_text = "read_bytes: 111\nwrite_bytes: 222\n"

    assert parse_proc_status_text(status) == {
        "rss_bytes": 2048 * 1024,
        "vms_bytes": 4096 * 1024,
        "threads": 5,
    }
    assert parse_proc_io_text(io_text) == {
        "read_bytes": 111,
        "write_bytes": 222,
    }


def test_parse_nvidia_smi_csv_helpers() -> None:
    devices = parse_nvidia_smi_device_csv(
        "0, NVIDIA RTX, 67, 8123, 16384, 550.54.14\n"
    )
    compute_apps = parse_nvidia_smi_compute_apps_csv("1001, 2048\n1002, 512\n")

    assert devices == [
        {
            "index": 0,
            "name": "NVIDIA RTX",
            "utilization_gpu_percent": 67.0,
            "memory_used_mb": 8123.0,
            "memory_total_mb": 16384.0,
            "driver_version": "550.54.14",
        }
    ]
    assert compute_apps == {
        1001: 2048.0,
        1002: 512.0,
    }


def test_descendant_pids_builds_tree() -> None:
    snapshots = {
        100: {"ppid": 1},
        101: {"ppid": 100},
        102: {"ppid": 101},
        103: {"ppid": 100},
        500: {"ppid": 1},
    }

    assert descendant_pids(100, snapshots) == [100, 101, 103, 102]


def test_nvidia_smi_collector_reports_missing_binary() -> None:
    collector = NvidiaSmiCollector(which=lambda _name: None)
    sample = collector.sample({123})

    assert sample["gpu_metrics_available"] is False
    assert "not found" in str(sample["gpu_error"])


def test_nvidia_smi_collector_parses_runtime_sample() -> None:
    def fake_runner(cmd, capture_output, text, check):
        joined = " ".join(cmd)
        if "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,driver_version" in joined:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="0, NVIDIA RTX, 75, 4096, 16384, 550.54.14\n",
                stderr="",
            )
        if "--query-compute-apps=pid,used_gpu_memory" in joined:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="123, 1024\n999, 256\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    collector = NvidiaSmiCollector(runner=fake_runner, which=lambda _name: "/usr/bin/nvidia-smi")
    sample = collector.sample({123})

    assert sample["gpu_metrics_available"] is True
    assert sample["gpu_device_count"] == 1
    assert sample["gpu_device_names"] == ["NVIDIA RTX"]
    assert sample["gpu_util_percent_max"] == 75.0
    assert sample["gpu_memory_used_total_mb"] == 4096.0
    assert sample["process_gpu_memory_mb"] == {123: 1024.0}
