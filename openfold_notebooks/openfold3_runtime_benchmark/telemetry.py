from __future__ import annotations

import os
import shutil
import subprocess
from collections import deque
from pathlib import Path
from typing import Any


PROC_ROOT = Path("/proc")
CLOCK_TICKS = os.sysconf(os.sysconf_names["SC_CLK_TCK"])


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    if not rows:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{_json_dumps(row)}\n")


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def parse_proc_stat_line(line: str) -> dict[str, Any]:
    stripped = line.strip()
    end_comm = stripped.rfind(")")
    if end_comm < 0:
        raise ValueError(f"Invalid /proc stat line: {line!r}")

    start_comm = stripped.find("(")
    pid = int(stripped[:start_comm].strip())
    comm = stripped[start_comm + 1 : end_comm]
    remainder = stripped[end_comm + 1 :].strip().split()
    if len(remainder) < 20:
        raise ValueError(f"Unexpected /proc stat field count: {line!r}")

    return {
        "pid": pid,
        "comm": comm,
        "state": remainder[0],
        "ppid": int(remainder[1]),
        "utime_ticks": int(remainder[11]),
        "stime_ticks": int(remainder[12]),
        "num_threads": int(remainder[17]),
    }


def parse_proc_status_text(text: str) -> dict[str, int]:
    values: dict[str, int] = {
        "rss_bytes": 0,
        "vms_bytes": 0,
        "threads": 0,
    }
    for raw_line in text.splitlines():
        if raw_line.startswith("VmRSS:"):
            values["rss_bytes"] = int(raw_line.split()[1]) * 1024
        elif raw_line.startswith("VmSize:"):
            values["vms_bytes"] = int(raw_line.split()[1]) * 1024
        elif raw_line.startswith("Threads:"):
            values["threads"] = int(raw_line.split()[1])
    return values


def parse_proc_io_text(text: str) -> dict[str, int]:
    values = {
        "read_bytes": 0,
        "write_bytes": 0,
    }
    for raw_line in text.splitlines():
        if raw_line.startswith("read_bytes:"):
            values["read_bytes"] = int(raw_line.split()[1])
        elif raw_line.startswith("write_bytes:"):
            values["write_bytes"] = int(raw_line.split()[1])
    return values


def parse_nvidia_smi_device_csv(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "utilization_gpu_percent": float(parts[2]),
                "memory_used_mb": float(parts[3]),
                "memory_total_mb": float(parts[4]),
                "driver_version": parts[5],
            }
        )
    return rows


def parse_nvidia_smi_compute_apps_csv(stdout: str) -> dict[int, float]:
    rows: dict[int, float] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) < 2:
            continue
        try:
            rows[int(parts[0])] = float(parts[1])
        except ValueError:
            continue
    return rows


class NvidiaSmiCollector:
    def __init__(self, *, runner=subprocess.run, which=shutil.which):
        self._runner = runner
        self._which = which
        self._probe_cache: dict[str, Any] | None = None

    def probe(self) -> dict[str, Any]:
        if self._probe_cache is not None:
            return self._probe_cache

        binary = self._which("nvidia-smi")
        if binary is None:
            self._probe_cache = {
                "available": False,
                "error": "nvidia-smi not found",
            }
            return self._probe_cache

        cmd = [
            binary,
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = self._runner(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            stderr = getattr(exc, "stderr", "") or str(exc)
            self._probe_cache = {
                "available": False,
                "command": " ".join(cmd),
                "error": stderr.strip(),
            }
            return self._probe_cache

        devices = parse_nvidia_smi_device_csv(completed.stdout)
        self._probe_cache = {
            "available": True,
            "command": " ".join(cmd),
            "devices": devices,
        }
        return self._probe_cache

    def sample(self, tracked_pids: set[int]) -> dict[str, Any]:
        probe = self.probe()
        if not probe.get("available"):
            return {
                "gpu_metrics_available": False,
                "gpu_error": probe.get("error"),
                "gpu_device_count": 0,
                "gpu_device_names": [],
                "gpu_util_percent_max": None,
                "gpu_memory_used_total_mb": None,
                "gpu_memory_total_mb_total": None,
                "process_gpu_memory_mb": {},
            }

        binary = self._which("nvidia-smi")
        if binary is None:
            return {
                "gpu_metrics_available": False,
                "gpu_error": "nvidia-smi not found",
                "gpu_device_count": 0,
                "gpu_device_names": [],
                "gpu_util_percent_max": None,
                "gpu_memory_used_total_mb": None,
                "gpu_memory_total_mb_total": None,
                "process_gpu_memory_mb": {},
            }

        device_cmd = [
            binary,
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
        app_cmd = [
            binary,
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]

        try:
            device_completed = self._runner(
                device_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            stderr = getattr(exc, "stderr", "") or str(exc)
            return {
                "gpu_metrics_available": False,
                "gpu_error": stderr.strip(),
                "gpu_device_count": 0,
                "gpu_device_names": [],
                "gpu_util_percent_max": None,
                "gpu_memory_used_total_mb": None,
                "gpu_memory_total_mb_total": None,
                "process_gpu_memory_mb": {},
            }

        devices = parse_nvidia_smi_device_csv(device_completed.stdout)
        process_gpu_memory: dict[int, float] = {}
        try:
            app_completed = self._runner(
                app_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            process_gpu_memory = parse_nvidia_smi_compute_apps_csv(
                app_completed.stdout,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            process_gpu_memory = {}

        device_memory_used_total = sum(device["memory_used_mb"] for device in devices)
        device_memory_total_total = sum(device["memory_total_mb"] for device in devices)
        util_values = [device["utilization_gpu_percent"] for device in devices]
        tracked_memory = {
            pid: process_gpu_memory[pid]
            for pid in tracked_pids
            if pid in process_gpu_memory
        }

        return {
            "gpu_metrics_available": True,
            "gpu_error": None,
            "gpu_device_count": len(devices),
            "gpu_device_names": [device["name"] for device in devices],
            "gpu_driver_versions": sorted(
                {device["driver_version"] for device in devices}
            ),
            "gpu_util_percent_max": max(util_values) if util_values else None,
            "gpu_util_percent_mean": (
                sum(util_values) / len(util_values) if util_values else None
            ),
            "gpu_memory_used_total_mb": device_memory_used_total,
            "gpu_memory_total_mb_total": device_memory_total_total,
            "process_gpu_memory_mb": tracked_memory,
        }


def _proc_numeric_dirs() -> list[Path]:
    if not PROC_ROOT.exists():
        return []
    return sorted(
        path
        for path in PROC_ROOT.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def collect_proc_snapshots() -> dict[int, dict[str, Any]]:
    snapshots: dict[int, dict[str, Any]] = {}
    for proc_dir in _proc_numeric_dirs():
        stat_text = _read_text_if_exists(proc_dir / "stat")
        if not stat_text:
            continue
        try:
            stat_data = parse_proc_stat_line(stat_text)
        except ValueError:
            continue

        status_data = parse_proc_status_text(_read_text_if_exists(proc_dir / "status"))
        io_data = parse_proc_io_text(_read_text_if_exists(proc_dir / "io"))
        cmdline = _read_text_if_exists(proc_dir / "cmdline").replace("\x00", " ").strip()
        label = Path(cmdline.split()[0]).name if cmdline else stat_data["comm"]

        snapshots[stat_data["pid"]] = {
            **stat_data,
            **status_data,
            **io_data,
            "cmdline": cmdline,
            "cmdline_label": label,
        }
    return snapshots


def descendant_pids(root_pid: int, snapshots: dict[int, dict[str, Any]]) -> list[int]:
    if root_pid not in snapshots:
        return []

    children_by_parent: dict[int, list[int]] = {}
    for pid, row in snapshots.items():
        children_by_parent.setdefault(int(row["ppid"]), []).append(pid)

    ordered: list[int] = []
    queue: deque[int] = deque([root_pid])
    seen: set[int] = set()
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        for child in sorted(children_by_parent.get(current, [])):
            queue.append(child)
    return ordered


def sample_process_tree(
    root_pid: int,
    *,
    previous_cpu_ticks: dict[int, tuple[float, float]],
    timestamp: float,
    relative_seconds: float,
    sample_seq: int,
    gpu_collector: NvidiaSmiCollector,
    run_id: str,
    case_id: str,
    benchmark_mode: str,
    run_mode: str,
) -> tuple[list[dict[str, Any]], dict[int, tuple[float, float]]]:
    snapshots = collect_proc_snapshots()
    tree_pids = descendant_pids(root_pid, snapshots)
    if not tree_pids:
        return [], previous_cpu_ticks

    gpu_metrics = gpu_collector.sample(set(tree_pids))
    next_cpu_ticks: dict[int, tuple[float, float]] = {}
    rows: list[dict[str, Any]] = []
    tree_total_cpu_percent = 0.0
    tree_total_rss_bytes = 0
    tree_total_vms_bytes = 0
    tree_total_read_bytes = 0
    tree_total_write_bytes = 0

    process_rows: list[dict[str, Any]] = []
    for pid in tree_pids:
        snapshot = snapshots[pid]
        total_ticks = float(snapshot["utime_ticks"] + snapshot["stime_ticks"])
        cpu_percent = 0.0
        if pid in previous_cpu_ticks:
            previous_ticks, previous_timestamp = previous_cpu_ticks[pid]
            elapsed = timestamp - previous_timestamp
            if elapsed > 0:
                cpu_percent = (
                    (total_ticks - previous_ticks) / CLOCK_TICKS / elapsed
                ) * 100.0
                cpu_percent = max(0.0, cpu_percent)

        next_cpu_ticks[pid] = (total_ticks, timestamp)
        tree_total_cpu_percent += cpu_percent
        tree_total_rss_bytes += int(snapshot["rss_bytes"])
        tree_total_vms_bytes += int(snapshot["vms_bytes"])
        tree_total_read_bytes += int(snapshot["read_bytes"])
        tree_total_write_bytes += int(snapshot["write_bytes"])

        process_rows.append(
            {
                "pid": pid,
                "parent_pid": int(snapshot["ppid"]),
                "is_root": pid == root_pid,
                "cmdline_label": snapshot["cmdline_label"],
                "cmdline": snapshot["cmdline"],
                "state": snapshot["state"],
                "cpu_percent": cpu_percent,
                "rss_bytes": int(snapshot["rss_bytes"]),
                "vms_bytes": int(snapshot["vms_bytes"]),
                "thread_count": int(snapshot["threads"] or snapshot["num_threads"]),
                "read_bytes": int(snapshot["read_bytes"]),
                "write_bytes": int(snapshot["write_bytes"]),
                "process_gpu_memory_mb": gpu_metrics["process_gpu_memory_mb"].get(pid),
            }
        )

    process_count = len(process_rows)
    child_count = max(0, process_count - 1)
    for row in process_rows:
        rows.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "benchmark_mode": benchmark_mode,
                "run_mode": run_mode,
                "sample_seq": sample_seq,
                "timestamp": timestamp,
                "relative_seconds": relative_seconds,
                "root_pid": root_pid,
                "process_count": process_count,
                "child_count": child_count,
                "tree_total_cpu_percent": tree_total_cpu_percent,
                "tree_total_rss_bytes": tree_total_rss_bytes,
                "tree_total_vms_bytes": tree_total_vms_bytes,
                "tree_total_read_bytes": tree_total_read_bytes,
                "tree_total_write_bytes": tree_total_write_bytes,
                "gpu_metrics_available": gpu_metrics["gpu_metrics_available"],
                "gpu_error": gpu_metrics["gpu_error"],
                "gpu_device_count": gpu_metrics["gpu_device_count"],
                "gpu_device_names": gpu_metrics["gpu_device_names"],
                "gpu_util_percent_max": gpu_metrics["gpu_util_percent_max"],
                "gpu_util_percent_mean": gpu_metrics["gpu_util_percent_mean"],
                "gpu_memory_used_total_mb": gpu_metrics["gpu_memory_used_total_mb"],
                "gpu_memory_total_mb_total": gpu_metrics["gpu_memory_total_mb_total"],
                **row,
            }
        )

    return rows, next_cpu_ticks
