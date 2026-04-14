from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .schemas import ResourceSnapshot


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_cpu_times() -> tuple[int, int]:
    stat_path = Path("/proc/stat")
    if not stat_path.exists():
        return 0, 0
    first_line = stat_path.read_text(encoding="utf-8").splitlines()[0]
    values = [int(value) for value in first_line.split()[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return idle, sum(values)


class ResourceSampler:
    def __init__(self, disk_path: Path) -> None:
        self.disk_path = disk_path
        self._previous_cpu: tuple[int, int] | None = None

    def snapshot(self) -> ResourceSnapshot:
        cpu_percent = self._cpu_percent()
        memory_used_gb, memory_percent = self._memory_snapshot()
        disk_free_gb, disk_percent = self._disk_snapshot()
        return ResourceSnapshot(
            timestamp_utc=utc_now(),
            cpu_percent=round(cpu_percent, 3),
            load_avg_1m=round(self._load_average(), 3),
            memory_used_gb=round(memory_used_gb, 3),
            memory_percent=round(memory_percent, 3),
            disk_free_gb=round(disk_free_gb, 3),
            disk_percent=round(disk_percent, 3),
            gpu=self._gpu_snapshot(),
        )

    def _cpu_percent(self) -> float:
        current = _read_cpu_times()
        previous = self._previous_cpu
        self._previous_cpu = current
        if previous is None or current == (0, 0):
            return 0.0
        prev_idle, prev_total = previous
        idle, total = current
        total_delta = total - prev_total
        idle_delta = idle - prev_idle
        if total_delta <= 0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))

    @staticmethod
    def _memory_snapshot() -> tuple[float, float]:
        meminfo_path = Path("/proc/meminfo")
        if not meminfo_path.exists():
            return 0.0, 0.0
        values: dict[str, int] = {}
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", maxsplit=1)
            values[key] = int(raw.strip().split()[0])
        total_kb = values.get("MemTotal", 0)
        available_kb = values.get("MemAvailable", 0)
        if total_kb <= 0:
            return 0.0, 0.0
        used_kb = total_kb - available_kb
        return used_kb / (1024 * 1024), used_kb / total_kb * 100.0

    def _disk_snapshot(self) -> tuple[float, float]:
        target = self.disk_path
        target.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(target)
        if usage.total <= 0:
            return 0.0, 0.0
        return usage.free / (1024**3), usage.used / usage.total * 100.0

    @staticmethod
    def _load_average() -> float:
        try:
            return float(os.getloadavg()[0])
        except (AttributeError, OSError):
            return 0.0

    @staticmethod
    def _gpu_snapshot() -> list[dict[str, Any]]:
        query = (
            "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu"
        )
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        rows: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 6:
                continue
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "utilization_percent": float(parts[4]),
                    "temperature_c": float(parts[5]),
                }
            )
        return rows


class TelemetryRecorder:
    def __init__(
        self,
        path: Path,
        sampler: ResourceSampler,
        interval_seconds: float,
    ) -> None:
        self.path = path
        self.sampler = sampler
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def _loop(self) -> None:
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp_utc",
                    "cpu_percent",
                    "load_avg_1m",
                    "memory_used_gb",
                    "memory_percent",
                    "disk_free_gb",
                    "disk_percent",
                    "gpu_json",
                ],
            )
            writer.writeheader()
            while not self._stop_event.is_set():
                snapshot = self.sampler.snapshot()
                writer.writerow(
                    {
                        **snapshot.model_dump(exclude={"gpu"}),
                        "gpu_json": json.dumps(snapshot.gpu, sort_keys=True),
                    }
                )
                handle.flush()
                if self._stop_event.wait(self.interval_seconds):
                    break
