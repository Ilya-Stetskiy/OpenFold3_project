from __future__ import annotations

import csv
import os
import shutil
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MonitoringArtifacts:
    resource_csv_path: Path
    stage_marks_path: Path
    monitor_plot_path: Path


def _read_cpu_times() -> tuple[int, int]:
    stat_path = Path("/proc/stat")
    if not stat_path.exists():
        return 0, 0

    parts = stat_path.read_text(encoding="utf-8").splitlines()[0].split()
    values = [int(value) for value in parts[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return idle, total


def _cpu_percent(previous: tuple[int, int] | None) -> tuple[float, tuple[int, int]]:
    current = _read_cpu_times()
    if previous is None or current == (0, 0):
        return 0.0, current

    prev_idle, prev_total = previous
    idle, total = current
    total_delta = total - prev_total
    idle_delta = idle - prev_idle
    if total_delta <= 0:
        return 0.0, current
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta))), current


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


def _disk_snapshot(path: Path) -> tuple[float, float]:
    usage = shutil.disk_usage(path)
    if usage.total <= 0:
        return 0.0, 0.0
    return usage.used / (1024**3), usage.used / usage.total * 100.0


def _load_average() -> float:
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return 0.0


class RunMonitor:
    def __init__(self, summary_dir: Path, *, sample_interval_seconds: float = 1.0) -> None:
        self.summary_dir = summary_dir
        self.sample_interval_seconds = sample_interval_seconds
        self.artifacts = MonitoringArtifacts(
            resource_csv_path=summary_dir / "resource_usage.csv",
            stage_marks_path=summary_dir / "stage_marks.csv",
            monitor_plot_path=summary_dir / "resource_usage.png",
        )
        self._started_at = time.monotonic()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._resource_handle = None
        self._resource_writer = None
        self._samples: list[dict[str, float | str]] = []

    def start(self) -> None:
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self._resource_handle = self.artifacts.resource_csv_path.open(
            "w", encoding="utf-8", newline=""
        )
        self._resource_writer = csv.DictWriter(
            self._resource_handle,
            fieldnames=[
                "timestamp_utc",
                "elapsed_seconds",
                "cpu_percent",
                "load_avg_1m",
                "memory_used_gb",
                "memory_percent",
                "disk_used_gb",
                "disk_percent",
            ],
        )
        self._resource_writer.writeheader()
        with self.artifacts.stage_marks_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp_utc", "elapsed_seconds", "stage", "details"])
        self.record_stage("monitor_started")
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()

    def record_stage(self, stage: str, details: str = "") -> None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        elapsed = time.monotonic() - self._started_at
        with self.artifacts.stage_marks_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([timestamp, f"{elapsed:.3f}", stage, details])

    def stop(self) -> MonitoringArtifacts:
        self.record_stage("monitor_stopping")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if self._resource_handle is not None:
            self._resource_handle.close()
        self._write_plot()
        return self.artifacts

    def _sampling_loop(self) -> None:
        previous_cpu = None
        while not self._stop_event.is_set():
            cpu_percent, previous_cpu = _cpu_percent(previous_cpu)
            memory_used_gb, memory_percent = _memory_snapshot()
            disk_used_gb, disk_percent = _disk_snapshot(self.summary_dir)
            sample = {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_seconds": round(time.monotonic() - self._started_at, 3),
                "cpu_percent": round(cpu_percent, 3),
                "load_avg_1m": round(_load_average(), 3),
                "memory_used_gb": round(memory_used_gb, 3),
                "memory_percent": round(memory_percent, 3),
                "disk_used_gb": round(disk_used_gb, 3),
                "disk_percent": round(disk_percent, 3),
            }
            self._samples.append(sample)
            assert self._resource_writer is not None
            self._resource_writer.writerow(sample)
            assert self._resource_handle is not None
            self._resource_handle.flush()
            if self._stop_event.wait(self.sample_interval_seconds):
                break

    def _write_plot(self) -> None:
        width = 1000
        height = 560
        image = _ImageCanvas(width=width, height=height)
        image.fill((255, 255, 255))
        if not self._samples:
            image.write_png(self.artifacts.monitor_plot_path)
            return

        left = 70
        right = width - 30
        top = 40
        bottom = height - 50
        chart_width = right - left
        chart_height = bottom - top

        image.draw_rect(left, top, right, bottom, (210, 210, 210))
        for grid_percent in (25, 50, 75):
            y = bottom - int(chart_height * (grid_percent / 100.0))
            image.draw_line(left, y, right, y, (235, 235, 235))

        max_elapsed = max(float(sample["elapsed_seconds"]) for sample in self._samples)
        if max_elapsed <= 0:
            max_elapsed = 1.0

        series = [
            ("cpu_percent", (32, 99, 155)),
            ("memory_percent", (44, 160, 44)),
            ("disk_percent", (214, 39, 40)),
        ]
        for field_name, color in series:
            points = []
            for sample in self._samples:
                elapsed = float(sample["elapsed_seconds"])
                value = max(0.0, min(100.0, float(sample[field_name])))
                x = left + int(chart_width * (elapsed / max_elapsed))
                y = bottom - int(chart_height * (value / 100.0))
                points.append((x, y))
            image.draw_polyline(points, color)

        for stage_elapsed in _read_stage_elapsed(self.artifacts.stage_marks_path):
            x = left + int(chart_width * (min(stage_elapsed, max_elapsed) / max_elapsed))
            image.draw_line(x, top, x, bottom, (160, 160, 160))

        image.write_png(self.artifacts.monitor_plot_path)


def _read_stage_elapsed(stage_marks_path: Path) -> list[float]:
    elapsed_values: list[float] = []
    with stage_marks_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                elapsed_values.append(float(row["elapsed_seconds"]))
            except (KeyError, TypeError, ValueError):
                continue
    return elapsed_values


class _ImageCanvas:
    def __init__(self, *, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.rows = [[255, 255, 255] * width for _ in range(height)]

    def fill(self, color: tuple[int, int, int]) -> None:
        row = list(color) * self.width
        for y in range(self.height):
            self.rows[y] = row.copy()

    def draw_rect(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        self.draw_line(x0, y0, x1, y0, color)
        self.draw_line(x0, y1, x1, y1, color)
        self.draw_line(x0, y0, x0, y1, color)
        self.draw_line(x1, y0, x1, y1, color)

    def draw_polyline(self, points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
        for start, end in zip(points, points[1:]):
            self.draw_line(start[0], start[1], end[0], end[1], color)

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self._set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            twice_err = 2 * err
            if twice_err >= dy:
                err += dy
                x0 += sx
            if twice_err <= dx:
                err += dx
                y0 += sy

    def _set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = x * 3
            self.rows[y][offset : offset + 3] = list(color)

    def write_png(self, path: Path) -> None:
        raw = bytearray()
        for row in self.rows:
            raw.append(0)
            raw.extend(bytes(row))

        def chunk(chunk_type: bytes, data: bytes) -> bytes:
            return (
                struct.pack("!I", len(data))
                + chunk_type
                + data
                + struct.pack("!I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
            )

        png = bytearray()
        png.extend(b"\x89PNG\r\n\x1a\n")
        png.extend(
            chunk(
                b"IHDR",
                struct.pack("!IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0),
            )
        )
        png.extend(chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
        png.extend(chunk(b"IEND", b""))
        path.write_bytes(bytes(png))
