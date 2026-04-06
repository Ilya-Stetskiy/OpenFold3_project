from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROC_ROOT = Path("/proc")
CLOCK_TICKS = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
PROFILE_EVENT_PREFIX = "OF3_PROFILE_EVENT"


@dataclass(slots=True, frozen=True)
class PanelProfilingArtifacts:
    output_root: Path
    events_path: Path
    samples_path: Path
    summary_path: Path
    timeline_svg_path: Path


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_profile_event_from_line(line: str) -> dict[str, Any] | None:
    index = line.find(PROFILE_EVENT_PREFIX)
    if index < 0:
        return None
    payload_text = line[index + len(PROFILE_EVENT_PREFIX) :].strip()
    if not payload_text:
        return None
    try:
        return json.loads(payload_text)
    except json.JSONDecodeError:
        return None


def _derive_events_from_log_line(
    line: str,
    *,
    run_id: str,
    timestamp: float,
    relative_seconds: float,
    source: str,
) -> list[dict[str, Any]]:
    base = {
        "run_id": run_id,
        "timestamp": timestamp,
        "relative_seconds": relative_seconds,
        "source": source,
        "raw_line": line.rstrip(),
    }
    payload = _parse_profile_event_from_line(line)
    if payload is not None:
        return [{**base, **payload}]
    if "Loading weights from" in line:
        return [{**base, "stage": "checkpoint_load", "event": "start", "inferred": True}]
    if "Beginning inference prediction" in line:
        return [
            {**base, "stage": "checkpoint_load", "event": "end", "inferred": True},
            {**base, "stage": "predict_total", "event": "start", "inferred": True},
        ]
    return []


def _proc_numeric_dirs() -> list[Path]:
    if not PROC_ROOT.exists():
        return []
    return sorted(
        path for path in PROC_ROOT.iterdir() if path.is_dir() and path.name.isdigit()
    )


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse_proc_stat_line(line: str) -> dict[str, Any]:
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


def _parse_proc_status_text(text: str) -> dict[str, int]:
    values = {"rss_bytes": 0, "vms_bytes": 0, "threads": 0}
    for raw_line in text.splitlines():
        if raw_line.startswith("VmRSS:"):
            values["rss_bytes"] = int(raw_line.split()[1]) * 1024
        elif raw_line.startswith("VmSize:"):
            values["vms_bytes"] = int(raw_line.split()[1]) * 1024
        elif raw_line.startswith("Threads:"):
            values["threads"] = int(raw_line.split()[1])
    return values


def _parse_proc_io_text(text: str) -> dict[str, int]:
    values = {"read_bytes": 0, "write_bytes": 0}
    for raw_line in text.splitlines():
        if raw_line.startswith("read_bytes:"):
            values["read_bytes"] = int(raw_line.split()[1])
        elif raw_line.startswith("write_bytes:"):
            values["write_bytes"] = int(raw_line.split()[1])
    return values


def _collect_proc_snapshots() -> dict[int, dict[str, Any]]:
    snapshots: dict[int, dict[str, Any]] = {}
    for proc_dir in _proc_numeric_dirs():
        stat_text = _read_text_if_exists(proc_dir / "stat")
        if not stat_text:
            continue
        try:
            stat_data = _parse_proc_stat_line(stat_text)
        except ValueError:
            continue
        status_data = _parse_proc_status_text(_read_text_if_exists(proc_dir / "status"))
        io_data = _parse_proc_io_text(_read_text_if_exists(proc_dir / "io"))
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


def _descendant_pids(root_pid: int, snapshots: dict[int, dict[str, Any]]) -> list[int]:
    if root_pid not in snapshots:
        return []
    children_by_parent: dict[int, list[int]] = {}
    for pid, row in snapshots.items():
        children_by_parent.setdefault(int(row["ppid"]), []).append(pid)
    ordered: list[int] = []
    queue = [root_pid]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        queue.extend(sorted(children_by_parent.get(current, [])))
    return ordered


class _NvidiaSmiCollector:
    def __init__(self) -> None:
        self._probe_cache: dict[str, Any] | None = None

    def probe(self) -> dict[str, Any]:
        if self._probe_cache is not None:
            return self._probe_cache
        binary = shutil.which("nvidia-smi")
        if binary is None:
            self._probe_cache = {"available": False, "error": "nvidia-smi not found"}
            return self._probe_cache
        cmd = [
            binary,
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            self._probe_cache = {"available": False, "error": str(exc)}
            return self._probe_cache
        devices = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.strip().split(",")]
            if len(parts) != 6:
                continue
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_gpu_percent": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "memory_total_mb": float(parts[4]),
                    "driver_version": parts[5],
                }
            )
        self._probe_cache = {"available": True, "devices": devices}
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
                "gpu_util_percent_mean": None,
                "gpu_memory_used_total_mb": None,
                "gpu_memory_total_mb_total": None,
                "process_gpu_memory_mb": {},
            }
        binary = shutil.which("nvidia-smi")
        if binary is None:
            return {
                "gpu_metrics_available": False,
                "gpu_error": "nvidia-smi not found",
                "gpu_device_count": 0,
                "gpu_device_names": [],
                "gpu_util_percent_max": None,
                "gpu_util_percent_mean": None,
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
            device_completed = subprocess.run(
                device_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            return {
                "gpu_metrics_available": False,
                "gpu_error": str(exc),
                "gpu_device_count": 0,
                "gpu_device_names": [],
                "gpu_util_percent_max": None,
                "gpu_util_percent_mean": None,
                "gpu_memory_used_total_mb": None,
                "gpu_memory_total_mb_total": None,
                "process_gpu_memory_mb": {},
            }
        devices = []
        for line in device_completed.stdout.splitlines():
            parts = [part.strip() for part in line.strip().split(",")]
            if len(parts) != 6:
                continue
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_gpu_percent": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "memory_total_mb": float(parts[4]),
                    "driver_version": parts[5],
                }
            )
        process_gpu_memory: dict[int, float] = {}
        try:
            app_completed = subprocess.run(
                app_cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            for line in app_completed.stdout.splitlines():
                parts = [part.strip() for part in line.strip().split(",")]
                if len(parts) < 2:
                    continue
                try:
                    process_gpu_memory[int(parts[0])] = float(parts[1])
                except ValueError:
                    continue
        except (FileNotFoundError, subprocess.CalledProcessError):
            process_gpu_memory = {}
        util_values = [device["utilization_gpu_percent"] for device in devices]
        return {
            "gpu_metrics_available": True,
            "gpu_error": None,
            "gpu_device_count": len(devices),
            "gpu_device_names": [device["name"] for device in devices],
            "gpu_util_percent_max": max(util_values) if util_values else None,
            "gpu_util_percent_mean": (
                sum(util_values) / len(util_values) if util_values else None
            ),
            "gpu_memory_used_total_mb": sum(device["memory_used_mb"] for device in devices),
            "gpu_memory_total_mb_total": sum(
                device["memory_total_mb"] for device in devices
            ),
            "process_gpu_memory_mb": {
                pid: process_gpu_memory[pid]
                for pid in tracked_pids
                if pid in process_gpu_memory
            },
        }


def _sample_process_tree(
    root_pid: int,
    *,
    previous_cpu_ticks: dict[int, tuple[float, float]],
    timestamp: float,
    relative_seconds: float,
    sample_seq: int,
    gpu_collector: _NvidiaSmiCollector,
    run_id: str,
) -> tuple[list[dict[str, Any]], dict[int, tuple[float, float]]]:
    snapshots = _collect_proc_snapshots()
    tree_pids = _descendant_pids(root_pid, snapshots)
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
                cpu_percent = ((total_ticks - previous_ticks) / CLOCK_TICKS / elapsed) * 100.0
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


def _event_rows_to_summary_metrics(
    samples: list[dict[str, Any]], events: list[dict[str, Any]], *, wall_seconds: float
) -> dict[str, Any]:
    aggregate_samples = []
    seen_seq: set[int] = set()
    for row in samples:
        sample_seq = int(row["sample_seq"])
        if sample_seq in seen_seq:
            continue
        seen_seq.add(sample_seq)
        aggregate_samples.append(row)

    def _stage_duration(stage: str) -> float | None:
        matching = [row for row in events if row.get("stage") == stage]
        if not matching:
            return None
        explicit = [
            float(row["duration_seconds"])
            for row in matching
            if row.get("event") == "end" and row.get("duration_seconds") is not None
        ]
        if explicit:
            return max(explicit)
        starts = [
            float(row["relative_seconds"])
            for row in matching
            if row.get("event") == "start" and row.get("relative_seconds") is not None
        ]
        ends = [
            float(row["relative_seconds"])
            for row in matching
            if row.get("event") == "end" and row.get("relative_seconds") is not None
        ]
        if not starts or not ends:
            return None
        return max(0.0, max(ends) - min(starts))

    if not aggregate_samples:
        return {
            "wall_seconds": wall_seconds,
            "checkpoint_load_seconds": _stage_duration("checkpoint_load"),
            "predict_seconds": _stage_duration("predict_total"),
            "peak_rss_gb": None,
            "peak_cpu_percent": None,
            "peak_gpu_memory_gb": None,
            "mean_gpu_util_percent": None,
            "max_gpu_util_percent": None,
            "process_count_peak": None,
            "gpu_metrics_available": False,
            "sample_count": 0,
            "event_count": len(events),
        }

    gpu_samples = [
        row for row in aggregate_samples if row.get("gpu_metrics_available") is True
    ]
    return {
        "wall_seconds": wall_seconds,
        "checkpoint_load_seconds": _stage_duration("checkpoint_load"),
        "predict_seconds": _stage_duration("predict_total"),
        "peak_rss_gb": max(float(row["tree_total_rss_bytes"]) for row in aggregate_samples)
        / (1024**3),
        "peak_cpu_percent": max(float(row["tree_total_cpu_percent"]) for row in aggregate_samples),
        "peak_gpu_memory_gb": (
            None
            if not gpu_samples
            else max(float(row["gpu_memory_used_total_mb"]) for row in gpu_samples) / 1024.0
        ),
        "mean_gpu_util_percent": (
            None
            if not gpu_samples
            else sum(float(row["gpu_util_percent_max"]) for row in gpu_samples) / len(gpu_samples)
        ),
        "max_gpu_util_percent": (
            None
            if not gpu_samples
            else max(float(row["gpu_util_percent_max"]) for row in gpu_samples)
        ),
        "process_count_peak": max(int(row["process_count"]) for row in aggregate_samples),
        "gpu_metrics_available": bool(gpu_samples),
        "sample_count": len(aggregate_samples),
        "event_count": len(events),
    }


def _write_timeline_svg(
    samples: list[dict[str, Any]],
    *,
    output_path: Path,
    title: str,
) -> Path:
    aggregate_samples = []
    seen_seq: set[int] = set()
    for row in samples:
        sample_seq = int(row["sample_seq"])
        if sample_seq in seen_seq:
            continue
        seen_seq.add(sample_seq)
        aggregate_samples.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not aggregate_samples:
        output_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='140'>"
            f"<text x='30' y='70'>{title}: no telemetry samples collected</text></svg>",
            encoding="utf-8",
        )
        return output_path

    width = 1200
    height = 820
    left = 90
    right = 40
    top = 70
    panel_height = 150
    panel_gap = 30
    plot_width = width - left - right
    x_min = min(float(row["relative_seconds"]) for row in aggregate_samples)
    x_max = max(float(row["relative_seconds"]) for row in aggregate_samples)
    if x_max <= x_min:
        x_max = x_min + 1.0

    panels = [
        ("CPU %", "tree_total_cpu_percent", 1.0),
        ("RSS (GiB)", "tree_total_rss_bytes", 1 / (1024**3)),
        ("GPU util %", "gpu_util_percent_max", 1.0),
        ("GPU mem (GiB)", "gpu_memory_used_total_mb", 1 / 1024.0),
    ]

    def _scale_x(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * plot_width

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<style>.axis{stroke:#444;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.line{fill:none;stroke-width:2}</style>",
        f"<text x='{left}' y='30' font-size='20' font-family='sans-serif'>{title}</text>",
    ]
    colors = ["#20639b", "#3caea3", "#ed553b", "#6c5ce7"]
    for panel_index, (label, field, multiplier) in enumerate(panels):
        panel_top = top + panel_index * (panel_height + panel_gap)
        panel_bottom = panel_top + panel_height
        values = [
            (float(row["relative_seconds"]), float(row[field]) * multiplier)
            for row in aggregate_samples
            if row.get(field) is not None
        ]
        y_values = [value for _, value in values] or [0.0, 1.0]
        y_min = min(y_values)
        y_max = max(y_values)
        if y_min == y_max:
            y_max = y_min + 1.0
        lines.append(
            f"<line class='axis' x1='{left}' y1='{panel_bottom}' x2='{width-right}' y2='{panel_bottom}'/>"
        )
        lines.append(
            f"<line class='axis' x1='{left}' y1='{panel_top}' x2='{left}' y2='{panel_bottom}'/>"
        )
        lines.append(
            f"<text x='{left-12}' y='{panel_top+14}' font-size='12' text-anchor='end'>{label}</text>"
        )
        if values:
            points = []
            for x_value, y_value in values:
                x = _scale_x(x_value)
                y = panel_bottom - ((y_value - y_min) / (y_max - y_min)) * panel_height
                points.append(f"{x:.1f},{y:.1f}")
            lines.append(
                f"<polyline class='line' stroke='{colors[panel_index % len(colors)]}' points='{' '.join(points)}'/>"
            )
    lines.append("</svg>")
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path


class PanelExperimentProfiler:
    def __init__(
        self,
        *,
        output_root: Path,
        run_id: str,
        sample_interval_seconds: float = 1.0,
        root_pid: int | None = None,
    ) -> None:
        self.output_root = output_root
        self.run_id = run_id
        self.sample_interval_seconds = sample_interval_seconds
        self.root_pid = os.getpid() if root_pid is None else root_pid
        self.artifacts = PanelProfilingArtifacts(
            output_root=output_root,
            events_path=output_root / "events.jsonl",
            samples_path=output_root / "samples.jsonl",
            summary_path=output_root / "summary.json",
            timeline_svg_path=output_root / "timeline.svg",
        )
        self._started_wall = time.perf_counter()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._events: list[dict[str, Any]] = []
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._previous_cpu_ticks: dict[int, tuple[float, float]] = {}
        self._gpu_collector = _NvidiaSmiCollector()

    def start(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.record_stage("experiment_total", "start", source="profiler")
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()

    def record_stage(
        self,
        stage: str,
        event: str,
        *,
        source: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "run_id": self.run_id,
            "timestamp": time.time(),
            "relative_seconds": time.perf_counter() - self._started_wall,
            "stage": stage,
            "event": event,
            "source": source,
        }
        if details:
            payload.update(details)
        with self._lock:
            self._events.append(payload)
        _append_jsonl(self.artifacts.events_path, [payload])

    def record_log_line(self, line: str, *, source: str) -> None:
        rows = _derive_events_from_log_line(
            line,
            run_id=self.run_id,
            timestamp=time.time(),
            relative_seconds=time.perf_counter() - self._started_wall,
            source=source,
        )
        if not rows:
            return
        with self._lock:
            self._events.extend(rows)
        _append_jsonl(self.artifacts.events_path, rows)

    def stop(self) -> PanelProfilingArtifacts:
        self.record_stage("experiment_total", "end", source="profiler")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            samples = list(self._samples)
            events = list(self._events)
        wall_seconds = time.perf_counter() - self._started_wall
        summary = _event_rows_to_summary_metrics(samples, events, wall_seconds=wall_seconds)
        self.artifacts.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_timeline_svg(samples, output_path=self.artifacts.timeline_svg_path, title=self.run_id)
        return self.artifacts

    def _sampling_loop(self) -> None:
        sample_seq = 0
        while not self._stop_event.is_set():
            sample_seq += 1
            rows, next_ticks = _sample_process_tree(
                self.root_pid,
                previous_cpu_ticks=self._previous_cpu_ticks,
                timestamp=time.time(),
                relative_seconds=time.perf_counter() - self._started_wall,
                sample_seq=sample_seq,
                gpu_collector=self._gpu_collector,
                run_id=self.run_id,
            )
            if rows:
                with self._lock:
                    self._samples.extend(rows)
                    self._previous_cpu_ticks = next_ticks
                _append_jsonl(self.artifacts.samples_path, rows)
            if self._stop_event.wait(self.sample_interval_seconds):
                break
