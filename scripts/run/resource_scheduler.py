#!/usr/bin/env python3
"""Resource-aware planning helpers for CPU-heavy executor stages."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_count: int
    load1: float | None
    load5: float | None
    load15: float | None
    total_ram_gb: float | None
    available_ram_gb: float | None
    disk_free_gb: float
    disk_path: str


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def detect_resources(disk_path: Path | str | None = None) -> ResourceSnapshot:
    cpu_count = os.cpu_count() or 1
    try:
        load1, load5, load15 = os.getloadavg()
    except (AttributeError, OSError):
        load1 = load5 = load15 = None
    total_ram_gb, available_ram_gb = _read_meminfo_gb()
    disk_target = Path(disk_path or ROOT)
    usage = shutil.disk_usage(disk_target)
    return ResourceSnapshot(
        cpu_count=cpu_count,
        load1=load1,
        load5=load5,
        load15=load15,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        disk_free_gb=round(usage.free / (1024**3), 3),
        disk_path=str(disk_target),
    )


def build_cpu_heavy_plan(
    config: dict[str, Any],
    snapshot: ResourceSnapshot,
    *,
    force_sequential: bool = False,
    requested_max_total_workers: int | None = None,
) -> dict[str, Any]:
    defaults = config.get("defaults", {})
    cpu_profile = config.get("cpu_heavy", {})
    tools = cpu_profile.get("tools", {})
    min_free_disk_gb = float(defaults.get("min_free_disk_gb", 20))
    min_free_ram_gb = float(defaults.get("min_free_ram_gb", 8))
    allow_parallel_tools = bool(defaults.get("allow_parallel_tools", True)) and not force_sequential
    max_total_workers = _max_total_workers(defaults.get("max_total_workers", "auto"), requested_max_total_workers)
    worker_budget = _worker_budget(defaults, snapshot, max_total_workers)
    disk_status = "ok" if snapshot.disk_free_gb >= min_free_disk_gb else "warning"
    warnings = []
    if disk_status != "ok":
        warnings.append(f"free disk {snapshot.disk_free_gb:.2f} GB is below configured minimum {min_free_disk_gb:.2f} GB")
    if snapshot.available_ram_gb is None:
        warnings.append("available RAM could not be detected; CPU/load limits only were used")
    if snapshot.load1 is None:
        warnings.append("load average could not be detected; CPU count limits only were used")

    enabled_tools = [
        (name, tool)
        for name, tool in tools.items()
        if bool(tool.get("enabled", False)) and tool.get("script_path")
    ]
    enabled_tools.sort(key=lambda item: int(item[1].get("priority", 100)))

    remaining_cpu_slots = worker_budget
    planned_tools: list[dict[str, Any]] = []
    for name, tool in enabled_tools:
        allocation = _allocate_tool_workers(tool, snapshot, remaining_cpu_slots, min_free_ram_gb)
        if allocation["workers"] > 0 and allow_parallel_tools:
            remaining_cpu_slots = max(0, remaining_cpu_slots - allocation["cpu_slots_used"])
        elif allocation["workers"] > 0 and not allow_parallel_tools:
            remaining_cpu_slots = 0
        planned_tools.append({"tool_name": name, **tool, **allocation})

    return {
        "resource_snapshot": asdict(snapshot),
        "worker_budget": worker_budget,
        "allow_parallel_tools": allow_parallel_tools,
        "disk_status": disk_status,
        "warnings": warnings,
        "tools": planned_tools,
    }


def summarize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    tools = plan.get("tools", [])
    return {
        "worker_budget": plan.get("worker_budget", 0),
        "planned_tools": len([tool for tool in tools if int(tool.get("workers", 0)) > 0]),
        "workers_by_tool": {tool["tool_name"]: int(tool.get("workers", 0)) for tool in tools},
        "warnings": plan.get("warnings", []),
    }


def plan_to_markdown(plan: dict[str, Any]) -> str:
    snapshot = plan["resource_snapshot"]
    lines = [
        "# CPU-heavy adaptive resource plan",
        "",
        "This is a planning report. Heavy tools are not launched unless the adaptive runner is called with `--run`.",
        "",
        "## Resource snapshot",
        "",
        f"- CPU cores: {snapshot.get('cpu_count')}",
        f"- Load average 1/5/15: {snapshot.get('load1')} / {snapshot.get('load5')} / {snapshot.get('load15')}",
        f"- Available RAM GB: {snapshot.get('available_ram_gb')}",
        f"- Disk free GB: {snapshot.get('disk_free_gb')} at `{snapshot.get('disk_path')}`",
        "",
        "## Plan",
        "",
        f"- Worker budget: {plan.get('worker_budget')}",
        f"- Parallel tools allowed: {plan.get('allow_parallel_tools')}",
        f"- Disk status: {plan.get('disk_status')}",
        "",
        "| tool | enabled | workers | cpu/job | ram/job GB | script | reason |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for tool in plan.get("tools", []):
        lines.append(
            "| {tool_name} | {enabled} | {workers} | {cpu_per_job} | {ram_gb_per_job} | `{script_path}` | {reason} |".format(
                **tool
            )
        )
    if plan.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in plan["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def plan_to_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _read_meminfo_gb() -> tuple[float | None, float | None]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None, None
    values: dict[str, float] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            values[parts[0][:-1]] = float(parts[1]) / (1024**2)
    total = values.get("MemTotal")
    available = values.get("MemAvailable", values.get("MemFree"))
    return (round(total, 3) if total is not None else None, round(available, 3) if available is not None else None)


def _max_total_workers(config_value: Any, requested: int | None) -> int | None:
    if requested is not None:
        return max(1, requested)
    if str(config_value).strip().lower() == "auto":
        return None
    try:
        return max(1, int(config_value))
    except (TypeError, ValueError):
        return None


def _worker_budget(defaults: dict[str, Any], snapshot: ResourceSnapshot, max_total_workers: int | None) -> int:
    reserve_cpu_cores = int(defaults.get("reserve_cpu_cores", 1))
    cpu_budget = max(1, snapshot.cpu_count - reserve_cpu_cores)
    if snapshot.load1 is not None:
        limit_per_core = float(defaults.get("load_average_limit_per_core", 0.85))
        load_budget = max(1, math.floor(snapshot.cpu_count * limit_per_core - snapshot.load1))
        cpu_budget = min(cpu_budget, load_budget)
    if max_total_workers is not None:
        cpu_budget = min(cpu_budget, max_total_workers)
    return max(1, cpu_budget)


def _allocate_tool_workers(tool: dict[str, Any], snapshot: ResourceSnapshot, remaining_cpu_slots: int, min_free_ram_gb: float) -> dict[str, Any]:
    if remaining_cpu_slots <= 0:
        return {"workers": 0, "cpu_slots_used": 0, "reason": "no remaining CPU budget"}
    cpu_per_job = max(1, int(tool.get("cpu_per_job", 1)))
    ram_gb_per_job = max(0.1, float(tool.get("ram_gb_per_job", 1)))
    min_workers = max(0, int(tool.get("min_workers", 0)))
    max_workers = max(min_workers, int(tool.get("max_workers", 1)))
    by_cpu = max(0, remaining_cpu_slots // cpu_per_job)
    if snapshot.available_ram_gb is None:
        by_ram = max_workers
    else:
        usable_ram = max(0.0, snapshot.available_ram_gb - min_free_ram_gb)
        by_ram = int(usable_ram // ram_gb_per_job)
    workers = min(max_workers, by_cpu, by_ram)
    if workers < min_workers and by_cpu >= min_workers and by_ram >= min_workers:
        workers = min_workers
    reason = "planned"
    if workers == 0:
        reason = f"insufficient budget by_cpu={by_cpu}, by_ram={by_ram}"
    return {"workers": workers, "cpu_slots_used": workers * cpu_per_job, "reason": reason}
