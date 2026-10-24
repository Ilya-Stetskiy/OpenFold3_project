#!/usr/bin/env python3
"""Adaptive CPU-heavy scheduler for FoldX/Rosetta experimental stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "run"))

from resource_scheduler import (  # noqa: E402
    build_cpu_heavy_plan,
    detect_resources,
    load_yaml,
    plan_to_json,
    plan_to_markdown,
    summarize_plan,
)


DEFAULT_RESOURCE_CONFIG = ROOT / "configs" / "resource_profiles.yaml"


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def report_paths(dataset: str) -> tuple[Path, Path]:
    report_dir = ROOT / "reports" / dataset
    report_dir.mkdir(parents=True, exist_ok=True)
    return (
        report_dir / "cpu_heavy_adaptive_plan.md",
        report_dir / "cpu_heavy_adaptive_plan.json",
    )


def write_plan_reports(dataset: str, plan: dict[str, Any]) -> tuple[Path, Path]:
    md_path, json_path = report_paths(dataset)
    md_path.write_text(plan_to_markdown(plan), encoding="utf-8")
    json_path.write_text(plan_to_json(plan), encoding="utf-8")
    return md_path, json_path


def build_executor_command(
    *,
    tool: dict[str, Any],
    dataset: str,
    mode: str,
    workers: int,
    max_jobs: int | None,
    resume: bool,
    force: bool,
    no_progress_bar: bool,
) -> list[str]:
    command = [sys.executable, str(ROOT / tool["script_path"]), "--dataset", dataset]
    if mode == "preflight":
        command.append("--preflight")
    elif mode == "materialize-plan":
        command.append("--materialize-plan")
    elif mode == "run":
        command.append("--run")
    else:
        command.append("--dry-run")
    command.extend(["--workers", str(workers)])
    if max_jobs is not None:
        command.extend(["--max-jobs", str(max_jobs)])
    if resume:
        command.append("--resume")
    if force:
        command.append("--force")
    if no_progress_bar:
        command.append("--no-progress-bar")
    return command


def executable_tools(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [tool for tool in plan.get("tools", []) if int(tool.get("workers", 0)) > 0 and tool.get("script_path")]


def enabled_tools(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [tool for tool in plan.get("tools", []) if bool(tool.get("enabled", False)) and tool.get("script_path")]


def run_commands(commands: list[list[str]], *, parallel: bool) -> int:
    if not commands:
        return 0
    if not parallel:
        exit_code = 0
        for command in commands:
            print("+ " + " ".join(command))
            completed = subprocess.run(command, cwd=ROOT, check=False)
            exit_code = max(exit_code, completed.returncode)
        return exit_code
    exit_code = 0
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = []
        for command in commands:
            print("+ " + " ".join(command))
            futures.append(executor.submit(subprocess.run, command, cwd=ROOT, check=False))
        for future in as_completed(futures):
            exit_code = max(exit_code, future.result().returncode)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--resource-config", type=Path, default=DEFAULT_RESOURCE_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--materialize-plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-total-workers", type=int)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--sequential-tools", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args(argv)

    selected_modes = [args.preflight, args.materialize_plan, args.run]
    if sum(bool(mode) for mode in selected_modes) > 1:
        raise SystemExit("Choose only one of --preflight, --materialize-plan, or --run.")
    mode = "dry-run"
    if args.preflight:
        mode = "preflight"
    elif args.materialize_plan:
        mode = "materialize-plan"
    elif args.run:
        mode = "run"

    config_path = args.resource_config if args.resource_config.is_absolute() else ROOT / args.resource_config
    config = load_yaml(config_path)
    snapshot = detect_resources(ROOT)
    plan = build_cpu_heavy_plan(
        config,
        snapshot,
        force_sequential=args.sequential_tools,
        requested_max_total_workers=args.max_total_workers,
    )
    md_path, json_path = write_plan_reports(args.dataset, plan)
    summary = summarize_plan(plan)
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {mode}")
    print(f"Resource config: {relative(config_path)}")
    print(f"Plan report: {relative(md_path)}")
    print(f"Worker budget: {summary['worker_budget']}")
    print(f"Workers by tool: {summary['workers_by_tool']}")
    if summary["warnings"]:
        print(f"Warnings: {summary['warnings']}")

    tools = executable_tools(plan) if mode == "run" else enabled_tools(plan)
    commands = [
        build_executor_command(
            tool=tool,
            dataset=args.dataset,
            mode=mode,
            workers=max(1, int(tool["workers"])),
            max_jobs=args.max_jobs,
            resume=args.resume,
            force=args.force,
            no_progress_bar=args.no_progress_bar,
        )
        for tool in tools
    ]
    if mode == "dry-run" or args.dry_run:
        print("Dry run: no CPU-heavy executors were launched.")
        for command in commands:
            print("planned: " + " ".join(command))
        return 0

    if mode == "run":
        print("Explicit --run received. CPU-heavy executors will be launched with adaptive worker counts.")
    parallel = bool(plan.get("allow_parallel_tools", True)) and not args.sequential_tools
    return run_commands(commands, parallel=parallel)


if __name__ == "__main__":
    raise SystemExit(main())
