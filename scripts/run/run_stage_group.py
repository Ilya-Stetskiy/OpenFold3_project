#!/usr/bin/env python3
"""Dry-run stage-group orchestrator for current_internal server execution."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


ROOT = Path(__file__).resolve().parents[2]
RUN_STAGES = ROOT / "configs" / "run_stages.yaml"
DATASETS = ROOT / "configs" / "datasets.yaml"
DEFAULT_SERVER_CONFIG = ROOT / "configs" / "server_paths.yaml"
EXAMPLE_SERVER_CONFIG = ROOT / "configs" / "server_paths.example.yaml"


def load_yaml(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise SystemExit(f"Missing required config: {path}")
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return data


def load_server_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path if path.is_absolute() else ROOT / path
    if path.exists():
        return load_yaml(path), {
            "server_config_path": path,
            "server_config_exists": True,
            "server_config_is_example_fallback": False,
        }
    if EXAMPLE_SERVER_CONFIG.exists():
        return load_yaml(EXAMPLE_SERVER_CONFIG), {
            "server_config_path": EXAMPLE_SERVER_CONFIG,
            "server_config_requested_path": path,
            "server_config_exists": False,
            "server_config_is_example_fallback": True,
        }
    return {}, {
        "server_config_path": path,
        "server_config_exists": False,
        "server_config_is_example_fallback": False,
    }


def marker_path(dataset: str, stage_group: str) -> Path:
    return ROOT / "data" / "run_state" / dataset / f"{stage_group}.completed.json"


def marker_status(dataset: str, stage_group: str) -> str:
    path = marker_path(dataset, stage_group)
    if not path.exists():
        return "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "invalid"
    return str(data.get("status", "unknown"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def estimate_counts(stage_group: str) -> dict[str, Any]:
    jobs = read_csv_rows(ROOT / "data" / "processed" / "current_internal" / "current_internal_recalc_jobs_resolved.csv")
    openfold_ready_jobs = read_csv_rows(ROOT / "data" / "processed" / "current_internal" / "current_internal_recalc_jobs_openfold_ready.csv")
    eval_tasks = read_csv_rows(ROOT / "data" / "processed" / "current_internal" / "current_internal_evaluation_tasks.csv")
    if stage_group == "cpu_heavy":
        return {
            "foldx_experimental_ready": sum(1 for row in jobs if row.get("method") == "foldx" and row.get("structure_source") == "experimental" and row.get("input_ready") == "true"),
            "rosetta_experimental_ready": sum(1 for row in jobs if row.get("method") == "rosetta" and row.get("structure_source") == "experimental" and row.get("input_ready") == "true"),
            "esm2_sequence_ready": sum(1 for row in jobs if row.get("method") == "esm2" and row.get("input_ready") == "true"),
        }
    if stage_group == "gpu_heavy":
        return {
            "pending_openfold_structure_jobs": sum(1 for row in jobs if row.get("structure_source") == "openfold" and row.get("input_status") == "pending_openfold_structure"),
            "esm2_sequence_ready": sum(1 for row in jobs if row.get("method") == "esm2" and row.get("structure_source") == "sequence_only" and row.get("input_ready") == "true"),
            "unique_openfold_sequences": sum(1 for _ in read_csv_rows(ROOT / "data" / "manifests" / "current_internal_sequence_manifest.csv")),
        }
    if stage_group == "post_gpu_cpu_heavy":
        source = openfold_ready_jobs or jobs
        return {
            "foldx_openfold_ready": sum(1 for row in source if row.get("method") == "foldx" and row.get("structure_source") == "openfold" and row.get("input_ready") == "true"),
            "rosetta_openfold_ready": sum(1 for row in source if row.get("method") == "rosetta" and row.get("structure_source") == "openfold" and row.get("input_ready") == "true"),
            "openfold_ready_jobs_table_exists": bool(openfold_ready_jobs),
        }
    if stage_group in {"analysis_reporting", "postprocess"}:
        return {
            "evaluation_tasks": len(eval_tasks),
            "evaluation_tasks_by_group": dict(Counter(row.get("metric_group", "") for row in eval_tasks)),
        }
    return {"resolved_jobs": len(jobs), "evaluation_tasks": len(eval_tasks)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-group", required=True)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--stage")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--server-config", type=Path, default=DEFAULT_SERVER_CONFIG)
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args()

    config = load_yaml(RUN_STAGES)
    datasets = load_yaml(DATASETS)
    server_config, server_meta = load_server_config(args.server_config)
    if args.dataset not in datasets.get("datasets", {}):
        raise SystemExit(f"Unknown dataset: {args.dataset}")
    groups = config.get("stage_groups", {})
    if args.stage_group not in groups:
        raise SystemExit(f"Unknown stage_group: {args.stage_group}")
    stages = [stage for stage in config.get("stages", []) if stage.get("stage_group") == args.stage_group]
    if args.stage:
        stages = [stage for stage in stages if stage.get("stage_name") == args.stage]
        if not stages:
            raise SystemExit(f"Stage not found in {args.stage_group}: {args.stage}")

    group = groups[args.stage_group]
    print(f"Stage group: {args.stage_group}")
    print(f"Dataset: {args.dataset}")
    print(f"Server config: {server_meta['server_config_path'].as_posix()}")
    print(f"Server config exists: {server_meta['server_config_exists']}")
    print(f"Server config example fallback: {server_meta['server_config_is_example_fallback']}")
    server_paths = server_config.get("server", {})
    tools = server_config.get("tools", {})
    gpu = server_config.get("gpu", {})
    print(f"Project root: {server_paths.get('project_root', '')}")
    print(f"Results root: {server_paths.get('results_root', '')}")
    print(f"Logs root: {server_paths.get('logs_root', '')}")
    print(f"Python bin: {tools.get('python_bin', 'python3')}")
    if args.stage_group == "gpu_heavy":
        print(f"CUDA_VISIBLE_DEVICES: {gpu.get('cuda_visible_devices', '')}")
        print(f"Max parallel OpenFold jobs: {gpu.get('max_parallel_openfold_jobs', '')}")
    print(f"Description: {group.get('description', '')}")
    print(f"Requires GPU: {group.get('requires_gpu', False)}")
    print(f"Can run parallel: {group.get('can_run_parallel', False)}")
    if group.get("parallel_with"):
        print(f"Parallel with: {', '.join(group['parallel_with'])}")
    print("Dependency markers:")
    dependency_errors = []
    for dependency in group.get("waits_for", []):
        status = marker_status(args.dataset, dependency)
        print(f"  - {dependency}: {status} ({marker_path(args.dataset, dependency).as_posix()})")
        if status not in {"completed", "completed_with_failures"}:
            dependency_errors.append(dependency)
    if dependency_errors:
        message = f"Dependencies not completed: {', '.join(dependency_errors)}"
        print(f"WARNING: {message}")
        if not args.dry_run and args.stage_group in {"post_gpu_cpu_heavy", "analysis_reporting", "postprocess"}:
            raise SystemExit(message)
    print(f"Estimated counts: {json.dumps(estimate_counts(args.stage_group), sort_keys=True)}")
    print("Stages:")
    for stage in stages:
        print(f"\n- {stage['stage_name']} [{stage.get('status', 'planned')}]")
        print(f"  script: {stage.get('script_path') or 'planned, executor not implemented yet'}")
        print(f"  requires_gpu: {stage.get('requires_gpu', False)}")
        print(f"  inputs: {stage.get('input_files', [])}")
        print(f"  outputs: {stage.get('output_files', [])}")
        print(f"  depends_on: {stage.get('depends_on', [])}")
        if stage.get("status") == "planned":
            print("  action: planned, executor not implemented yet")
        else:
            print("  action: dry-run only; no heavy tool is launched")
    if args.dry_run:
        print("\nDry run: no stages were executed and no completion markers were created.")
        return 0
    print("\nNon-dry execution is not implemented for this orchestrator yet.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
