#!/usr/bin/env python3
"""Dry-run/materialized-plan skeleton for future DDG executors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from execution_utils import (
    build_job_manifest,
    build_job_output_dir,
    ensure_dir,
    filter_jobs_for_stage,
    load_yaml,
    read_csv_dicts,
    safe_write_text,
    stage_group_for_stage,
    utc_now_iso,
    write_job_manifest,
    write_json,
    write_stage_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_STAGES = {"foldx_experimental", "rosetta_experimental", "esm2_cpu_optional", "esm2_gpu_optional"}


def resolved_jobs_path(dataset: str) -> Path:
    return ROOT / "data" / "processed" / dataset / f"{dataset}_recalc_jobs_resolved.csv"


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parsed_result_path(dataset: str, method: str, structure_source: str, mutation_record_id: str) -> str:
    return f"results/{dataset}/{method}/{structure_source}/{mutation_record_id}/parsed_result.json"


def command_placeholder(stage_name: str, job: dict[str, str]) -> str:
    return (
        "# Placeholder only. Real executor is not implemented yet.\n"
        f"# stage_name={stage_name}\n"
        f"# job_id={job.get('job_id', '')}\n"
        f"# method={job.get('method', '')}\n"
        f"# structure_source={job.get('structure_source', '')}\n"
    )


def build_plan(dataset: str, stage_name: str, max_jobs: int | None = None, mutation_record_id: str | None = None) -> list[dict[str, str]]:
    rows = read_csv_dicts(resolved_jobs_path(dataset))
    rows = filter_jobs_for_stage(rows, stage_name)
    if mutation_record_id:
        rows = [row for row in rows if row.get("mutation_record_id") == mutation_record_id]
    if max_jobs is not None:
        rows = rows[:max_jobs]
    return rows


def write_dry_run_plan(dataset: str, stage_group: str, stage_name: str, planned_jobs: list[dict[str, str]], force: bool) -> Path:
    plan_path = ROOT / "data" / "run_state" / dataset / f"{stage_group}.dry_run_plan.json"
    payload = {
        "dataset_name": dataset,
        "stage_group": stage_group,
        "stage_name": stage_name,
        "status": "dry_run_plan",
        "created_at": utc_now_iso(),
        "planned_jobs": len(planned_jobs),
        "notes": "Dry-run plan only. No completion marker was created.",
    }
    write_json(plan_path, payload, force=force)
    return plan_path


def write_reports(dataset: str, stage_name: str, stage_group: str, planned_jobs: list[dict[str, str]], force: bool) -> tuple[Path, Path]:
    report_dir = ROOT / "reports" / dataset
    ensure_dir(report_dir)
    md_path = report_dir / f"{dataset}_{stage_name}_dry_run_plan.md"
    json_path = report_dir / f"{dataset}_{stage_name}_dry_run_plan.json"
    summary = {
        "dataset_name": dataset,
        "stage_group": stage_group,
        "stage_name": stage_name,
        "planned_jobs": len(planned_jobs),
        "materialized": False,
    }
    lines = [
        f"# {dataset} {stage_name} dry-run plan",
        "",
        "This is a planning report only. No FoldX, Rosetta, ESM2, or OpenFold executor was launched.",
        "",
        f"- Stage group: `{stage_group}`",
        f"- Planned jobs: {len(planned_jobs)}",
        "",
        "## First planned jobs",
        "",
    ]
    for job in planned_jobs[:20]:
        lines.append(f"- `{job.get('job_id', '')}` / `{job.get('mutation_record_id', '')}`")
    if len(planned_jobs) > 20:
        lines.append(f"- ... {len(planned_jobs) - 20} more")
    safe_write_text(md_path, "\n".join(lines) + "\n", force=force)
    write_json(json_path, {"summary": summary, "jobs": planned_jobs}, force=force)
    return md_path, json_path


def materialize_plan(dataset: str, stage_name: str, stage_group: str, planned_jobs: list[dict[str, str]], force: bool) -> tuple[Path, Path]:
    manifest_rows: list[dict[str, Any]] = []
    for job in planned_jobs:
        mutation_record_id = job.get("mutation_record_id", "")
        method = job.get("method", "")
        structure_source = job.get("structure_source", "")
        job_dir = build_job_output_dir(dataset, method, structure_source, mutation_record_id, root=ROOT)
        ensure_dir(job_dir / "raw")
        command_path = job_dir / "command.txt"
        safe_write_text(command_path, command_placeholder(stage_name, job), force=force)
        parsed_path = parsed_result_path(dataset, method, structure_source, mutation_record_id)
        manifest = build_job_manifest(
            dataset_name=dataset,
            stage_group=stage_group,
            stage_name=stage_name,
            job=job,
            job_status="pending",
            input_files=[job.get("experimental_structure_path", ""), job.get("sequence_input_fasta_path", "")],
            output_files=[relative(command_path), f"results/{dataset}/{method}/{structure_source}/{mutation_record_id}/job_manifest.json"],
            parsed_result_path=parsed_path,
            notes="Materialized plan only. Real executor has not run.",
        )
        write_job_manifest(job_dir, manifest, force=force)
        manifest_rows.append(manifest)
    return write_stage_manifest(dataset, stage_name, manifest_rows, force=force, root=ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--mutation-record-id")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--materialize-plan", action="store_true")
    args = parser.parse_args(argv)

    if args.stage_name not in SUPPORTED_STAGES:
        raise SystemExit(f"Unsupported skeleton stage: {args.stage_name}. Supported: {', '.join(sorted(SUPPORTED_STAGES))}")
    run_stages = load_yaml(ROOT / "configs" / "run_stages.yaml")
    load_yaml(ROOT / "configs" / "execution_contract.yaml")
    stage_group = stage_group_for_stage(run_stages, args.stage_name)
    if not stage_group:
        raise SystemExit(f"Stage not found in configs/run_stages.yaml: {args.stage_name}")

    planned_jobs = build_plan(args.dataset, args.stage_name, max_jobs=args.max_jobs, mutation_record_id=args.mutation_record_id)
    print(f"Dataset: {args.dataset}")
    print(f"Stage: {args.stage_name}")
    print(f"Stage group: {stage_group}")
    print(f"Workers requested: {args.workers}")
    print(f"Planned jobs: {len(planned_jobs)}")
    for job in planned_jobs[:10]:
        print(f"  - {job.get('job_id', '')} ({job.get('mutation_record_id', '')})")
    if len(planned_jobs) > 10:
        print(f"  ... {len(planned_jobs) - 10} more")

    try:
        report_paths = write_reports(args.dataset, args.stage_name, stage_group, planned_jobs, force=True)
        plan_path = write_dry_run_plan(args.dataset, stage_group, args.stage_name, planned_jobs, force=True)
        print(f"Dry-run report: {relative(report_paths[0])}")
        print(f"Dry-run plan: {relative(plan_path)}")
        if args.materialize_plan:
            stage_manifest_paths = materialize_plan(args.dataset, args.stage_name, stage_group, planned_jobs, force=args.force)
            print(f"Materialized pending job manifests: {len(planned_jobs)}")
            print(f"Stage manifest: {relative(stage_manifest_paths[0])}")
            print("No parsed_result.json files were created.")
        else:
            print("No result directories were created. Use --materialize-plan to write placeholder manifests.")
    except FileExistsError as exc:
        raise SystemExit(f"{exc}\nUse --force to overwrite skeleton planning outputs.") from exc

    print("No completion marker was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
