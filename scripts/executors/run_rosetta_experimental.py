#!/usr/bin/env python3
"""Rosetta experimental executor preflight/materialization/real-run layer."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "run"))
sys.path.insert(0, str(ROOT / "scripts" / "executors"))

from execution_utils import (  # noqa: E402
    build_job_manifest,
    build_job_output_dir,
    ensure_dir,
    filter_jobs_for_stage,
    read_csv_dicts,
    safe_write_text,
    utc_now_iso,
    write_job_manifest,
    write_json,
    write_stage_manifest,
)
from rosetta_adapter import (  # noqa: E402
    build_rosetta_command,
    load_rosetta_config,
    parse_rosetta_ddg,
    parsed_result_payload,
    validate_rosetta_binary,
    validate_rosetta_job_inputs,
    write_mutation_file,
    write_parsed_result,
)


STAGE_NAME = "rosetta_experimental"
STAGE_GROUP = "cpu_heavy"
DEFAULT_CONFIG = ROOT / "configs" / "tools" / "rosetta.yaml"
EXAMPLE_CONFIG = ROOT / "configs" / "tools" / "rosetta.example.yaml"


def resolved_jobs_path(dataset: str) -> Path:
    return ROOT / "data" / "processed" / dataset / f"{dataset}_recalc_jobs_resolved.csv"


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_config_for_mode(path: Path, allow_example_fallback: bool) -> tuple[dict[str, Any], list[str]]:
    warnings = []
    try:
        return load_rosetta_config(path), warnings
    except FileNotFoundError:
        if not allow_example_fallback:
            raise
        warnings.append(f"Missing Rosetta config: {relative(path)}; using example config for planning only")
        return load_rosetta_config(EXAMPLE_CONFIG), warnings


def select_jobs(dataset: str, max_jobs: int | None = None, mutation_record_id: str | None = None) -> list[dict[str, str]]:
    rows = read_csv_dicts(resolved_jobs_path(dataset))
    rows = [row for row in filter_jobs_for_stage(rows, STAGE_NAME) if row.get("dataset_name", dataset) == dataset]
    if mutation_record_id:
        rows = [row for row in rows if row.get("mutation_record_id") == mutation_record_id]
    if max_jobs is not None:
        rows = rows[:max_jobs]
    return rows


def job_dir(dataset: str, job: dict[str, str]) -> Path:
    return build_job_output_dir(dataset, "rosetta", "experimental", job["mutation_record_id"], root=ROOT)


def parsed_result_path(dataset: str, job: dict[str, str]) -> Path:
    return job_dir(dataset, job) / "parsed_result.json"


def existing_status(dataset: str, job: dict[str, str]) -> str:
    parsed = parsed_result_path(dataset, job)
    manifest = job_dir(dataset, job) / "job_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "existing_failed"
        if data.get("job_status") == "success" and parsed.exists():
            return "existing_success"
        if data.get("job_status") in {"failed", "parse_failed", "timeout", "running"}:
            return "existing_failed"
    return "none"


def build_manifest_row(
    dataset: str,
    job: dict[str, str],
    config: dict[str, Any],
    *,
    status: str,
    error: str = "",
    started_at: str = "",
    finished_at: str = "",
    runtime_seconds: float | str = "",
    rosetta_ddg: Any = "",
    parser_status: str = "",
    notes: str = "",
) -> dict[str, Any]:
    out_dir = job_dir(dataset, job)
    command_path = out_dir / "command.txt"
    mutation_file = out_dir / "mutation_file.txt"
    manifest = build_job_manifest(
        dataset_name=dataset,
        stage_group=STAGE_GROUP,
        stage_name=STAGE_NAME,
        job=job,
        job_status=status,
        input_files=[job.get("experimental_structure_path", ""), relative(mutation_file)],
        output_files=[relative(command_path), relative(out_dir / "job_manifest.json")],
        parsed_result_path=relative(out_dir / "parsed_result.json"),
        error_message=error,
        notes=notes or "Rosetta experimental execution record.",
    )
    manifest["started_at"] = started_at
    manifest["finished_at"] = finished_at
    manifest["runtime_seconds"] = runtime_seconds
    manifest["rosetta_binary"] = config.get("rosetta", {}).get("binary_path", "")
    row = dict(manifest)
    row.update({"output_dir": relative(out_dir), "rosetta_ddg": rosetta_ddg, "parser_status": parser_status})
    return row


def write_manifest_row(dataset: str, job: dict[str, str], row: dict[str, Any], force: bool = True) -> None:
    write_json(job_dir(dataset, job) / "job_manifest.json", row, force=force)


def preflight_jobs(dataset: str, jobs: list[dict[str, str]], config: dict[str, Any], force: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    counts = Counter()
    for job in jobs:
        ok, errors = validate_rosetta_job_inputs(job, root=ROOT)
        status = existing_status(dataset, job)
        if status == "existing_success" and not force:
            counts["existing_success"] += 1
            ok = False
            errors.append("existing_success_without_force")
        if status == "existing_failed":
            counts["existing_failed"] += 1
        counts["ready_for_execution" if ok else "blocked_preflight_failed"] += 1
        rows.append(
            {
                "dataset_name": dataset,
                "job_id": job.get("job_id", ""),
                "mutation_record_id": job.get("mutation_record_id", ""),
                "chain_id": job.get("chain_id", ""),
                "mutation": job.get("mutation", ""),
                "experimental_structure_path": job.get("experimental_structure_path", ""),
                "output_dir": relative(job_dir(dataset, job)),
                "preflight_status": "ready" if ok else "preflight_failed",
                "existing_status": status,
                "error_message": "; ".join(errors),
            }
        )
    summary = {
        "total_selected_jobs": len(jobs),
        "ready_for_execution": counts["ready_for_execution"],
        "blocked_preflight_failed": counts["blocked_preflight_failed"],
        "existing_success": counts["existing_success"],
        "existing_failed": counts["existing_failed"],
        "number_of_runs": int(config.get("rosetta", {}).get("number_of_runs", 3)),
    }
    return rows, summary


def write_preflight_report(dataset: str, rows: list[dict[str, Any]], summary: dict[str, Any], global_checks: dict[str, Any]) -> tuple[Path, Path]:
    report_dir = ROOT / "reports" / dataset
    ensure_dir(report_dir)
    md_path = report_dir / "rosetta_experimental_preflight.md"
    json_path = report_dir / "rosetta_experimental_preflight.json"
    lines = [
        "# Rosetta experimental preflight",
        "",
        "This report validates Rosetta configuration and selected experimental Rosetta jobs. It does not run the full Rosetta batch.",
        "",
        "## Global checks",
        "",
    ]
    for key, value in global_checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Summary", ""])
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## First errors", ""])
    errors = [row for row in rows if row.get("error_message")]
    if not errors:
        lines.append("- none")
    for row in errors[:20]:
        lines.append(f"- `{row['mutation_record_id']}`: {row['error_message']}")
    safe_write_text(md_path, "\n".join(lines) + "\n", force=True)
    write_json(json_path, {"global_checks": global_checks, "summary": summary, "jobs": rows}, force=True)
    return md_path, json_path


def materialize_job(dataset: str, job: dict[str, str], config: dict[str, Any], force: bool, status: str = "pending", error: str = "") -> dict[str, Any]:
    out_dir = job_dir(dataset, job)
    work_dir = out_dir / "rosetta_work"
    ensure_dir(out_dir / "raw")
    ensure_dir(work_dir)
    input_structure = out_dir / "input_structure.pdb"
    source_structure = ROOT / job.get("experimental_structure_path", "")
    if source_structure.exists() and (force or not input_structure.exists()):
        shutil.copy2(source_structure, input_structure)
    mutation_file = out_dir / "mutation_file.txt"
    try:
        mutation_file = write_mutation_file(out_dir, job.get("chain_id", ""), job.get("mutation", ""))
    except ValueError as exc:
        if not error:
            error = str(exc)
        safe_write_text(mutation_file, f"# invalid Rosetta mutation input: {exc}\n", force=True)
    work_input_structure = work_dir / input_structure.name
    work_mutation_file = work_dir / mutation_file.name
    if input_structure.exists() and (force or not work_input_structure.exists()):
        shutil.copy2(input_structure, work_input_structure)
    if mutation_file.exists() and (force or not work_mutation_file.exists()):
        shutil.copy2(mutation_file, work_mutation_file)
    number_of_runs = int(config.get("rosetta", {}).get("number_of_runs", 3))
    extra_args = list(config.get("rosetta", {}).get("extra_args", []) or [])
    command = build_rosetta_command(
        config.get("rosetta", {}).get("binary_path", "/path/to/rosetta/ddg_monomer"),
        input_structure.name,
        mutation_file.name,
        "score.sc",
        number_of_runs,
        extra_args=extra_args,
    )
    command_path = out_dir / "command.txt"
    safe_write_text(command_path, " ".join(command) + "\n", force=force)
    row = build_manifest_row(
        dataset,
        job,
        config,
        status=status,
        error=error,
        notes="Rosetta experimental plan materialized; real Rosetta may not have run yet.",
    )
    write_job_manifest(out_dir, row, force=force)
    return row


def run_one_job(dataset: str, job: dict[str, str], config: dict[str, Any], force: bool) -> dict[str, Any]:
    existing = existing_status(dataset, job)
    if existing == "existing_success" and not force:
        row = build_manifest_row(dataset, job, config, status="skipped", notes="Skipped because successful parsed_result already exists. Use --force to rerun.")
        write_manifest_row(dataset, job, row)
        return row
    if existing == "existing_failed" and not force:
        row = build_manifest_row(dataset, job, config, status="skipped", notes="Skipped because previous failed/incomplete output exists. Use --force to retry.")
        write_manifest_row(dataset, job, row)
        return row
    ok, errors = validate_rosetta_job_inputs(job, root=ROOT)
    if not ok:
        return materialize_job(dataset, job, config, force=True, status="blocked", error="; ".join(errors))
    out_dir = job_dir(dataset, job)
    work_dir = out_dir / "rosetta_work"
    started_at = utc_now_iso()
    start = time.time()
    materialize_job(dataset, job, config, force=True, status="running")
    timeout = int(config.get("rosetta", {}).get("timeout_seconds", 1800))
    command = (out_dir / "command.txt").read_text(encoding="utf-8").split()
    try:
        with (out_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (out_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
            subprocess.run(command, cwd=work_dir, stdout=stdout, stderr=stderr, timeout=timeout, check=True)
    except subprocess.TimeoutExpired as exc:
        return _finalize_failed(dataset, job, config, "timeout", f"timeout after {exc.timeout} seconds", start, started_at)
    except (subprocess.CalledProcessError, OSError) as exc:
        return _finalize_failed(dataset, job, config, "failed", str(exc), start, started_at)
    parse_result = parse_rosetta_ddg(out_dir)
    number_of_runs = int(config.get("rosetta", {}).get("number_of_runs", 3))
    write_parsed_result(out_dir / "parsed_result.json", parsed_result_payload(dataset, job, parse_result, number_of_runs))
    status = "success" if parse_result["parser_status"] == "success" else "parse_failed"
    row = build_manifest_row(
        dataset,
        job,
        config,
        status=status,
        error=parse_result.get("error_message", ""),
        started_at=started_at,
        finished_at=utc_now_iso(),
        runtime_seconds=round(time.time() - start, 3),
        rosetta_ddg=parse_result.get("rosetta_ddg"),
        parser_status=parse_result.get("parser_status", ""),
        notes="Rosetta executed in isolated rosetta_work directory.",
    )
    write_manifest_row(dataset, job, row)
    return row


def _finalize_failed(dataset: str, job: dict[str, str], config: dict[str, Any], status: str, error: str, start: float, started_at: str) -> dict[str, Any]:
    row = build_manifest_row(
        dataset,
        job,
        config,
        status=status,
        error=error,
        started_at=started_at,
        finished_at=utc_now_iso(),
        runtime_seconds=round(time.time() - start, 3),
        notes="Rosetta execution failed.",
    )
    write_manifest_row(dataset, job, row)
    return row


def write_stage_outputs(dataset: str, rows: list[dict[str, Any]], force: bool) -> tuple[Path, Path]:
    return write_stage_manifest(dataset, STAGE_NAME, rows, force=force, root=ROOT)


def create_stage_completion_marker(dataset: str, rows: list[dict[str, Any]], command: str, force: bool) -> Path:
    marker = ROOT / "data" / "run_state" / dataset / f"{STAGE_GROUP}.{STAGE_NAME}.completed.json"
    terminal = {"success", "failed", "skipped", "blocked", "parse_failed", "timeout", "cancelled"}
    statuses = Counter(str(row.get("job_status", "")) for row in rows)
    status = "completed" if set(statuses).issubset(terminal) and not (set(statuses) - {"success", "skipped"}) else "completed_with_failures"
    payload = {
        "dataset_name": dataset,
        "stage_group": STAGE_GROUP,
        "stage_name": STAGE_NAME,
        "status": status,
        "started_at": "",
        "completed_at": utc_now_iso(),
        "host": "",
        "command": command,
        "outputs": [f"results/{dataset}/_manifests/{STAGE_NAME}_jobs.csv"],
        "summary": dict(statuses),
        "notes": "Per-stage marker only. This is not the global cpu_heavy completion marker.",
    }
    write_json(marker, payload, force=force)
    return marker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--materialize-plan", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--mutation-record-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args(argv)

    allow_example = not (args.run or args.smoke_run)
    config, config_warnings = load_config_for_mode(args.config, allow_example_fallback=allow_example)
    jobs = select_jobs(args.dataset, max_jobs=args.max_jobs, mutation_record_id=args.mutation_record_id)
    binary_ok, binary_error = validate_rosetta_binary(config.get("rosetta", {}).get("binary_path", ""))
    global_checks = {
        "config_path": relative(args.config),
        "config_exists": args.config.exists(),
        "binary_ok": binary_ok,
        "binary_error": binary_error,
        "resolved_jobs_csv_exists": resolved_jobs_path(args.dataset).exists(),
        "selected_jobs": len(jobs),
        "warnings": config_warnings,
    }
    print(f"Dataset: {args.dataset}")
    print(f"Stage: {STAGE_NAME}")
    print(f"Selected jobs: {len(jobs)}")

    if args.dry_run or not any([args.materialize_plan, args.preflight, args.smoke_run, args.run]):
        print("Dry run only. Rosetta was not executed.")
        return 0

    if args.preflight:
        rows, summary = preflight_jobs(args.dataset, jobs, config, force=args.force)
        md_path, _ = write_preflight_report(args.dataset, rows, summary, global_checks)
        print(f"Preflight report: {relative(md_path)}")
        print(f"Ready for execution: {summary['ready_for_execution']}")
        print(f"Blocked/preflight_failed: {summary['blocked_preflight_failed']}")
        return 0

    if args.materialize_plan:
        rows = []
        for job in jobs:
            ok, errors = validate_rosetta_job_inputs(job, root=ROOT)
            rows.append(materialize_job(args.dataset, job, config, force=args.force, status="pending" if ok else "blocked", error="; ".join(errors)))
        csv_path, _ = write_stage_outputs(args.dataset, rows, force=True)
        print(f"Materialized jobs: {len(rows)}")
        print(f"Stage manifest: {relative(csv_path)}")
        print("Rosetta was not executed. No completion marker was created.")
        return 0

    if args.smoke_run and args.max_jobs is None:
        jobs = jobs[:1]
    if (args.run or args.smoke_run) and not binary_ok:
        raise SystemExit(f"Rosetta binary is not runnable: {binary_error}")
    if args.smoke_run and len(jobs) > 1:
        raise SystemExit("Smoke-run is limited; pass --max-jobs 1 or omit --max-jobs.")
    if not args.force:
        skipped_existing = [job for job in jobs if existing_status(args.dataset, job) != "none"]
        if skipped_existing:
            print(f"Existing outputs detected for {len(skipped_existing)} jobs; they will be skipped without --force.")

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_one_job, args.dataset, job, config, args.force) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    write_stage_outputs(args.dataset, rows, force=True)
    marker = create_stage_completion_marker(args.dataset, rows, " ".join(sys.argv), force=True)
    print(f"Executed jobs: {len(rows)}")
    print(f"Per-stage completion marker: {relative(marker)}")
    print("Global cpu_heavy.completed.json was not created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

