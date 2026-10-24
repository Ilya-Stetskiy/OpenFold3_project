#!/usr/bin/env python3
"""FoldX executor preflight/materialization/run layer."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "run"))
sys.path.insert(0, str(ROOT / "scripts" / "executors"))

from execution_utils import (  # noqa: E402
    build_job_manifest,
    build_job_output_dir,
    ensure_dir,
    read_csv_dicts,
    safe_write_text,
    utc_now_iso,
    write_job_manifest,
    write_json,
    write_stage_manifest,
)
from foldx_adapter import (  # noqa: E402
    build_buildmodel_command,
    build_repair_pdb_command,
    load_foldx_config,
    parse_foldx_ddg,
    parsed_result_payload,
    validate_foldx_binary,
    validate_foldx_job_inputs,
    write_individual_list,
    write_parsed_result,
)


DEFAULT_STRUCTURE_SOURCE = "experimental"
DEFAULT_CONFIG = ROOT / "configs" / "tools" / "foldx.yaml"
EXAMPLE_CONFIG = ROOT / "configs" / "tools" / "foldx.example.yaml"


def stage_name_for(structure_source: str) -> str:
    return f"foldx_{structure_source}"


def stage_group_for(structure_source: str) -> str:
    return "post_gpu_cpu_heavy" if structure_source == "openfold" else "cpu_heavy"


def jobs_path(dataset: str, structure_source: str) -> Path:
    suffix = "openfold_ready" if structure_source == "openfold" else "resolved"
    return ROOT / "data" / "processed" / dataset / f"{dataset}_recalc_jobs_{suffix}.csv"


def structure_path_field(structure_source: str) -> str:
    return "openfold_structure_path" if structure_source == "openfold" else "experimental_structure_path"


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_config_for_mode(path: Path, allow_example_fallback: bool) -> tuple[dict[str, Any], list[str]]:
    warnings = []
    try:
        return load_foldx_config(path), warnings
    except FileNotFoundError:
        if not allow_example_fallback:
            raise
        warnings.append(f"Missing FoldX config: {relative(path)}; using example config for planning only")
        return load_foldx_config(EXAMPLE_CONFIG), warnings


def select_jobs(dataset: str, structure_source: str = DEFAULT_STRUCTURE_SOURCE, max_jobs: int | None = None, mutation_record_id: str | None = None) -> list[dict[str, str]]:
    path = jobs_path(dataset, structure_source)
    if not path.exists():
        hint = (
            "Run collect_openfold_outputs.py after OpenFold WT generation to create it."
            if structure_source == "openfold"
            else "Run current_internal normalization/override preparation to create it."
        )
        raise SystemExit(f"Missing FoldX jobs table: {relative(path)}. {hint}")
    rows = read_csv_dicts(path)
    rows = [
        row
        for row in rows
        if row.get("dataset_name", dataset) == dataset
        and row.get("method") == "foldx"
        and row.get("structure_source") == structure_source
        and str(row.get("input_ready", "")).lower() == "true"
    ]
    if mutation_record_id:
        rows = [row for row in rows if row.get("mutation_record_id") == mutation_record_id]
    if max_jobs is not None:
        rows = rows[:max_jobs]
    return rows


def job_dir(dataset: str, job: dict[str, str]) -> Path:
    return build_job_output_dir(dataset, "foldx", job.get("structure_source", DEFAULT_STRUCTURE_SOURCE), job["mutation_record_id"], root=ROOT)


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
    foldx_ddg: Any = "",
    parser_status: str = "",
    notes: str = "",
) -> dict[str, Any]:
    out_dir = job_dir(dataset, job)
    command_path = out_dir / "command.txt"
    individual_list = out_dir / "individual_list.txt"
    manifest = build_job_manifest(
        dataset_name=dataset,
        stage_group=stage_group_for(job.get("structure_source", DEFAULT_STRUCTURE_SOURCE)),
        stage_name=stage_name_for(job.get("structure_source", DEFAULT_STRUCTURE_SOURCE)),
        job=job,
        job_status=status,
        input_files=[job.get(structure_path_field(job.get("structure_source", DEFAULT_STRUCTURE_SOURCE)), ""), relative(individual_list)],
        output_files=[relative(command_path), relative(out_dir / "job_manifest.json")],
        parsed_result_path=relative(out_dir / "parsed_result.json"),
        error_message=error,
        notes=notes or f"FoldX {job.get('structure_source', DEFAULT_STRUCTURE_SOURCE)} execution record.",
    )
    manifest["started_at"] = started_at
    manifest["finished_at"] = finished_at
    manifest["runtime_seconds"] = runtime_seconds
    manifest["foldx_binary"] = config.get("foldx", {}).get("binary_path", "")
    row = dict(manifest)
    row.update({"output_dir": relative(out_dir), "foldx_ddg": foldx_ddg, "parser_status": parser_status})
    return row


def write_manifest_row(dataset: str, job: dict[str, str], row: dict[str, Any], force: bool = True) -> None:
    write_json(job_dir(dataset, job) / "job_manifest.json", row, force=force)


def preflight_jobs(dataset: str, jobs: list[dict[str, str]], config: dict[str, Any], force: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    counts = Counter()
    for job in jobs:
        ok, errors = validate_foldx_job_inputs(job, root=ROOT, structure_path_field=structure_path_field(job.get("structure_source", DEFAULT_STRUCTURE_SOURCE)))
        status = existing_status(dataset, job)
        if status == "existing_success" and not force:
            counts["existing_success"] += 1
            ok = False
            errors.append("existing_success_without_force")
        if status == "existing_failed":
            counts["existing_failed"] += 1
        if ok:
            counts["ready_for_execution"] += 1
        else:
            counts["blocked_preflight_failed"] += 1
        rows.append(
            {
                "dataset_name": dataset,
                "method": "foldx",
                "structure_source": job.get("structure_source", DEFAULT_STRUCTURE_SOURCE),
                "stage_name": stage_name_for(job.get("structure_source", DEFAULT_STRUCTURE_SOURCE)),
                "job_id": job.get("job_id", ""),
                "mutation_record_id": job.get("mutation_record_id", ""),
                "chain_id": job.get("chain_id", ""),
                "mutation": job.get("mutation", ""),
                "structure_path": job.get(structure_path_field(job.get("structure_source", DEFAULT_STRUCTURE_SOURCE)), ""),
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
        "number_of_runs": int(config.get("foldx", {}).get("number_of_runs", 3)),
    }
    return rows, summary


def write_preflight_report(dataset: str, rows: list[dict[str, Any]], summary: dict[str, Any], global_checks: dict[str, Any]) -> tuple[Path, Path]:
    report_dir = ROOT / "reports" / dataset
    ensure_dir(report_dir)
    source = str(global_checks.get("structure_source") or (rows[0].get("structure_source", DEFAULT_STRUCTURE_SOURCE) if rows else DEFAULT_STRUCTURE_SOURCE))
    md_path = report_dir / f"foldx_{source}_preflight.md"
    json_path = report_dir / f"foldx_{source}_preflight.json"
    lines = [
        f"# FoldX {source} preflight",
        "",
        f"This report validates FoldX configuration and selected {source} FoldX jobs. It does not run the full FoldX batch.",
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
    ensure_dir(out_dir / "raw")
    ensure_dir(out_dir / "foldx_work")
    input_structure = out_dir / "input_structure.pdb"
    source_structure = ROOT / job.get(structure_path_field(job.get("structure_source", DEFAULT_STRUCTURE_SOURCE)), "")
    if source_structure.exists() and (force or not input_structure.exists()):
        shutil.copy2(source_structure, input_structure)
    individual_list = out_dir / "individual_list.txt"
    try:
        individual_list = write_individual_list(out_dir, job.get("chain_id", ""), job.get("mutation", ""))
    except ValueError as exc:
        if not error:
            error = str(exc)
        safe_write_text(individual_list, f"# invalid FoldX mutation input: {exc}\n", force=True)
    work_input_structure = out_dir / "foldx_work" / input_structure.name
    work_individual_list = out_dir / "foldx_work" / individual_list.name
    if input_structure.exists() and (force or not work_input_structure.exists()):
        shutil.copy2(input_structure, work_input_structure)
    if individual_list.exists() and (force or not work_individual_list.exists()):
        shutil.copy2(individual_list, work_individual_list)
    number_of_runs = int(config.get("foldx", {}).get("number_of_runs", 3))
    repaired_name = f"RepairPDB_{input_structure.name}"
    commands = [
        " ".join(build_repair_pdb_command(config.get("foldx", {}).get("binary_path", "/path/to/foldx"), input_structure.name)),
        " ".join(build_buildmodel_command(config.get("foldx", {}).get("binary_path", "/path/to/foldx"), repaired_name, individual_list.name, number_of_runs)),
    ]
    command_path = out_dir / "command.txt"
    safe_write_text(command_path, "\n".join(commands) + "\n", force=force)
    row = build_manifest_row(
        dataset,
        job,
        config,
        status=status,
        error=error,
        notes=f"FoldX {job.get('structure_source', DEFAULT_STRUCTURE_SOURCE)} plan materialized; real FoldX may not have run yet.",
    )
    write_job_manifest(out_dir, row, force=force)
    return row


def run_one_job(dataset: str, job: dict[str, str], config: dict[str, Any], force: bool, resume: bool) -> dict[str, Any]:
    existing = existing_status(dataset, job)
    if existing == "existing_success" and not force:
        row = build_manifest_row(
            dataset,
            job,
            config,
            status="skipped",
            notes="Skipped because successful parsed_result already exists. Use --force to rerun.",
        )
        write_manifest_row(dataset, job, row)
        return row
    if existing == "existing_failed" and not force:
        row = build_manifest_row(
            dataset,
            job,
            config,
            status="skipped",
            notes="Skipped because previous failed/incomplete output exists. Use --force to retry.",
        )
        write_manifest_row(dataset, job, row)
        return row
    ok, errors = validate_foldx_job_inputs(job, root=ROOT, structure_path_field=structure_path_field(job.get("structure_source", DEFAULT_STRUCTURE_SOURCE)))
    if not ok:
        return materialize_job(dataset, job, config, force=True, status="blocked", error="; ".join(errors))
    out_dir = job_dir(dataset, job)
    start = time.time()
    started_at = utc_now_iso()
    materialize_job(dataset, job, config, force=True, status="running")
    binary = config.get("foldx", {}).get("binary_path", "")
    timeout = int(config.get("foldx", {}).get("timeout_seconds", 600))
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    work_dir = out_dir / "foldx_work"
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            repair_command = build_repair_pdb_command(binary, "input_structure.pdb")
            subprocess.run(repair_command, cwd=work_dir, stdout=stdout, stderr=stderr, timeout=timeout, check=True)
            repaired_name = find_repaired_pdb(work_dir, "input_structure.pdb")
            build_command = build_buildmodel_command(
                binary,
                repaired_name,
                "individual_list.txt",
                int(config.get("foldx", {}).get("number_of_runs", 3)),
            )
            subprocess.run(build_command, cwd=work_dir, stdout=stdout, stderr=stderr, timeout=timeout, check=True)
    except subprocess.TimeoutExpired as exc:
        return _finalize_failed(dataset, job, config, "timeout", f"timeout after {exc.timeout} seconds", start, started_at)
    except (subprocess.CalledProcessError, OSError) as exc:
        return _finalize_failed(dataset, job, config, "failed", str(exc), start, started_at)
    parse_result = parse_foldx_ddg(out_dir)
    number_of_runs = int(config.get("foldx", {}).get("number_of_runs", 3))
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
        foldx_ddg=parse_result.get("foldx_ddg"),
        parser_status=parse_result.get("parser_status", ""),
        notes="FoldX executed in isolated foldx_work directory.",
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
        notes="FoldX execution failed.",
    )
    write_manifest_row(dataset, job, row)
    return row


def find_repaired_pdb(work_dir: Path, input_name: str) -> str:
    candidates = sorted(work_dir.glob(f"*{Path(input_name).stem}*.pdb"))
    for candidate in candidates:
        if candidate.name != input_name:
            return candidate.name
    expected = work_dir / f"RepairPDB_{input_name}"
    if expected.exists():
        return expected.name
    return input_name


def write_stage_outputs(dataset: str, rows: list[dict[str, Any]], force: bool) -> tuple[Path, Path]:
    source = rows[0].get("structure_source", DEFAULT_STRUCTURE_SOURCE) if rows else DEFAULT_STRUCTURE_SOURCE
    return write_stage_manifest(dataset, stage_name_for(source), rows, force=force, root=ROOT)


def stage_manifest_paths(dataset: str, structure_source: str) -> tuple[Path, Path]:
    stage_name = stage_name_for(structure_source)
    manifest_dir = ROOT / "results" / dataset / "_manifests"
    return manifest_dir / f"{stage_name}_jobs.csv", manifest_dir / f"{stage_name}_jobs.json"


def ensure_stage_manifest_writable(dataset: str, structure_source: str, force: bool) -> None:
    if force:
        return
    existing = [path for path in stage_manifest_paths(dataset, structure_source) if path.exists()]
    if existing:
        formatted = ", ".join(relative(path) for path in existing)
        raise SystemExit(f"Existing stage manifest found: {formatted}. Use --force to overwrite.")


def create_stage_completion_marker(dataset: str, rows: list[dict[str, Any]], command: str, force: bool) -> Path:
    source = rows[0].get("structure_source", DEFAULT_STRUCTURE_SOURCE) if rows else DEFAULT_STRUCTURE_SOURCE
    stage_group = stage_group_for(source)
    stage_name = stage_name_for(source)
    marker = ROOT / "data" / "run_state" / dataset / f"{stage_group}.{stage_name}.completed.json"
    terminal = {"success", "failed", "skipped", "blocked", "parse_failed", "timeout", "cancelled"}
    statuses = Counter(str(row.get("job_status", "")) for row in rows)
    status = "completed" if set(statuses).issubset(terminal) and not (set(statuses) - {"success", "skipped"}) else "completed_with_failures"
    payload = {
        "dataset_name": dataset,
        "stage_group": stage_group,
        "stage_name": stage_name,
        "status": status,
        "started_at": "",
        "completed_at": utc_now_iso(),
        "host": "",
        "command": command,
        "outputs": [f"results/{dataset}/_manifests/{stage_name}_jobs.csv"],
        "summary": dict(statuses),
        "notes": f"Per-stage marker only. This is not the global {stage_group} completion marker.",
    }
    write_json(marker, payload, force=force)
    return marker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--structure-source", choices=["experimental", "openfold"], default=DEFAULT_STRUCTURE_SOURCE)
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
    jobs = select_jobs(args.dataset, structure_source=args.structure_source, max_jobs=args.max_jobs, mutation_record_id=args.mutation_record_id)
    binary_ok, binary_error = validate_foldx_binary(config.get("foldx", {}).get("binary_path", ""))
    stage_name = stage_name_for(args.structure_source)
    global_checks = {
        "config_path": relative(args.config),
        "config_exists": args.config.exists(),
        "binary_ok": binary_ok,
        "binary_error": binary_error,
        "jobs_csv": relative(jobs_path(args.dataset, args.structure_source)),
        "jobs_csv_exists": jobs_path(args.dataset, args.structure_source).exists(),
        "structure_source": args.structure_source,
        "selected_jobs": len(jobs),
        "warnings": config_warnings,
    }
    print(f"Dataset: {args.dataset}")
    print(f"Stage: {stage_name}")
    print(f"Structure source: {args.structure_source}")
    print(f"Selected jobs: {len(jobs)}")

    if args.dry_run or not any([args.materialize_plan, args.preflight, args.smoke_run, args.run]):
        print("Dry run only. FoldX was not executed.")
        return 0

    if args.preflight:
        rows, summary = preflight_jobs(args.dataset, jobs, config, force=args.force)
        md_path, _ = write_preflight_report(args.dataset, rows, summary, global_checks)
        print(f"Preflight report: {relative(md_path)}")
        print(f"Ready for execution: {summary['ready_for_execution']}")
        print(f"Blocked/preflight_failed: {summary['blocked_preflight_failed']}")
        return 0

    if args.materialize_plan:
        ensure_stage_manifest_writable(args.dataset, args.structure_source, force=args.force)
        rows = []
        for job in jobs:
            ok, errors = validate_foldx_job_inputs(job, root=ROOT, structure_path_field=structure_path_field(job.get("structure_source", DEFAULT_STRUCTURE_SOURCE)))
            rows.append(materialize_job(args.dataset, job, config, force=args.force, status="pending" if ok else "blocked", error="; ".join(errors)))
        csv_path, _ = write_stage_outputs(args.dataset, rows, force=args.force)
        print(f"Materialized jobs: {len(rows)}")
        print(f"Stage manifest: {relative(csv_path)}")
        print("FoldX was not executed. No completion marker was created.")
        return 0

    if args.smoke_run and args.max_jobs is None:
        jobs = jobs[:1]
    if (args.run or args.smoke_run) and not binary_ok:
        raise SystemExit(f"FoldX binary is not runnable: {binary_error}")
    if args.smoke_run and len(jobs) > 1:
        raise SystemExit("Smoke-run is limited; pass --max-jobs 1 or omit --max-jobs.")

    if not args.force:
        skipped_existing = [job for job in jobs if existing_status(args.dataset, job) != "none"]
        if skipped_existing:
            print(f"Existing outputs detected for {len(skipped_existing)} jobs; they will be skipped without --force.")

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_one_job, args.dataset, job, config, args.force, args.resume) for job in jobs]
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
