#!/usr/bin/env python3
"""ESM2 GPU executor preflight/materialization/real-run layer."""

from __future__ import annotations

import argparse
import json
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

from esm2_adapter import (  # noqa: E402
    build_esm2_score_command,
    find_sequence_for_job,
    load_esm2_config,
    parse_esm2_result,
    parsed_result_payload,
    validate_esm2_job_inputs,
    validate_python_bin,
    write_job_input_json,
)
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


STAGE_NAME = "esm2_gpu_optional"
STAGE_GROUP = "gpu_heavy"
DEFAULT_CONFIG = ROOT / "configs" / "tools" / "esm2.yaml"
EXAMPLE_CONFIG = ROOT / "configs" / "tools" / "esm2.example.yaml"
DEFAULT_SCORER = ROOT / "scripts" / "executors" / "esm2_score_single.py"


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
        return load_esm2_config(path), warnings
    except FileNotFoundError:
        if not allow_example_fallback:
            raise
        warnings.append(f"Missing ESM2 config: {relative(path)}; using example config for planning only")
        return load_esm2_config(EXAMPLE_CONFIG), warnings


def select_jobs(dataset: str, max_jobs: int | None = None, mutation_record_id: str | None = None) -> list[dict[str, str]]:
    rows = read_csv_dicts(resolved_jobs_path(dataset))
    rows = [row for row in filter_jobs_for_stage(rows, STAGE_NAME) if row.get("dataset_name", dataset) == dataset]
    if mutation_record_id:
        rows = [row for row in rows if row.get("mutation_record_id") == mutation_record_id]
    if max_jobs is not None:
        rows = rows[:max_jobs]
    return rows


def job_dir(dataset: str, job: dict[str, str]) -> Path:
    return build_job_output_dir(dataset, "esm2", "sequence_only", job["mutation_record_id"], root=ROOT)


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
    esm2_score: Any = "",
    parser_status: str = "",
    notes: str = "",
) -> dict[str, Any]:
    out_dir = job_dir(dataset, job)
    command_path = out_dir / "command.txt"
    input_json = out_dir / "job_input.json"
    manifest = build_job_manifest(
        dataset_name=dataset,
        stage_group=STAGE_GROUP,
        stage_name=STAGE_NAME,
        job=job,
        job_status=status,
        input_files=[job.get("sequence_input_fasta_path", ""), relative(input_json)],
        output_files=[relative(command_path), relative(out_dir / "job_manifest.json")],
        parsed_result_path=relative(out_dir / "parsed_result.json"),
        error_message=error,
        notes=notes or "ESM2 GPU execution record.",
    )
    manifest["started_at"] = started_at
    manifest["finished_at"] = finished_at
    manifest["runtime_seconds"] = runtime_seconds
    manifest["esm2_model_name"] = config.get("esm2", {}).get("model_name", "")
    manifest["esm2_device"] = config.get("esm2", {}).get("device", "")
    row = dict(manifest)
    row.update({"output_dir": relative(out_dir), "esm2_score": esm2_score, "parser_status": parser_status})
    return row


def write_manifest_row(dataset: str, job: dict[str, str], row: dict[str, Any], force: bool = True) -> None:
    write_json(job_dir(dataset, job) / "job_manifest.json", row, force=force)


def preflight_jobs(dataset: str, jobs: list[dict[str, str]], config: dict[str, Any], force: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    counts = Counter()
    for job in jobs:
        ok, errors = validate_esm2_job_inputs(job, root=ROOT)
        status = existing_status(dataset, job)
        if status == "existing_success" and not force:
            counts["existing_success"] += 1
            ok = False
            errors.append("existing_success_without_force")
        if status == "existing_failed":
            counts["existing_failed"] += 1
        if ok:
            try:
                find_sequence_for_job(job, ROOT / job.get("sequence_input_fasta_path", ""))
            except (OSError, ValueError) as exc:
                ok = False
                errors.append(str(exc))
        counts["ready_for_execution" if ok else "blocked_preflight_failed"] += 1
        rows.append(
            {
                "dataset_name": dataset,
                "job_id": job.get("job_id", ""),
                "mutation_record_id": job.get("mutation_record_id", ""),
                "pdb_id": job.get("pdb_id", ""),
                "chain_id": job.get("chain_id", ""),
                "mutation": job.get("mutation", ""),
                "sequence_input_fasta_path": job.get("sequence_input_fasta_path", ""),
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
        "model_name": config.get("esm2", {}).get("model_name", ""),
        "device": config.get("esm2", {}).get("device", ""),
    }
    return rows, summary


def write_preflight_report(dataset: str, rows: list[dict[str, Any]], summary: dict[str, Any], global_checks: dict[str, Any]) -> tuple[Path, Path]:
    report_dir = ROOT / "reports" / dataset
    ensure_dir(report_dir)
    md_path = report_dir / "esm2_gpu_preflight.md"
    json_path = report_dir / "esm2_gpu_preflight.json"
    lines = [
        "# ESM2 GPU preflight",
        "",
        "This report validates ESM2 configuration and selected sequence-only jobs. It does not run ESM2 scoring.",
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


def materialize_job(
    dataset: str,
    job: dict[str, str],
    config: dict[str, Any],
    scorer_script: Path,
    force: bool,
    status: str = "pending",
    error: str = "",
) -> dict[str, Any]:
    out_dir = job_dir(dataset, job)
    work_dir = out_dir / "esm2_work"
    ensure_dir(out_dir / "raw")
    ensure_dir(work_dir)
    input_json = out_dir / "job_input.json"
    try:
        input_json = write_job_input_json(out_dir, job, job.get("sequence_input_fasta_path", ""), root=ROOT, force=force)
    except FileExistsError:
        raise
    except (OSError, ValueError) as exc:
        if not error:
            error = str(exc)
        safe_write_text(input_json, json.dumps({"error": str(exc)}, indent=2) + "\n", force=True)
    command = build_esm2_score_command(
        config.get("esm2", {}).get("python_bin", "python3"),
        scorer_script,
        input_json,
        out_dir / "parsed_result.json",
        config,
    )
    safe_write_text(out_dir / "command.txt", " ".join(command) + "\n", force=force)
    row = build_manifest_row(
        dataset,
        job,
        config,
        status=status,
        error=error,
        notes="ESM2 GPU plan materialized; real ESM2 may not have run yet.",
    )
    write_job_manifest(out_dir, row, force=force)
    return row


def run_one_job(dataset: str, job: dict[str, str], config: dict[str, Any], scorer_script: Path, force: bool) -> dict[str, Any]:
    existing = existing_status(dataset, job)
    if existing == "existing_success" and not force:
        row = build_manifest_row(dataset, job, config, status="skipped", notes="Skipped because successful parsed_result already exists. Use --force to rerun.")
        write_manifest_row(dataset, job, row)
        return row
    if existing == "existing_failed" and not force:
        row = build_manifest_row(dataset, job, config, status="skipped", notes="Skipped because previous failed/incomplete output exists. Use --force to retry.")
        write_manifest_row(dataset, job, row)
        return row
    ok, errors = validate_esm2_job_inputs(job, root=ROOT)
    if not ok:
        return materialize_job(dataset, job, config, scorer_script, force=True, status="blocked", error="; ".join(errors))
    out_dir = job_dir(dataset, job)
    work_dir = out_dir / "esm2_work"
    started_at = utc_now_iso()
    start = time.time()
    materialize_job(dataset, job, config, scorer_script, force=True, status="running")
    timeout = int(config.get("esm2", {}).get("timeout_seconds", 900))
    command = (out_dir / "command.txt").read_text(encoding="utf-8").split()
    try:
        with (out_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (out_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
            subprocess.run(command, cwd=work_dir, stdout=stdout, stderr=stderr, timeout=timeout, check=True)
    except subprocess.TimeoutExpired as exc:
        return _finalize_failed(dataset, job, config, "timeout", f"timeout after {exc.timeout} seconds", start, started_at)
    except (subprocess.CalledProcessError, OSError) as exc:
        return _finalize_failed(dataset, job, config, "failed", str(exc), start, started_at)
    parse_result = parse_esm2_result(out_dir)
    if parse_result.get("parser_status") != "success":
        payload = parsed_result_payload(dataset, job, parse_result, config)
        write_json(out_dir / "parsed_result.json", payload, force=True)
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
        esm2_score=parse_result.get("esm2_score"),
        parser_status=parse_result.get("parser_status", ""),
        notes="ESM2 scoring executed in isolated esm2_work directory.",
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
        notes="ESM2 execution failed.",
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
        "notes": "Per-stage marker only. This is not the global gpu_heavy completion marker.",
    }
    write_json(marker, payload, force=force)
    return marker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scorer-script", type=Path, default=DEFAULT_SCORER)
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
    python_ok, python_error = validate_python_bin(config.get("esm2", {}).get("python_bin", ""))
    global_checks = {
        "config_path": relative(args.config),
        "config_exists": args.config.exists(),
        "python_ok": python_ok,
        "python_error": python_error,
        "scorer_script_exists": args.scorer_script.exists(),
        "resolved_jobs_csv_exists": resolved_jobs_path(args.dataset).exists(),
        "selected_jobs": len(jobs),
        "warnings": config_warnings,
    }
    print(f"Dataset: {args.dataset}")
    print(f"Stage: {STAGE_NAME}")
    print(f"Selected jobs: {len(jobs)}")

    if args.dry_run or not any([args.materialize_plan, args.preflight, args.smoke_run, args.run]):
        print("Dry run only. ESM2 was not executed.")
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
            ok, errors = validate_esm2_job_inputs(job, root=ROOT)
            rows.append(materialize_job(args.dataset, job, config, args.scorer_script, force=args.force, status="pending" if ok else "blocked", error="; ".join(errors)))
        csv_path, _ = write_stage_outputs(args.dataset, rows, force=True)
        print(f"Materialized jobs: {len(rows)}")
        print(f"Stage manifest: {relative(csv_path)}")
        print("ESM2 was not executed. No completion marker was created.")
        return 0

    if args.smoke_run and args.max_jobs is None:
        jobs = jobs[:1]
    if (args.run or args.smoke_run) and not python_ok:
        raise SystemExit(f"ESM2 python executable is not runnable: {python_error}")
    if (args.run or args.smoke_run) and not args.scorer_script.exists():
        raise SystemExit(f"ESM2 scorer script does not exist: {args.scorer_script}")
    if args.smoke_run and len(jobs) > 1:
        raise SystemExit("Smoke-run is limited; pass --max-jobs 1 or omit --max-jobs.")
    if not args.force:
        skipped_existing = [job for job in jobs if existing_status(args.dataset, job) != "none"]
        if skipped_existing:
            print(f"Existing outputs detected for {len(skipped_existing)} jobs; they will be skipped without --force.")

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_one_job, args.dataset, job, config, args.scorer_script, args.force) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    write_stage_outputs(args.dataset, rows, force=True)
    marker = create_stage_completion_marker(args.dataset, rows, " ".join(sys.argv), force=True)
    print(f"Executed jobs: {len(rows)}")
    print(f"Per-stage completion marker: {relative(marker)}")
    print("Global gpu_heavy.completed.json was not created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
