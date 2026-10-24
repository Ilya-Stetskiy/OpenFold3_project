#!/usr/bin/env python3
"""OpenFold WT generation executor planning layer."""

from __future__ import annotations

import argparse
import json
import os
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
    ensure_dir,
    safe_write_text,
    utc_now_iso,
    write_job_manifest,
    write_json,
    write_stage_manifest,
)
from openfold_adapter import (  # noqa: E402
    batch_sequence_rows,
    build_openfold_predict_command,
    load_openfold_config,
    normalize_openfold_outputs,
    read_sequence_manifest,
    select_sequence_rows,
    validate_openfold_import,
    validate_python_bin,
    validate_sequence_row,
    write_query_json,
)


STAGE_NAME = "openfold_wt_generation"
STAGE_GROUP = "gpu_heavy"
DEFAULT_CONFIG = ROOT / "configs" / "tools" / "openfold.yaml"
EXAMPLE_CONFIG = ROOT / "configs" / "tools" / "openfold.example.yaml"


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_config_for_mode(path: Path, allow_example_fallback: bool) -> tuple[dict[str, Any], list[str]]:
    warnings = []
    try:
        return load_openfold_config(path), warnings
    except FileNotFoundError:
        if not allow_example_fallback:
            raise
        warnings.append(f"Missing OpenFold config: {relative(path)}; using example config for planning only")
        return load_openfold_config(EXAMPLE_CONFIG), warnings


def resolve_config_path(value: str | None, root: Path = ROOT) -> Path | None:
    if value in {None, ""}:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def sequence_manifest_path(config: dict[str, Any]) -> Path:
    return resolve_config_path(config.get("paths", {}).get("sequence_manifest"), ROOT) or ROOT / "data/manifests/current_internal_sequence_manifest.csv"


def select_jobs(
    dataset: str,
    config: dict[str, Any],
    *,
    max_sequences: int | None = None,
    sequence_id: str | None = None,
    pdb_id: str | None = None,
) -> list[dict[str, str]]:
    rows = read_sequence_manifest(sequence_manifest_path(config))
    return select_sequence_rows(rows, dataset=dataset, max_sequences=max_sequences, sequence_id=sequence_id, pdb_id=pdb_id)


def job_dir(dataset: str, job: dict[str, str]) -> Path:
    return ROOT / "results" / dataset / "openfold" / "wt" / job["sequence_id"]


def batch_dir(dataset: str, batch_index: int) -> Path:
    return ROOT / "results" / dataset / "openfold" / "wt" / "_batches" / f"batch_{batch_index:04d}"


def model_path(dataset: str, job: dict[str, str]) -> Path:
    return job_dir(dataset, job) / "model.cif"


def confidence_path(dataset: str, job: dict[str, str]) -> Path:
    return job_dir(dataset, job) / "confidence.json"


def existing_status(dataset: str, job: dict[str, str]) -> str:
    manifest = job_dir(dataset, job) / "job_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "existing_failed"
        if data.get("job_status") == "success" and model_path(dataset, job).exists():
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
    batch_id: str = "",
    started_at: str = "",
    finished_at: str = "",
    runtime_seconds: float | str = "",
    notes: str = "",
) -> dict[str, Any]:
    out_dir = job_dir(dataset, job)
    openfold = config.get("openfold", {})
    manifest = build_job_manifest(
        dataset_name=dataset,
        stage_group=STAGE_GROUP,
        stage_name=STAGE_NAME,
        job={
            "job_id": f"{dataset}__openfold_wt__{job.get('sequence_id', '')}",
            "mutation_record_id": "",
            "method": "openfold",
            "structure_source": "wt",
            "input_ready": True,
        },
        job_status=status,
        input_files=[relative(sequence_manifest_path(config))],
        output_files=[relative(out_dir / "model.cif"), relative(out_dir / "model.pdb"), relative(out_dir / "confidence.json"), relative(out_dir / "job_manifest.json")],
        parsed_result_path=relative(out_dir / "openfold_result.json"),
        error_message=error,
        notes=notes or "OpenFold WT generation record.",
    )
    manifest.update(
        {
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime_seconds": runtime_seconds,
            "sequence_id": job.get("sequence_id", ""),
            "uniprot_id": job.get("uniprot_id", ""),
            "protein_id": job.get("protein_id", ""),
            "pdb_id": job.get("pdb_id", ""),
            "chain_id": job.get("chain_id", ""),
            "sequence_length": job.get("sequence_length", ""),
            "batch_id": batch_id,
            "openfold_python": openfold.get("python_bin", ""),
            "openfold_repo_dir": openfold.get("openfold_repo_dir", ""),
            "model_path": relative(out_dir / "model.cif"),
            "pdb_path": relative(out_dir / "model.pdb"),
            "confidence_path": relative(out_dir / "confidence.json"),
            "output_dir": relative(out_dir),
        }
    )
    return manifest


def preflight_jobs(dataset: str, jobs: list[dict[str, str]], config: dict[str, Any], force: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    counts = Counter()
    for job in jobs:
        ok, errors = validate_sequence_row(job)
        status = existing_status(dataset, job)
        if status == "existing_success" and not force:
            ok = False
            errors.append("existing_success_without_force")
            counts["existing_success"] += 1
        if status == "existing_failed":
            counts["existing_failed"] += 1
        counts["ready_for_execution" if ok else "blocked_preflight_failed"] += 1
        rows.append(
            {
                "dataset_name": dataset,
                "sequence_id": job.get("sequence_id", ""),
                "pdb_id": job.get("pdb_id", ""),
                "chain_id": job.get("chain_id", ""),
                "sequence_length": job.get("sequence_length", ""),
                "output_dir": relative(job_dir(dataset, job)),
                "preflight_status": "ready" if ok else "preflight_failed",
                "existing_status": status,
                "error_message": "; ".join(errors),
            }
        )
    return rows, {
        "total_selected_sequences": len(jobs),
        "ready_for_execution": counts["ready_for_execution"],
        "blocked_preflight_failed": counts["blocked_preflight_failed"],
        "existing_success": counts["existing_success"],
        "existing_failed": counts["existing_failed"],
        "batch_size": int(config.get("openfold", {}).get("batch_size", 1)),
    }


def write_preflight_report(dataset: str, rows: list[dict[str, Any]], summary: dict[str, Any], global_checks: dict[str, Any]) -> tuple[Path, Path]:
    report_dir = ROOT / "reports" / dataset
    ensure_dir(report_dir)
    md_path = report_dir / "openfold_wt_preflight.md"
    json_path = report_dir / "openfold_wt_preflight.json"
    lines = [
        "# OpenFold WT preflight",
        "",
        "This report validates OpenFold WT sequence targets and configuration. It does not run OpenFold.",
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
        lines.append(f"- `{row['sequence_id']}`: {row['error_message']}")
    safe_write_text(md_path, "\n".join(lines) + "\n", force=True)
    write_json(json_path, {"global_checks": global_checks, "summary": summary, "jobs": rows}, force=True)
    return md_path, json_path


def materialize_batches(dataset: str, jobs: list[dict[str, str]], config: dict[str, Any], force: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    openfold = config.get("openfold", {})
    batch_size = int(openfold.get("batch_size", 1))
    batches = batch_sequence_rows(jobs, batch_size)
    for batch_index, batch_rows in enumerate(batches, start=1):
        rows.extend(materialize_batch(dataset, batch_index, batch_rows, config, force=force))
    return rows


def materialize_batch(dataset: str, batch_index: int, batch_rows: list[dict[str, str]], config: dict[str, Any], force: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    openfold = config.get("openfold", {})
    batch_id = f"batch_{batch_index:04d}"
    bdir = batch_dir(dataset, batch_index)
    ensure_dir(bdir / "openfold_output")
    query_json = write_query_json(
        bdir / "query.json",
        batch_rows,
        use_msas=bool(openfold.get("use_msa_server", False)),
        use_templates=bool(openfold.get("use_templates", False)),
        force=force,
    )
    command = build_openfold_predict_command(
        openfold.get("python_bin", "python3"),
        query_json,
        bdir / "openfold_output",
        config,
    )
    safe_write_text(bdir / "command.txt", json.dumps(command, ensure_ascii=False) + "\n", force=force)
    write_json(
        bdir / "batch_manifest.json",
        {
            "dataset_name": dataset,
            "stage_group": STAGE_GROUP,
            "stage_name": STAGE_NAME,
            "batch_id": batch_id,
            "sequence_ids": [row["sequence_id"] for row in batch_rows],
            "query_json": relative(query_json),
            "output_dir": relative(bdir / "openfold_output"),
            "job_status": "pending",
            "created_at": utc_now_iso(),
        },
        force=force,
    )
    for job in batch_rows:
        out_dir = job_dir(dataset, job)
        ensure_dir(out_dir)
        row = build_manifest_row(dataset, job, config, status="pending", batch_id=batch_id, notes="OpenFold WT plan materialized; OpenFold was not run.")
        write_job_manifest(out_dir, row, force=force)
        rows.append(row)
    return rows


def run_batch(dataset: str, batch_index: int, batch_rows: list[dict[str, str]], config: dict[str, Any], force: bool) -> list[dict[str, Any]]:
    materialize_batch(dataset, batch_index, batch_rows, config, force=True)
    bdir = batch_dir(dataset, batch_index)
    command = json.loads((bdir / "command.txt").read_text(encoding="utf-8"))
    repo_dir = resolve_config_path(config.get("openfold", {}).get("openfold_repo_dir"), ROOT) or ROOT / "openfold-3"
    env = _openfold_env(repo_dir)
    started_at = utc_now_iso()
    start = time.time()
    timeout = int(config.get("openfold", {}).get("timeout_seconds", 7200))
    try:
        with (bdir / "stdout.log").open("w", encoding="utf-8") as stdout, (bdir / "stderr.log").open("w", encoding="utf-8") as stderr:
            subprocess.run(command, cwd=repo_dir, env=env, stdout=stdout, stderr=stderr, timeout=timeout, check=True)
    except subprocess.TimeoutExpired as exc:
        return [
            _finalize_sequence(dataset, job, config, "timeout", f"timeout after {exc.timeout} seconds", f"batch_{batch_index:04d}", started_at, start)
            for job in batch_rows
        ]
    except (subprocess.CalledProcessError, OSError) as exc:
        return [
            _finalize_sequence(dataset, job, config, "failed", str(exc), f"batch_{batch_index:04d}", started_at, start)
            for job in batch_rows
        ]

    rows = []
    for job in batch_rows:
        try:
            result = normalize_openfold_outputs(bdir / "openfold_output", job["sequence_id"], job_dir(dataset, job), force=force)
            row = build_manifest_row(
                dataset,
                job,
                config,
                status="success",
                batch_id=f"batch_{batch_index:04d}",
                started_at=started_at,
                finished_at=utc_now_iso(),
                runtime_seconds=round(time.time() - start, 3),
                notes=f"OpenFold WT completed. sample_ranking_score={result.get('summary_row', {}).get('sample_ranking_score', '')}",
            )
        except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
            row = _finalize_sequence(dataset, job, config, "parse_failed", str(exc), f"batch_{batch_index:04d}", started_at, start)
        write_job_manifest(job_dir(dataset, job), row, force=True)
        rows.append(row)
    status_counts = Counter(row.get("job_status", "") for row in rows)
    batch_status = "success" if set(status_counts) == {"success"} else "completed_with_failures"
    _update_batch_manifest(dataset, batch_index, batch_status, rows, force=True)
    return rows


def _finalize_sequence(dataset: str, job: dict[str, str], config: dict[str, Any], status: str, error: str, batch_id: str, started_at: str, start: float) -> dict[str, Any]:
    row = build_manifest_row(
        dataset,
        job,
        config,
        status=status,
        error=error,
        batch_id=batch_id,
        started_at=started_at,
        finished_at=utc_now_iso(),
        runtime_seconds=round(time.time() - start, 3),
        notes="OpenFold WT execution failed or output normalization failed.",
    )
    write_job_manifest(job_dir(dataset, job), row, force=True)
    return row


def _update_batch_manifest(dataset: str, batch_index: int, status: str, rows: list[dict[str, Any]], force: bool) -> None:
    path = batch_dir(dataset, batch_index) / "batch_manifest.json"
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload.update(
        {
            "job_status": status,
            "finished_at": utc_now_iso(),
            "summary": dict(Counter(row.get("job_status", "") for row in rows)),
        }
    )
    write_json(path, payload, force=force)


def write_stage_outputs(dataset: str, rows: list[dict[str, Any]], force: bool) -> tuple[Path, Path]:
    return write_stage_manifest(dataset, STAGE_NAME, rows, force=force, root=ROOT)


def create_stage_completion_marker(dataset: str, rows: list[dict[str, Any]], command: str, force: bool) -> Path:
    marker = ROOT / "data" / "run_state" / dataset / f"{STAGE_GROUP}.{STAGE_NAME}.completed.json"
    statuses = Counter(str(row.get("job_status", "")) for row in rows)
    terminal = {"success", "failed", "skipped", "blocked", "parse_failed", "timeout", "cancelled"}
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


def _openfold_env(repo_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{repo_dir}:{ROOT}" + (f":{existing}" if existing else "")
    return env


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
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--sequence-id")
    parser.add_argument("--pdb-id")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args(argv)

    if sum(bool(value) for value in [args.materialize_plan, args.preflight, args.smoke_run, args.run]) > 1:
        raise SystemExit("Choose only one of --preflight, --materialize-plan, --smoke-run, or --run.")
    allow_example = not (args.smoke_run or args.run)
    config, config_warnings = load_config_for_mode(args.config, allow_example_fallback=allow_example)
    if args.batch_size is not None:
        config.setdefault("openfold", {})["batch_size"] = args.batch_size
    jobs = select_jobs(args.dataset, config, max_sequences=args.max_sequences, sequence_id=args.sequence_id, pdb_id=args.pdb_id)
    openfold = config.get("openfold", {})
    python_bin = openfold.get("python_bin", "python3")
    repo_dir = resolve_config_path(openfold.get("openfold_repo_dir"), ROOT) or ROOT / "openfold-3"
    python_ok, python_error = validate_python_bin(python_bin)
    import_ok, import_error = validate_openfold_import(python_bin, repo_dir, ROOT) if python_ok and repo_dir.exists() else (False, "repo_dir_missing_or_python_invalid")
    global_checks = {
        "config_path": relative(args.config),
        "config_exists": args.config.exists(),
        "sequence_manifest": relative(sequence_manifest_path(config)),
        "sequence_manifest_exists": sequence_manifest_path(config).exists(),
        "openfold_repo_dir": relative(repo_dir),
        "openfold_repo_dir_exists": repo_dir.exists(),
        "python_ok": python_ok,
        "python_error": python_error,
        "openfold_import_ok": import_ok,
        "openfold_import_error": import_error,
        "selected_sequences": len(jobs),
        "batch_size": int(openfold.get("batch_size", 1)),
        "warnings": config_warnings,
    }
    print(f"Dataset: {args.dataset}")
    print(f"Stage: {STAGE_NAME}")
    print(f"Selected sequences: {len(jobs)}")
    print(f"Batch size: {global_checks['batch_size']}")

    if args.smoke_run and args.max_batches is None:
        args.max_batches = 1

    if args.dry_run or not any([args.materialize_plan, args.preflight, args.smoke_run, args.run]):
        print("Dry run only. OpenFold was not executed.")
        batches = batch_sequence_rows(jobs, int(openfold.get("batch_size", 1)))
        print(f"Planned batches: {len(batches)}")
        return 0

    if args.preflight:
        rows, summary = preflight_jobs(args.dataset, jobs, config, force=args.force)
        md_path, _ = write_preflight_report(args.dataset, rows, summary, global_checks)
        print(f"Preflight report: {relative(md_path)}")
        print(f"Ready for execution: {summary['ready_for_execution']}")
        print(f"Blocked/preflight_failed: {summary['blocked_preflight_failed']}")
        return 0

    if args.materialize_plan:
        rows = materialize_batches(args.dataset, jobs, config, force=args.force)
        csv_path, _ = write_stage_outputs(args.dataset, rows, force=args.force)
        print(f"Materialized sequences: {len(rows)}")
        print(f"Stage manifest: {relative(csv_path)}")
        print("OpenFold was not executed. No completion marker was created.")
        return 0

    if (args.smoke_run or args.run) and not python_ok:
        raise SystemExit(f"OpenFold python executable is not runnable: {python_error}")
    if (args.smoke_run or args.run) and not import_ok:
        raise SystemExit(f"OpenFold import check failed: {import_error}")
    batches = batch_sequence_rows(jobs, int(openfold.get("batch_size", 1)))
    if args.max_batches is not None:
        batches = batches[: args.max_batches]
    if args.smoke_run and len(batches) > 1:
        raise SystemExit("Smoke-run is limited; pass --max-batches 1 or omit --max-batches.")
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_batch, args.dataset, index, batch_rows, config, args.force) for index, batch_rows in enumerate(batches, start=1)]
        for future in as_completed(futures):
            rows.extend(future.result())
    csv_path, _ = write_stage_outputs(args.dataset, rows, force=True)
    marker = create_stage_completion_marker(args.dataset, rows, " ".join(sys.argv), force=True)
    print(f"Executed batches: {len(batches)}")
    print(f"Updated sequences: {len(rows)}")
    print(f"Stage manifest: {relative(csv_path)}")
    print(f"Per-stage completion marker: {relative(marker)}")
    print("Global gpu_heavy.completed.json was not created.")
    return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
