#!/usr/bin/env python3
"""Check local/server readiness for current_internal run stage groups."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_CONFIG = ROOT / "configs" / "run_stages.yaml"
DEFAULT_SERVER_CONFIG = ROOT / "configs" / "server_paths.yaml"
EXAMPLE_SERVER_CONFIG = ROOT / "configs" / "server_paths.example.yaml"
REPORT_DIR = ROOT / "reports" / "server_environment"


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
        return load_yaml(path, required=True), {
            "server_config_path": rel(path),
            "server_config_exists": True,
            "server_config_is_example_fallback": False,
        }
    fallback = EXAMPLE_SERVER_CONFIG
    if not fallback.exists():
        return {}, {
            "server_config_path": rel(path),
            "server_config_exists": False,
            "server_config_is_example_fallback": False,
            "server_config_warning": f"missing server config and fallback: {rel(fallback)}",
        }
    return load_yaml(fallback, required=True), {
        "server_config_path": rel(fallback),
        "server_config_requested_path": rel(path),
        "server_config_exists": False,
        "server_config_is_example_fallback": True,
        "server_config_warning": f"missing {rel(path)}; using read-only example fallback",
    }


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_path(path: Path, label: str, critical: bool, results: list[dict[str, Any]]) -> None:
    exists = path.exists()
    results.append(
        {
            "label": label,
            "path": rel(path),
            "status": "ok" if exists else ("missing" if critical else "warning"),
            "critical": critical,
            "exists": exists,
            "is_dir": path.is_dir(),
            "writable": os.access(path, os.W_OK) if exists else False,
            "message": "" if exists else "missing",
        }
    )


def check_writable(path: Path, label: str, results: list[dict[str, Any]]) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_path = path / ".write_test"
        test_path.write_text("ok\n", encoding="utf-8")
        test_path.unlink()
        status, message = "ok", ""
    except OSError as exc:
        status, message = "missing", str(exc)
    results.append({"label": label, "path": rel(path), "status": status, "critical": True, "message": message})


def resolve_config_path(value: str | None, fallback: Path | None = None) -> Path:
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path
    if fallback is None:
        return ROOT
    return fallback


def check_effective_dir(path: Path, label: str, required: bool, results: list[dict[str, Any]]) -> None:
    exists = path.exists()
    is_dir = path.is_dir()
    writable = os.access(path, os.W_OK) if exists else False
    parent = path.parent
    parent_writable = parent.exists() and os.access(parent, os.W_OK)
    if exists and is_dir and writable:
        status, critical, message = "ok", False, ""
    elif exists and not is_dir:
        status, critical, message = "error", required, "exists but is not a directory"
    elif exists and not writable:
        status, critical, message = "error", required, "exists but is not writable"
    elif parent_writable:
        status, critical, message = "warning", False, "missing, but parent is writable"
    else:
        status, critical, message = ("error" if required else "warning"), required, "missing and parent is not writable"
    results.append(
        {
            "label": label,
            "path": path.as_posix(),
            "status": status,
            "critical": critical and status == "error",
            "exists": exists,
            "is_dir": is_dir,
            "writable": writable,
            "message": message,
        }
    )


def count_jobs(rows: list[dict[str, str]], *, method: set[str] | None = None, source: set[str] | None = None, status: str | None = None) -> int:
    total = 0
    for row in rows:
        if method and row.get("method") not in method:
            continue
        if source and row.get("structure_source") not in source:
            continue
        if status and row.get("input_status") != status:
            continue
        total += 1
    return total


def check_stage_group(stage_group: str, server_config: dict[str, Any], server_config_meta: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "stage_group": stage_group,
        "python_version": sys.version.split()[0],
        "cwd": os.getcwd(),
        "host": platform.node(),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    server_paths = server_config.get("server", {})
    project_root = resolve_config_path(server_paths.get("project_root"), ROOT)
    data_root = resolve_config_path(server_paths.get("data_root"), project_root / "data")
    results_root = resolve_config_path(server_paths.get("results_root"), project_root / "results")
    logs_root = resolve_config_path(server_paths.get("logs_root"), project_root / "logs")
    reports_root = resolve_config_path(server_paths.get("reports_root"), project_root / "reports")
    reports_fallback = "server.reports_root missing; using project_root/reports" if not server_paths.get("reports_root") else ""
    summary["effective_project_root"] = project_root.as_posix()
    summary["effective_data_root"] = data_root.as_posix()
    summary["effective_results_root"] = results_root.as_posix()
    summary["effective_logs_root"] = logs_root.as_posix()
    summary["effective_reports_root"] = reports_root.as_posix()
    summary["effective_reports_root_fallback"] = reports_fallback

    for path, label in [
        (ROOT / "configs" / "datasets.yaml", "datasets registry"),
        (ROOT / "configs" / "run_stages.yaml", "run stages registry"),
        (ROOT / "data" / "processed" / "current_internal" / "current_internal_recalc_jobs_resolved.csv", "resolved jobs"),
    ]:
        check_path(path, label, True, checks)
    check_path(ROOT / "data" / "processed" / "current_internal" / "current_internal_evaluation_tasks.csv", "evaluation tasks", False, checks)
    server_config_exists = bool(server_config_meta.get("server_config_exists"))
    using_example = bool(server_config_meta.get("server_config_is_example_fallback"))
    enforce_server_paths = server_config_exists and not using_example
    check_effective_dir(project_root, "effective server.project_root", enforce_server_paths, checks)
    check_effective_dir(data_root, "effective server.data_root", enforce_server_paths, checks)
    check_effective_dir(results_root, "effective server.results_root", enforce_server_paths, checks)
    check_effective_dir(logs_root, "effective server.logs_root", enforce_server_paths, checks)
    check_effective_dir(reports_root, "effective server.reports_root", enforce_server_paths, checks)

    jobs = read_csv_rows(ROOT / "data" / "processed" / "current_internal" / "current_internal_recalc_jobs_resolved.csv")
    eval_tasks = read_csv_rows(ROOT / "data" / "processed" / "current_internal" / "current_internal_evaluation_tasks.csv")
    summary["resolved_jobs"] = len(jobs)
    summary["evaluation_tasks"] = len(eval_tasks)

    if stage_group in {"all", "preprepare"}:
        for path, label in [
            (ROOT / "data" / "processed" / "current_internal" / "current_internal_mutations.csv", "normalized mutations"),
            (ROOT / "data" / "manifests" / "current_internal_structure_manifest_resolved.csv", "resolved structure manifest"),
            (ROOT / "data" / "manifests" / "current_internal_sequence_manifest.csv", "sequence manifest"),
            (ROOT / "data" / "manifests" / "current_internal_structure_override_template.csv", "override template"),
            (ROOT / "configs" / "evaluation_metrics.yaml", "evaluation registry"),
        ]:
            check_path(path, label, True, checks)

    if stage_group in {"all", "cpu_heavy"}:
        tools = server_config.get("tools", {})
        foldx = str(tools.get("foldx_binary", "") or "")
        rosetta = str(tools.get("rosetta_bin_dir", "") or "")
        if foldx and "/path/to/" not in foldx:
            check_path(Path(foldx), "FoldX binary", False, checks)
        else:
            checks.append({"label": "FoldX binary", "path": foldx, "status": "warning", "critical": False, "message": "not configured"})
        if rosetta and "/path/to/" not in rosetta:
            check_path(Path(rosetta), "Rosetta bin dir", False, checks)
        else:
            checks.append({"label": "Rosetta bin dir", "path": rosetta, "status": "warning", "critical": False, "message": "not configured"})
        check_path(ROOT / "data" / "structures" / "chain" / "current_internal", "prepared chain structures", True, checks)
        summary["input_ready_experimental_cpu_jobs"] = count_jobs(jobs, method={"foldx", "rosetta"}, source={"experimental"}, status="ready")

    if stage_group in {"all", "gpu_heavy"}:
        try:
            import torch  # type: ignore

            summary["torch_import"] = "ok"
            summary["torch_cuda_available"] = bool(torch.cuda.is_available())
        except Exception as exc:  # pragma: no cover - environment dependent
            summary["torch_import"] = f"warning: {exc}"
            summary["torch_cuda_available"] = False
        nvidia = shutil.which("nvidia-smi")
        summary["nvidia_smi"] = nvidia or "not found"
        if nvidia:
            proc = subprocess.run([nvidia, "-L"], text=True, capture_output=True, timeout=10)
            summary["nvidia_smi_output"] = proc.stdout.strip() or proc.stderr.strip()
        summary["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        check_path(ROOT / "data" / "sequences" / "current_internal" / "current_internal_openfold_inputs.fasta", "OpenFold FASTA", True, checks)
        summary["pending_openfold_structure_jobs"] = count_jobs(jobs, source={"openfold"}, status="pending_openfold_structure")

    if stage_group in {"all", "post_gpu_cpu_heavy"}:
        expected_dirs = [
            ROOT / "results" / "current_internal" / "openfold" / "wt",
        ]
        for path in expected_dirs:
            check_path(path, f"optional expected GPU outputs: {rel(path)}", False, checks)
        openfold_ready_jobs = read_csv_rows(ROOT / "data" / "processed" / "current_internal" / "current_internal_recalc_jobs_openfold_ready.csv")
        summary["openfold_ready_jobs_table_exists"] = bool(openfold_ready_jobs)
        summary["input_ready_post_gpu_cpu_jobs"] = count_jobs(openfold_ready_jobs, method={"foldx", "rosetta"}, source={"openfold"}, status="ready")

    if stage_group in {"all", "analysis_reporting", "postprocess"}:
        expected_dirs = [
            ROOT / "results" / "current_internal" / "foldx" / "experimental",
            ROOT / "results" / "current_internal" / "rosetta" / "experimental",
            ROOT / "results" / "current_internal" / "openfold" / "wt",
            ROOT / "results" / "current_internal" / "foldx" / "openfold",
            ROOT / "results" / "current_internal" / "rosetta" / "openfold",
            ROOT / "results" / "evaluation" / "current_internal",
        ]
        for path in expected_dirs:
            check_path(path, f"optional expected outputs: {rel(path)}", False, checks)
        summary["evaluation_task_readiness"] = dict(Counter(row.get("input_ready", "") for row in eval_tasks))
        summary["evaluation_blocking_reasons"] = dict(Counter(row.get("blocking_reason", "") for row in eval_tasks if row.get("blocking_reason", "")))

    return checks, summary


def write_reports(stage_group: str, checks: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "checks": checks}
    json_path = REPORT_DIR / f"{stage_group}_environment_check.json"
    md_path = REPORT_DIR / f"{stage_group}_environment_check.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# Environment Check: {stage_group}",
        "",
        f"- Python: {summary['python_version']}",
        f"- CWD: `{summary['cwd']}`",
        f"- Host: `{summary['host']}`",
        f"- Server config: `{summary.get('server_config_path', '')}`",
        f"- Server config fallback: `{summary.get('server_config_is_example_fallback', False)}`",
        "",
        "| status | critical | exists | is_dir | writable | label | path | message |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check['status']} | {check['critical']} | {check.get('exists', '')} | {check.get('is_dir', '')} | "
            f"{check.get('writable', '')} | {check['label']} | `{check.get('path', '')}` | {check.get('message', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-group", required=True, choices=["preprepare", "cpu_heavy", "gpu_heavy", "post_gpu_cpu_heavy", "analysis_reporting", "postprocess", "all"])
    parser.add_argument("--server-config", type=Path, default=DEFAULT_SERVER_CONFIG)
    args = parser.parse_args()

    load_yaml(DEFAULT_STAGE_CONFIG)
    server_config, server_config_summary = load_server_config(args.server_config)
    checks, summary = check_stage_group(args.stage_group, server_config, server_config_summary)
    summary.update(server_config_summary)
    write_reports(args.stage_group, checks, summary)
    critical_missing = [check for check in checks if check["critical"] and check["status"] != "ok"]
    warnings = [check for check in checks if not check["critical"] and check["status"] != "ok"]
    print(f"Stage group: {args.stage_group}")
    print(f"Server config: {summary.get('server_config_path')} fallback={summary.get('server_config_is_example_fallback')}")
    print(f"Checks: {len(checks)}, critical missing: {len(critical_missing)}, warnings: {len(warnings)}")
    print(f"Report: reports/server_environment/{args.stage_group}_environment_check.md")
    return 1 if critical_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
