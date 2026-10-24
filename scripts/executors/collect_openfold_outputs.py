#!/usr/bin/env python3
"""Collect OpenFold WT outputs and derive downstream OpenFold-readiness tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "run"))

from execution_utils import ensure_dir, read_csv_dicts, safe_write_text, write_csv_dicts, write_json  # noqa: E402


STAGE_NAME = "collect_openfold_outputs"
STAGE_GROUP = "post_gpu_cpu_heavy"

MANIFEST_FIELDS = [
    "dataset_name",
    "sequence_id",
    "uniprot_id",
    "protein_id",
    "pdb_id",
    "chain_id",
    "sequence_length",
    "openfold_output_dir",
    "model_path",
    "model_pdb_path",
    "confidence_path",
    "openfold_result_path",
    "job_manifest_path",
    "job_status",
    "output_status",
    "mean_plddt",
    "plddt_source_field",
    "sample_ranking_score",
    "error_message",
    "notes",
]


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def bool_from_text(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    return read_csv_dicts(path)


def sequence_manifest_path(dataset: str) -> Path:
    return ROOT / "data" / "manifests" / f"{dataset}_sequence_manifest.csv"


def resolved_jobs_path(dataset: str) -> Path:
    return ROOT / "data" / "processed" / dataset / f"{dataset}_recalc_jobs_resolved.csv"


def output_jobs_path(dataset: str) -> Path:
    return ROOT / "data" / "processed" / dataset / f"{dataset}_recalc_jobs_openfold_ready.csv"


def openfold_dir(dataset: str, sequence_id: str) -> Path:
    return ROOT / "results" / dataset / "openfold" / "wt" / sequence_id


def collect_sequence_output(dataset: str, sequence: dict[str, str]) -> dict[str, Any]:
    sequence_id = sequence["sequence_id"]
    out_dir = openfold_dir(dataset, sequence_id)
    model = out_dir / "model.cif"
    model_pdb = out_dir / "model.pdb"
    confidence = out_dir / "confidence.json"
    result = out_dir / "openfold_result.json"
    job_manifest = out_dir / "job_manifest.json"
    errors: list[str] = []
    job_status = ""
    sample_ranking_score = ""
    mean_plddt = ""
    plddt_source_field = ""

    if job_manifest.exists():
        try:
            payload = json.loads(job_manifest.read_text(encoding="utf-8"))
            job_status = str(payload.get("job_status", ""))
        except json.JSONDecodeError:
            errors.append("job_manifest_json_invalid")
    else:
        errors.append("missing_job_manifest")

    if not model.exists():
        errors.append("missing_model_cif")
    if not model_pdb.exists():
        errors.append("missing_model_pdb")
    if not confidence.exists():
        errors.append("missing_confidence_json")
    if not result.exists():
        errors.append("missing_openfold_result_json")

    if result.exists():
        try:
            payload = json.loads(result.read_text(encoding="utf-8"))
            sample_ranking_score = str(payload.get("summary_row", {}).get("sample_ranking_score", ""))
        except json.JSONDecodeError:
            errors.append("openfold_result_json_invalid")
    if confidence.exists():
        try:
            mean_plddt, plddt_source_field = extract_mean_plddt(json.loads(confidence.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(f"confidence_parse_failed:{exc}")

    output_status = "ready" if model.exists() and model_pdb.exists() and confidence.exists() and not errors else "incomplete"
    if job_status and job_status != "success":
        output_status = "job_not_success"
    return {
        "dataset_name": dataset,
        "sequence_id": sequence_id,
        "uniprot_id": sequence.get("uniprot_id", ""),
        "protein_id": sequence.get("protein_id", ""),
        "pdb_id": sequence.get("pdb_id", "").upper(),
        "chain_id": sequence.get("chain_id", ""),
        "sequence_length": sequence.get("sequence_length", ""),
        "openfold_output_dir": relative(out_dir),
        "model_path": relative(model),
        "model_pdb_path": relative(model_pdb),
        "confidence_path": relative(confidence),
        "openfold_result_path": relative(result),
        "job_manifest_path": relative(job_manifest),
        "job_status": job_status,
        "output_status": output_status,
        "mean_plddt": mean_plddt,
        "plddt_source_field": plddt_source_field,
        "sample_ranking_score": sample_ranking_score,
        "error_message": "; ".join(errors),
        "notes": "Collected from normalized OpenFold WT output layout.",
    }


def extract_mean_plddt(payload: Any) -> tuple[str, str]:
    if isinstance(payload, dict):
        for key in ("mean_plddt", "global_plddt", "plddt_mean", "avg_plddt"):
            value = payload.get(key)
            if _is_number(value):
                return str(round(float(value), 6)), key
        for key in ("plddt", "plddts", "token_plddts", "atom_plddts", "residue_plddts"):
            value = payload.get(key)
            numbers = _flatten_numbers(value)
            if numbers:
                return str(round(statistics.fmean(numbers), 6)), key
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                nested, nested_key = extract_mean_plddt(value)
                if nested:
                    return nested, f"{key}.{nested_key}"
    if isinstance(payload, list):
        numbers = _flatten_numbers(payload)
        if numbers:
            return str(round(statistics.fmean(numbers), 6)), "list"
    return "", ""


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _flatten_numbers(value: Any) -> list[float]:
    numbers: list[float] = []
    if _is_number(value):
        numbers.append(float(value))
    elif isinstance(value, list):
        for item in value:
            numbers.extend(_flatten_numbers(item))
    elif isinstance(value, dict):
        for item in value.values():
            numbers.extend(_flatten_numbers(item))
    return numbers


def update_openfold_jobs(dataset: str, jobs: list[dict[str, str]], outputs: list[dict[str, Any]], sequences: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_chain = {(row.get("pdb_id", "").upper(), row.get("chain_id", "")): row for row in sequences}
    output_by_sequence = {row["sequence_id"]: row for row in outputs}
    updated: list[dict[str, Any]] = []
    for job in jobs:
        row: dict[str, Any] = dict(job)
        row.setdefault("sequence_id", "")
        row.setdefault("openfold_structure_path", "")
        row.setdefault("openfold_confidence_path", "")
        row.setdefault("openfold_output_status", "")
        if row.get("method") in {"foldx", "rosetta"} and row.get("structure_source") == "openfold":
            sequence = by_chain.get((row.get("pdb_id", "").upper(), row.get("chain_id", "")), {})
            sequence_id = sequence.get("sequence_id", "")
            output = output_by_sequence.get(sequence_id, {})
            row["sequence_id"] = sequence_id
            row["openfold_structure_path"] = output.get("model_pdb_path", "") or output.get("model_path", "")
            row["openfold_confidence_path"] = output.get("confidence_path", "")
            row["openfold_output_status"] = output.get("output_status", "missing_sequence_output")
            if output.get("output_status") == "ready":
                row["input_ready"] = "true"
                row["input_status"] = "ready"
                row["blocking_reason"] = ""
                row["notes"] = _append_note(row.get("notes", ""), "OpenFold WT structure is available from collected outputs.")
            else:
                row["input_ready"] = "false"
                row["input_status"] = "blocked"
                row["blocking_reason"] = output.get("error_message") or "pending_openfold_structure"
        updated.append(row)
    return updated


def _append_note(existing: str, note: str) -> str:
    existing = existing.strip()
    return f"{existing} {note}".strip() if existing else note


def build_qc(dataset: str, outputs: list[dict[str, Any]], updated_jobs: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(row.get("output_status", "") for row in outputs)
    error_counts = Counter()
    for row in outputs:
        for part in str(row.get("error_message", "")).split(";"):
            part = part.strip()
            if part:
                error_counts[part.split(":", 1)[0]] += 1
    openfold_jobs = [row for row in updated_jobs if row.get("method") in {"foldx", "rosetta"} and row.get("structure_source") == "openfold"]
    ready_openfold_jobs = [row for row in openfold_jobs if bool_from_text(row.get("input_ready", ""))]
    return {
        "dataset_name": dataset,
        "sequence_outputs": {
            "total_sequences": len(outputs),
            "by_output_status": dict(status_counts),
            "errors": dict(error_counts),
            "mean_plddt_available": sum(1 for row in outputs if row.get("mean_plddt") not in {"", None}),
        },
        "jobs": {
            "openfold_structure_jobs": len(openfold_jobs),
            "openfold_structure_jobs_ready": len(ready_openfold_jobs),
            "blocking_reasons": dict(Counter(row.get("blocking_reason", "") or "ready" for row in openfold_jobs)),
        },
        "examples": outputs[:20],
    }


def write_report(dataset: str, qc: dict[str, Any], force: bool) -> tuple[Path, Path]:
    report_dir = ROOT / "reports" / dataset
    ensure_dir(report_dir)
    md_path = report_dir / f"{dataset}_openfold_collection_qc.md"
    json_path = report_dir / f"{dataset}_openfold_collection_qc.json"
    lines = [
        "# OpenFold WT output collection QC",
        "",
        "This report collects already generated OpenFold WT outputs. It does not run OpenFold.",
        "",
        "## Sequence outputs",
        "",
    ]
    for key, value in qc["sequence_outputs"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## OpenFold-dependent jobs", ""])
    for key, value in qc["jobs"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## First examples", ""])
    for row in qc["examples"][:10]:
        lines.append(f"- `{row['sequence_id']}`: {row['output_status']} {row.get('error_message', '')}".rstrip())
    safe_write_text(md_path, "\n".join(lines) + "\n", force=force)
    write_json(json_path, qc, force=force)
    return md_path, json_path


def write_outputs(dataset: str, manifest: list[dict[str, Any]], updated_jobs: list[dict[str, Any]], qc: dict[str, Any], force: bool) -> dict[str, Path]:
    manifests_dir = ROOT / "data" / "manifests"
    processed_dir = ROOT / "data" / "processed" / dataset
    ensure_dir(manifests_dir)
    ensure_dir(processed_dir)
    manifest_csv = manifests_dir / f"{dataset}_openfold_wt_manifest.csv"
    manifest_json = manifests_dir / f"{dataset}_openfold_wt_manifest.json"
    errors_csv = manifests_dir / f"{dataset}_openfold_wt_errors.csv"
    jobs_csv = output_jobs_path(dataset)
    write_csv_with_fields(manifest_csv, manifest, MANIFEST_FIELDS, force=force)
    write_json(manifest_json, {"dataset_name": dataset, "outputs": manifest}, force=force)
    write_csv_with_fields(errors_csv, [row for row in manifest if row.get("output_status") != "ready"], MANIFEST_FIELDS, force=force)
    write_csv_dicts(jobs_csv, updated_jobs, force=force)
    report_md, report_json = write_report(dataset, qc, force=force)
    return {
        "manifest_csv": manifest_csv,
        "manifest_json": manifest_json,
        "errors_csv": errors_csv,
        "jobs_csv": jobs_csv,
        "report_md": report_md,
        "report_json": report_json,
    }


def write_csv_with_fields(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file without --force: {path}")
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args(argv)

    if args.dataset != "current_internal":
        raise SystemExit("This collector currently supports only --dataset current_internal")

    sequence_path = sequence_manifest_path(args.dataset)
    jobs_path = resolved_jobs_path(args.dataset)
    for path in (sequence_path, jobs_path):
        if not path.exists():
            raise SystemExit(f"Missing required input: {relative(path)}")

    sequences = read_csv(sequence_path)
    jobs = read_csv(jobs_path)
    manifest = [collect_sequence_output(args.dataset, row) for row in sequences]
    updated_jobs = update_openfold_jobs(args.dataset, jobs, manifest, sequences)
    qc = build_qc(args.dataset, manifest, updated_jobs)

    print(f"Dataset: {args.dataset}")
    print(f"Sequences checked: {len(manifest)}")
    print(f"Sequence output status: {json.dumps(qc['sequence_outputs']['by_output_status'], sort_keys=True)}")
    print(f"OpenFold structure jobs ready: {qc['jobs']['openfold_structure_jobs_ready']} / {qc['jobs']['openfold_structure_jobs']}")
    if args.dry_run:
        print("Dry run only. No files were written.")
        return 0

    paths = write_outputs(args.dataset, manifest, updated_jobs, qc, force=args.force)
    print(f"OpenFold manifest: {relative(paths['manifest_csv'])}")
    print(f"Derived jobs table: {relative(paths['jobs_csv'])}")
    print(f"QC report: {relative(paths['report_md'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
