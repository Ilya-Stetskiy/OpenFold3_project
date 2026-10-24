#!/usr/bin/env python3
"""Apply manual current_internal structure override decisions to derived outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


DEFAULT_DATASETS_CONFIG = Path("configs/datasets.yaml")
VALID_DECISIONS = {
    "accept_original",
    "accept_position_remap",
    "exclude",
    "needs_review",
    "accept_custom_structure",
    "accept_custom_chain",
}
RESOLVED_MANIFEST_FIELDS = [
    "dataset_name",
    "mutation_record_id",
    "uniprot_id",
    "protein_id",
    "pdb_id",
    "chain_id",
    "mutation",
    "wild_type_aa",
    "original_position",
    "resolved_position",
    "mutant_aa",
    "original_structure_wt_aa",
    "resolved_structure_wt_aa",
    "original_chain_structure_path",
    "resolved_chain_structure_path",
    "original_wt_residue_matches_mutation",
    "resolved_wt_residue_matches_mutation",
    "curator_decision",
    "decision_reason",
    "structure_resolution_status",
    "structure_resolution_flags",
    "error_message",
]
JOB_EXTRA_FIELDS = [
    "resolved_position",
    "structure_resolution_status",
    "structure_resolution_flags",
    "override_applied",
]
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return data


def project_root_from(config_path: Path) -> Path:
    return config_path.parent.parent.resolve()


def dataset_config(config: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    datasets = config.get("datasets", {})
    if not isinstance(datasets, dict) or dataset_name not in datasets:
        raise SystemExit(f"Unknown dataset in configs/datasets.yaml: {dataset_name}")
    return datasets[dataset_name]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing output without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_text(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing output without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def bool_str(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_position(value: str) -> tuple[str, str]:
    match = re.fullmatch(r"\s*(-?\d+)([A-Za-z]?)\s*", str(value or ""))
    if not match:
        return "", ""
    return match.group(1), match.group(2).upper()


def relative_or_original(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_existing_path(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def build_residue_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], str]:
    index: dict[tuple[str, str, str, str], str] = {}
    for row in rows:
        key = (
            row.get("pdb_id", "").upper(),
            row.get("chain_id", ""),
            row.get("original_residue_id", ""),
            row.get("original_insertion_code", "").upper(),
        )
        index[key] = row.get("residue_name_1letter", "")
    return index


def residue_from_mapping(
    residue_index: dict[tuple[str, str, str, str], str],
    pdb_id: str,
    chain_id: str,
    position_value: str,
) -> str:
    residue_id, insertion_code = parse_position(position_value)
    if not residue_id:
        return ""
    return residue_index.get((pdb_id.upper(), chain_id, residue_id, insertion_code), "")


def residue_from_pdb(path: Path, chain_id: str, position_value: str) -> str:
    residue_id, insertion_code = parse_position(position_value)
    if not residue_id or not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 27:
                continue
            if line[21].strip() != chain_id:
                continue
            if line[22:26].strip() != residue_id:
                continue
            if line[26].strip().upper() != insertion_code:
                continue
            return AA3_TO_1.get(line[17:20].strip().upper(), "")
    return ""


def validate_template_rows(rows: list[dict[str, str]]) -> None:
    bad = sorted({row.get("curator_decision", "") for row in rows if row.get("curator_decision", "") not in VALID_DECISIONS})
    if bad:
        raise SystemExit(f"Invalid curator_decision value(s): {', '.join(bad)}")


def append_note(existing: str, addition: str) -> str:
    if not addition:
        return existing
    return f"{existing}; {addition}" if existing else addition


def decide_resolution(
    original: dict[str, str],
    override: dict[str, str] | None,
    residue_index: dict[tuple[str, str, str, str], str],
    root: Path,
) -> dict[str, Any]:
    original_match = bool_str(original.get("wt_residue_matches_mutation", ""))
    original_position = original.get("position", "")
    original_path = original.get("chain_structure_path", "")
    base = {
        "dataset_name": original.get("dataset_name", ""),
        "mutation_record_id": original.get("mutation_record_id", ""),
        "uniprot_id": original.get("uniprot_id", ""),
        "protein_id": original.get("protein_id", ""),
        "pdb_id": original.get("pdb_id", "").upper(),
        "chain_id": original.get("chain_id", ""),
        "mutation": original.get("mutation", ""),
        "wild_type_aa": original.get("wild_type_aa", ""),
        "original_position": original_position,
        "resolved_position": original_position,
        "mutant_aa": original.get("mutant_aa", ""),
        "original_structure_wt_aa": original.get("structure_wt_aa", ""),
        "resolved_structure_wt_aa": original.get("structure_wt_aa", ""),
        "original_chain_structure_path": original_path,
        "resolved_chain_structure_path": original_path,
        "original_wt_residue_matches_mutation": str(original_match).lower(),
        "resolved_wt_residue_matches_mutation": str(original_match).lower(),
        "curator_decision": "",
        "decision_reason": "",
        "structure_resolution_status": "original_valid" if original_match else "failed",
        "structure_resolution_flags": "",
        "error_message": original.get("error_message", ""),
        "allow_experimental_structure_jobs": original_match,
        "blocking_reason": "" if original_match else "wt_residue_mismatch",
        "override_applied": False,
    }
    if override is None:
        return base

    decision = override.get("curator_decision", "")
    flags: list[str] = []
    errors: list[str] = []
    base["curator_decision"] = decision
    base["decision_reason"] = override.get("decision_reason", "")
    base["override_applied"] = True

    if decision == "accept_original":
        if not original_match:
            flags.append("accepted_original_despite_mismatch")
        base["resolved_wt_residue_matches_mutation"] = "true"
        base["structure_resolution_status"] = "original_valid"
        base["allow_experimental_structure_jobs"] = True
        base["blocking_reason"] = ""
    elif decision == "accept_position_remap":
        curated_position = override.get("curated_position", "")
        residue = residue_from_mapping(residue_index, original.get("pdb_id", ""), original.get("chain_id", ""), curated_position)
        base["resolved_position"] = curated_position
        base["resolved_structure_wt_aa"] = residue
        if residue and residue == original.get("wild_type_aa", ""):
            base["resolved_wt_residue_matches_mutation"] = "true"
            base["structure_resolution_status"] = "remapped_valid"
            base["allow_experimental_structure_jobs"] = True
            base["blocking_reason"] = ""
            flags.append("position_remap_validated")
        else:
            base["resolved_wt_residue_matches_mutation"] = "false"
            base["structure_resolution_status"] = "failed"
            base["allow_experimental_structure_jobs"] = False
            base["blocking_reason"] = "curated_remap_validation_failed"
            errors.append("Curated position does not resolve to expected wild-type residue in residue_mapping.")
            flags.append("curated_remap_validation_failed")
    elif decision == "accept_custom_structure":
        curated_path = override.get("curated_structure_path", "")
        chain_id = override.get("curated_chain_id", "") or original.get("chain_id", "")
        position = override.get("curated_position", "") or original_position
        path = resolve_existing_path(curated_path, root) if curated_path else Path("")
        base["resolved_position"] = position
        base["resolved_chain_structure_path"] = curated_path
        if curated_path and path.exists():
            residue = residue_from_pdb(path, chain_id, position)
            base["resolved_structure_wt_aa"] = residue
            if residue == original.get("wild_type_aa", ""):
                base["resolved_wt_residue_matches_mutation"] = "true"
                base["structure_resolution_status"] = "custom_structure_valid"
                base["allow_experimental_structure_jobs"] = True
                base["blocking_reason"] = ""
                flags.append("custom_structure_validated")
            else:
                base["structure_resolution_status"] = "failed"
                base["allow_experimental_structure_jobs"] = False
                base["blocking_reason"] = "custom_structure_validation_failed"
                errors.append("Custom structure file exists but residue validation failed.")
                flags.append("custom_structure_validation_failed")
        else:
            base["structure_resolution_status"] = "failed"
            base["allow_experimental_structure_jobs"] = False
            base["blocking_reason"] = "custom_structure_missing"
            errors.append("Custom structure path is empty or missing.")
            flags.append("custom_structure_missing")
    elif decision == "accept_custom_chain":
        curated_chain = override.get("curated_chain_id", "")
        position = override.get("curated_position", "") or original_position
        residue = residue_from_mapping(residue_index, original.get("pdb_id", ""), curated_chain, position) if curated_chain else ""
        chain_path = Path("data") / "structures" / "chain" / "current_internal" / f"{original.get('pdb_id', '').upper()}_{curated_chain}.pdb"
        base["chain_id"] = curated_chain or original.get("chain_id", "")
        base["resolved_position"] = position
        base["resolved_chain_structure_path"] = chain_path.as_posix() if curated_chain else ""
        base["resolved_structure_wt_aa"] = residue
        if curated_chain and residue == original.get("wild_type_aa", "") and (root / chain_path).exists():
            base["resolved_wt_residue_matches_mutation"] = "true"
            base["structure_resolution_status"] = "custom_chain_valid"
            base["allow_experimental_structure_jobs"] = True
            base["blocking_reason"] = ""
            flags.append("custom_chain_validated")
        else:
            base["structure_resolution_status"] = "failed"
            base["allow_experimental_structure_jobs"] = False
            base["blocking_reason"] = "custom_chain_validation_failed"
            errors.append("Custom chain could not be validated against residue_mapping and prepared chain file.")
            flags.append("custom_chain_validation_failed")
    elif decision == "exclude":
        base["resolved_wt_residue_matches_mutation"] = "false"
        base["structure_resolution_status"] = "excluded"
        base["allow_experimental_structure_jobs"] = False
        base["blocking_reason"] = "excluded_by_structure_override"
        flags.append("excluded_by_structure_override")
    elif decision == "needs_review":
        base["resolved_wt_residue_matches_mutation"] = "false"
        base["structure_resolution_status"] = "needs_review"
        base["allow_experimental_structure_jobs"] = False
        base["blocking_reason"] = "needs_manual_structure_review"
        flags.append("needs_manual_structure_review")

    if bool_str(override.get("exclude_from_experimental_structure_jobs", "")):
        flags.append("curator_requested_exclusion")
    if bool_str(override.get("allow_experimental_structure_jobs", "")) and not base["allow_experimental_structure_jobs"]:
        flags.append("allow_requested_but_validation_failed")
    base["structure_resolution_flags"] = ";".join(sorted(set(flags)))
    base["error_message"] = "; ".join([msg for msg in [base.get("error_message", "")] + errors if msg])
    return base


def update_jobs(
    jobs: list[dict[str, str]],
    resolutions_by_id: dict[str, dict[str, Any]],
    original_fieldnames: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    fieldnames = list(original_fieldnames)
    for field in JOB_EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    updated: list[dict[str, Any]] = []
    for job in jobs:
        row: dict[str, Any] = dict(job)
        resolution = resolutions_by_id.get(job.get("mutation_record_id", ""))
        if resolution:
            row["resolved_position"] = resolution["resolved_position"]
            row["structure_resolution_status"] = resolution["structure_resolution_status"]
            row["structure_resolution_flags"] = resolution["structure_resolution_flags"]
            row["override_applied"] = str(bool(resolution["override_applied"])).lower()
        else:
            row["resolved_position"] = ""
            row["structure_resolution_status"] = ""
            row["structure_resolution_flags"] = ""
            row["override_applied"] = "false"

        if (
            resolution
            and job.get("method") in {"foldx", "rosetta"}
            and job.get("structure_source") == "experimental"
        ):
            if resolution["allow_experimental_structure_jobs"]:
                row["input_ready"] = "true"
                row["input_status"] = "ready"
                row["experimental_structure_path"] = resolution["resolved_chain_structure_path"]
                row["blocking_reason"] = ""
            else:
                row["input_ready"] = "false"
                row["input_status"] = "blocked_for_structure"
                row["blocking_reason"] = resolution["blocking_reason"]
            if resolution["override_applied"]:
                row["notes"] = append_note(row.get("notes", ""), f"structure_override={resolution['structure_resolution_status']}")
        updated.append(row)
    return updated, fieldnames


def input_ready_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if bool_str(row.get("input_ready", "")))


def blocked_experimental_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("method") in {"foldx", "rosetta"}
        and row.get("structure_source") == "experimental"
        and not bool_str(row.get("input_ready", ""))
    )


def build_qc(
    template_rows: list[dict[str, str]],
    resolutions: list[dict[str, Any]],
    original_jobs: list[dict[str, str]],
    resolved_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    resolution_status = Counter(row["structure_resolution_status"] for row in resolutions)
    decision_counter = Counter(row.get("curator_decision", "") for row in template_rows)
    recommended_counter = Counter(row.get("recommended_action_from_audit", "") for row in template_rows)
    blocking_after = Counter(
        row.get("blocking_reason", "") or "none"
        for row in resolved_jobs
        if row.get("method") in {"foldx", "rosetta"}
        and row.get("structure_source") == "experimental"
        and not bool_str(row.get("input_ready", ""))
    )
    flags = Counter()
    for row in resolutions:
        for flag in str(row.get("structure_resolution_flags", "")).split(";"):
            if flag:
                flags[flag] += 1
    return {
        "input_summary": {
            "override_rows": len(template_rows),
            "unique_override_mutations": len({row.get("mutation_record_id", "") for row in template_rows}),
            "resolved_structure_rows": len(resolutions),
        },
        "decisions": {
            "count_by_curator_decision": dict(decision_counter),
            "count_by_recommended_action_from_audit": dict(recommended_counter),
            "accepted_remaps": resolution_status.get("remapped_valid", 0),
            "excluded": resolution_status.get("excluded", 0),
            "still_needs_review": resolution_status.get("needs_review", 0),
        },
        "validation": {
            "remap_validation_passed": resolution_status.get("remapped_valid", 0),
            "remap_validation_failed": flags.get("curated_remap_validation_failed", 0),
            "custom_structure_validation_passed": resolution_status.get("custom_structure_valid", 0),
            "custom_structure_validation_failed": flags.get("custom_structure_validation_failed", 0) + flags.get("custom_structure_missing", 0),
            "custom_chain_validation_passed": resolution_status.get("custom_chain_valid", 0),
            "custom_chain_validation_failed": flags.get("custom_chain_validation_failed", 0),
        },
        "jobs_readiness": {
            "total_jobs": len(resolved_jobs),
            "input_ready_jobs_before": input_ready_count(original_jobs),
            "input_ready_jobs_after": input_ready_count(resolved_jobs),
            "blocked_experimental_jobs_before": blocked_experimental_count(original_jobs),
            "blocked_experimental_jobs_after": blocked_experimental_count(resolved_jobs),
            "blocking_reasons_after": dict(blocking_after),
        },
        "warnings": {
            "accept_original_despite_mismatch": flags.get("accepted_original_despite_mismatch", 0),
            "possible_pdb_already_mutant_excluded": sum(
                1
                for row in template_rows
                if row.get("recommended_action_from_audit") == "possible_pdb_already_mutant"
                and row.get("curator_decision") == "exclude"
            ),
            "unresolved_manual_review_remains_blocked": resolution_status.get("needs_review", 0),
        },
    }


def markdown_report(qc: dict[str, Any]) -> str:
    lines = [
        "# Current Internal Structure Override QC",
        "",
        "This report describes only the derived override layer. Raw data, normalized mutations, and the original jobs-with-inputs table are unchanged.",
        "",
        "## Input Summary",
        "",
        f"- Override rows: {qc['input_summary']['override_rows']}",
        f"- Unique override mutations: {qc['input_summary']['unique_override_mutations']}",
        f"- Resolved structure rows: {qc['input_summary']['resolved_structure_rows']}",
        "",
        "## Decisions",
        "",
        f"- Count by curator_decision: `{json.dumps(qc['decisions']['count_by_curator_decision'], sort_keys=True)}`",
        f"- Count by recommended_action_from_audit: `{json.dumps(qc['decisions']['count_by_recommended_action_from_audit'], sort_keys=True)}`",
        f"- Accepted remaps: {qc['decisions']['accepted_remaps']}",
        f"- Excluded: {qc['decisions']['excluded']}",
        f"- Still needs review: {qc['decisions']['still_needs_review']}",
        "",
        "## Validation",
        "",
        f"- Remap validation passed: {qc['validation']['remap_validation_passed']}",
        f"- Remap validation failed: {qc['validation']['remap_validation_failed']}",
        f"- Custom structure validation passed/failed: {qc['validation']['custom_structure_validation_passed']} / {qc['validation']['custom_structure_validation_failed']}",
        f"- Custom chain validation passed/failed: {qc['validation']['custom_chain_validation_passed']} / {qc['validation']['custom_chain_validation_failed']}",
        "",
        "## Jobs Readiness After Overrides",
        "",
        f"- Total jobs: {qc['jobs_readiness']['total_jobs']}",
        f"- Input-ready jobs before: {qc['jobs_readiness']['input_ready_jobs_before']}",
        f"- Input-ready jobs after: {qc['jobs_readiness']['input_ready_jobs_after']}",
        f"- Blocked experimental jobs before: {qc['jobs_readiness']['blocked_experimental_jobs_before']}",
        f"- Blocked experimental jobs after: {qc['jobs_readiness']['blocked_experimental_jobs_after']}",
        f"- Blocking reasons after: `{json.dumps(qc['jobs_readiness']['blocking_reasons_after'], sort_keys=True)}`",
        "",
        "## Warnings",
        "",
        f"- accept_original despite mismatch: {qc['warnings']['accept_original_despite_mismatch']}",
        f"- possible PDB already mutant excluded: {qc['warnings']['possible_pdb_already_mutant_excluded']}",
        f"- unresolved manual_review remains blocked: {qc['warnings']['unresolved_manual_review_remains_blocked']}",
        "",
        "Manual decisions must be reviewed before using the resolved jobs table for execution.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--datasets-config", type=Path, default=DEFAULT_DATASETS_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.dataset != "current_internal":
        raise SystemExit("This override applier currently supports only --dataset current_internal")

    root = project_root_from(args.datasets_config)
    config = load_yaml(args.datasets_config)
    dataset = dataset_config(config, args.dataset)
    processed_dir = root / dataset["processed_dir"]
    reports_dir = root / dataset["reports_dir"]
    manifests_dir = root / "data" / "manifests"

    structure_manifest_path = manifests_dir / "current_internal_structure_manifest.csv"
    residue_mapping_path = manifests_dir / "current_internal_residue_mapping.csv"
    template_path = manifests_dir / "current_internal_structure_override_template.csv"
    jobs_path = processed_dir / "current_internal_recalc_jobs_with_inputs.csv"

    for path in (structure_manifest_path, residue_mapping_path, template_path, jobs_path):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path.relative_to(root).as_posix()}")

    structure_manifest = read_csv(structure_manifest_path)
    residue_mapping = read_csv(residue_mapping_path)
    template_rows = read_csv(template_path)
    jobs = read_csv(jobs_path)
    validate_template_rows(template_rows)

    with jobs_path.open("r", encoding="utf-8", newline="") as handle:
        job_fieldnames = list(csv.DictReader(handle).fieldnames or [])

    overrides_by_id: dict[str, dict[str, str]] = {}
    duplicate_ids: list[str] = []
    for row in template_rows:
        mutation_id = row.get("mutation_record_id", "")
        if mutation_id in overrides_by_id:
            duplicate_ids.append(mutation_id)
        overrides_by_id[mutation_id] = row
    if duplicate_ids:
        raise SystemExit(f"Duplicate mutation_record_id in override template: {', '.join(sorted(set(duplicate_ids)))}")

    residue_index = build_residue_index(residue_mapping)
    resolutions = [decide_resolution(row, overrides_by_id.get(row.get("mutation_record_id", "")), residue_index, root) for row in structure_manifest]
    resolutions_by_id = {row["mutation_record_id"]: row for row in resolutions}
    resolved_jobs, resolved_job_fields = update_jobs(jobs, resolutions_by_id, job_fieldnames)
    qc = build_qc(template_rows, resolutions, jobs, resolved_jobs)

    resolved_manifest_path = manifests_dir / "current_internal_structure_manifest_resolved.csv"
    resolved_jobs_path = processed_dir / "current_internal_recalc_jobs_resolved.csv"
    report_md_path = reports_dir / "current_internal_structure_override_qc.md"
    report_json_path = reports_dir / "current_internal_structure_override_qc.json"

    print(f"Dataset: {args.dataset}")
    print(f"Override rows: {len(template_rows)}")
    print(f"Input-ready jobs before: {qc['jobs_readiness']['input_ready_jobs_before']}")
    print(f"Input-ready jobs after: {qc['jobs_readiness']['input_ready_jobs_after']}")
    print(f"Blocked experimental jobs after: {qc['jobs_readiness']['blocked_experimental_jobs_after']}")
    print(f"Blocking reasons after: {json.dumps(qc['jobs_readiness']['blocking_reasons_after'], sort_keys=True)}")
    if args.dry_run:
        print("Dry run: resolved outputs were not written.")
        return 0

    write_csv(resolved_manifest_path, resolutions, RESOLVED_MANIFEST_FIELDS, args.force)
    write_csv(resolved_jobs_path, resolved_jobs, resolved_job_fields, args.force)
    write_text(report_json_path, json.dumps(qc, indent=2, ensure_ascii=False, sort_keys=True) + "\n", args.force)
    write_text(report_md_path, markdown_report(qc), args.force)
    print(f"Resolved structure manifest: {relative_or_original(resolved_manifest_path, root)}")
    print(f"Resolved jobs table: {relative_or_original(resolved_jobs_path, root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
