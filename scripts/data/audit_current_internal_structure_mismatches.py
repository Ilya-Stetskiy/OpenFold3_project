#!/usr/bin/env python3
"""Audit current_internal WT residue mismatches without modifying source tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


DEFAULT_DATASETS_CONFIG = Path("configs/datasets.yaml")
AUDIT_ISSUES = {"wt_residue_mismatch", "missing_residue_near_mutation", "insertion_code_ambiguous", "mutation_site_not_found"}
AUDIT_FIELDS = [
    "dataset_name",
    "mutation_record_id",
    "uniprot_id",
    "protein_id",
    "pdb_id",
    "chain_id",
    "mutation",
    "wild_type_aa",
    "position",
    "mutant_aa",
    "structure_wt_aa",
    "experimental_ddg",
    "chain_structure_path",
    "raw_structure_path",
    "issue_type",
    "structure_quality_flags",
    "nearby_sequence_window",
    "nearby_positions",
    "wild_type_found_within_plus_minus_10",
    "candidate_corrected_position",
    "candidate_reason",
    "same_mutant_already_in_structure",
    "insertion_code_candidate",
    "recommended_action",
    "affected_jobs",
    "notes",
]


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


def canonical_pdb_id(value: str) -> str:
    return value.strip().upper()


def issue_set(flags: str) -> set[str]:
    return {flag for flag in flags.split(";") if flag}


def int_or_none(value: str) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def residue_key(row: dict[str, str]) -> tuple[str, str]:
    return row["original_residue_id"], row.get("original_insertion_code", "")


def residue_label(row: dict[str, str]) -> str:
    ins = row.get("original_insertion_code", "")
    suffix = ins if ins else ""
    return f"{row['original_residue_id']}{suffix}:{row['residue_name_1letter']}"


def window_residues(residues: list[dict[str, str]], center: int, radius: int) -> list[dict[str, str]]:
    selected = []
    for row in residues:
        pos = int_or_none(row.get("original_residue_id", ""))
        if pos is not None and center - radius <= pos <= center + radius:
            selected.append(row)
    return selected


def build_audit_rows(
    dataset_name: str,
    structure_manifest: list[dict[str, str]],
    residue_mapping: list[dict[str, str]],
    mutations_by_id: dict[str, dict[str, str]],
    jobs_by_mutation: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    residues_by_chain: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in residue_mapping:
        residues_by_chain[(canonical_pdb_id(row["pdb_id"]), row["chain_id"])].append(row)

    audit_rows: list[dict[str, Any]] = []
    for row in structure_manifest:
        flags = issue_set(row.get("structure_quality_flags", ""))
        relevant = flags & AUDIT_ISSUES
        if not relevant:
            continue
        mutation_id = row["mutation_record_id"]
        mutation = mutations_by_id.get(mutation_id, {})
        pdb_id = canonical_pdb_id(row["pdb_id"])
        chain_id = row["chain_id"]
        position = int_or_none(row["position"])
        residues = residues_by_chain.get((pdb_id, chain_id), [])
        nearby5 = window_residues(residues, position, 5) if position is not None else []
        nearby10 = window_residues(residues, position, 10) if position is not None else []
        wt = row.get("wild_type_aa", "")
        mutant = row.get("mutant_aa", "")
        wt_candidates = [item for item in nearby10 if item.get("residue_name_1letter") == wt]
        same_number_with_ins = []
        if position is not None:
            same_number_with_ins = [
                item for item in residues if int_or_none(item.get("original_residue_id", "")) == position and item.get("original_insertion_code")
            ]
        same_mutant = bool(mutant and row.get("structure_wt_aa") == mutant)
        candidate_position = ""
        candidate_reason = ""
        if len(wt_candidates) == 1:
            candidate_position = wt_candidates[0]["original_residue_id"] + (wt_candidates[0].get("original_insertion_code", "") or "")
            candidate_reason = "single nearby residue has expected wild-type amino acid"
        elif len(wt_candidates) > 1:
            candidate_reason = "multiple nearby residues have expected wild-type amino acid"

        if same_mutant:
            recommended = "possible_pdb_already_mutant"
        elif len(wt_candidates) == 1 and not same_number_with_ins:
            recommended = "candidate_position_remap"
        elif len(wt_candidates) > 1 or same_number_with_ins:
            recommended = "manual_review"
        elif "wt_residue_mismatch" in flags:
            recommended = "exclude_from_experimental_structure_jobs"
        else:
            recommended = "keep_blocked"

        affected_jobs = [
            job["job_id"]
            for job in jobs_by_mutation.get(mutation_id, [])
            if job.get("structure_source") == "experimental" and job.get("method") in {"foldx", "rosetta"}
        ]
        issue_type = ";".join(sorted(relevant))
        notes = []
        if "missing_residue_near_mutation" in flags:
            notes.append("Neighboring residue gap detected around mutation site.")
        if "wt_residue_mismatch" in flags:
            notes.append("Experimental structure residue does not match normalized mutation WT amino acid.")
        if not wt_candidates:
            notes.append("Wild-type amino acid not found within +/-10 residue-number window.")
        audit_rows.append(
            {
                "dataset_name": dataset_name,
                "mutation_record_id": mutation_id,
                "uniprot_id": row.get("uniprot_id", ""),
                "protein_id": row.get("protein_id", ""),
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "mutation": row.get("mutation", ""),
                "wild_type_aa": wt,
                "position": row.get("position", ""),
                "mutant_aa": mutant,
                "structure_wt_aa": row.get("structure_wt_aa", ""),
                "experimental_ddg": mutation.get("experimental_ddg", ""),
                "chain_structure_path": row.get("chain_structure_path", ""),
                "raw_structure_path": row.get("raw_structure_path", ""),
                "issue_type": issue_type,
                "structure_quality_flags": row.get("structure_quality_flags", ""),
                "nearby_sequence_window": "".join(item.get("residue_name_1letter", "X") for item in nearby5),
                "nearby_positions": ";".join(residue_label(item) for item in nearby5),
                "wild_type_found_within_plus_minus_10": str(bool(wt_candidates)).lower(),
                "candidate_corrected_position": candidate_position,
                "candidate_reason": candidate_reason,
                "same_mutant_already_in_structure": str(same_mutant).lower(),
                "insertion_code_candidate": str(bool(same_number_with_ins)).lower(),
                "recommended_action": recommended,
                "affected_jobs": ";".join(affected_jobs),
                "notes": " ".join(notes),
            }
        )
    return audit_rows


def build_qc(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatch_rows = [row for row in audit_rows if "wt_residue_mismatch" in row["issue_type"]]
    return {
        "total_audit_rows": len(audit_rows),
        "total_mismatch_mutations": len(mismatch_rows),
        "mismatch_count_by_pdb_id": dict(Counter(row["pdb_id"] for row in mismatch_rows)),
        "mismatch_count_by_uniprot_id": dict(Counter(row["uniprot_id"] for row in mismatch_rows)),
        "same_mutant_already_in_structure_count": sum(1 for row in audit_rows if row["same_mutant_already_in_structure"] == "true"),
        "wild_type_found_within_plus_minus_10_count": sum(1 for row in audit_rows if row["wild_type_found_within_plus_minus_10"] == "true"),
        "count_by_recommended_action": dict(Counter(row["recommended_action"] for row in audit_rows)),
        "examples_top20": audit_rows[:20],
    }


def write_markdown(path: Path, qc: dict[str, Any], force: bool) -> None:
    lines = [
        "# Current Internal Structure Mismatch Audit",
        "",
        f"- Total audit rows: {qc['total_audit_rows']}",
        f"- WT mismatch mutations: {qc['total_mismatch_mutations']}",
        f"- structure_wt_aa == mutant_aa: {qc['same_mutant_already_in_structure_count']}",
        f"- wild_type_aa found within +/-10: {qc['wild_type_found_within_plus_minus_10_count']}",
        f"- Recommended actions: `{json.dumps(qc['count_by_recommended_action'], sort_keys=True)}`",
        "",
        "## Mismatches By PDB ID",
        "",
        json.dumps(qc["mismatch_count_by_pdb_id"], indent=2, sort_keys=True),
        "",
        "## Mismatches By UniProt",
        "",
        json.dumps(qc["mismatch_count_by_uniprot_id"], indent=2, sort_keys=True),
        "",
        "## Top Examples",
        "",
        "| mutation_record_id | pdb_id | chain | mutation | structure_wt_aa | recommended_action | notes |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in qc["examples_top20"]:
        lines.append(
            f"| {row['mutation_record_id']} | {row['pdb_id']} | {row['chain_id']} | {row['mutation']} | {row['structure_wt_aa']} | {row['recommended_action']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Warning",
            "",
            "This audit does not modify normalized mutations, jobs, structures, or raw data. Recommended actions require manual review before any correction or exclusion policy is applied.",
            "",
        ]
    )
    write_text(path, "\n".join(lines), force)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--datasets-config", type=Path, default=DEFAULT_DATASETS_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.dataset != "current_internal":
        raise SystemExit("This audit currently supports only --dataset current_internal")

    root = project_root_from(args.datasets_config)
    config = load_yaml(args.datasets_config)
    dataset = dataset_config(config, args.dataset)
    processed_dir = root / dataset["processed_dir"]
    reports_dir = root / dataset["reports_dir"]
    manifests_dir = root / "data" / "manifests"

    structure_manifest_path = manifests_dir / "current_internal_structure_manifest.csv"
    structure_errors_path = manifests_dir / "current_internal_structure_errors.csv"
    residue_mapping_path = manifests_dir / "current_internal_residue_mapping.csv"
    mutations_path = processed_dir / "current_internal_mutations.csv"
    jobs_path = processed_dir / "current_internal_recalc_jobs_with_inputs.csv"

    for path in (structure_manifest_path, structure_errors_path, residue_mapping_path, mutations_path, jobs_path):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path.relative_to(root).as_posix()}")

    structure_manifest = read_csv(structure_manifest_path)
    structure_errors = read_csv(structure_errors_path)
    residue_mapping = read_csv(residue_mapping_path)
    mutations = read_csv(mutations_path)
    jobs = read_csv(jobs_path)
    mutations_by_id = {row["mutation_record_id"]: row for row in mutations}
    jobs_by_mutation: dict[str, list[dict[str, str]]] = defaultdict(list)
    for job in jobs:
        jobs_by_mutation[job["mutation_record_id"]].append(job)

    error_ids = {row["mutation_record_id"] for row in structure_errors if issue_set(row.get("error_type", "")) & AUDIT_ISSUES}
    manifest_subset = [
        row for row in structure_manifest if row["mutation_record_id"] in error_ids or (issue_set(row.get("structure_quality_flags", "")) & AUDIT_ISSUES)
    ]
    audit_rows = build_audit_rows(args.dataset, manifest_subset, residue_mapping, mutations_by_id, jobs_by_mutation)
    qc = build_qc(audit_rows)

    audit_csv_path = manifests_dir / "current_internal_structure_mismatch_audit.csv"
    audit_json_path = manifests_dir / "current_internal_structure_mismatch_audit.json"
    report_md_path = reports_dir / "current_internal_structure_mismatch_audit.md"
    report_json_path = reports_dir / "current_internal_structure_mismatch_audit.json"

    print(f"Dataset: {args.dataset}")
    print(f"Audit rows: {len(audit_rows)}")
    print(f"WT mismatch mutations: {qc['total_mismatch_mutations']}")
    print(f"Recommended actions: {json.dumps(qc['count_by_recommended_action'], sort_keys=True)}")
    if args.dry_run:
        print("Dry run: no audit outputs were written.")
        return 0

    write_csv(audit_csv_path, audit_rows, AUDIT_FIELDS, args.force)
    audit_payload = {"audit_rows": audit_rows, "qc": qc}
    write_text(audit_json_path, json.dumps(audit_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", args.force)
    write_text(report_json_path, json.dumps(qc, indent=2, ensure_ascii=False, sort_keys=True) + "\n", args.force)
    write_markdown(report_md_path, qc, args.force)
    print("Audit outputs written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
