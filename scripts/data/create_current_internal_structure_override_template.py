#!/usr/bin/env python3
"""Create a manual decision template for current_internal structure mismatches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


DEFAULT_DATASETS_CONFIG = Path("configs/datasets.yaml")
TEMPLATE_FIELDS = [
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
    "candidate_corrected_position",
    "candidate_reason",
    "recommended_action_from_audit",
    "curator_decision",
    "curated_position",
    "curated_chain_id",
    "curated_structure_path",
    "exclude_from_experimental_structure_jobs",
    "allow_experimental_structure_jobs",
    "decision_reason",
    "curator",
    "decision_date",
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
        raise SystemExit(f"Refusing to overwrite existing template without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def default_decision(recommended_action: str) -> tuple[str, str, str, str]:
    if recommended_action == "possible_pdb_already_mutant":
        return (
            "exclude",
            "true",
            "false",
            "PDB residue already matches the mutant amino acid; keep experimental-structure jobs excluded unless manually overridden.",
        )
    if recommended_action == "candidate_position_remap":
        return (
            "needs_review",
            "true",
            "false",
            "Candidate remap requires curator review before experimental-structure jobs are allowed.",
        )
    return (
        "needs_review",
        "true",
        "false",
        "Ambiguous structure mismatch requires manual review before experimental-structure jobs are allowed.",
    )


def build_template_rows(audit_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    template_rows: list[dict[str, Any]] = []
    for row in audit_rows:
        recommended = row.get("recommended_action", "")
        curator_decision, exclude_jobs, allow_jobs, decision_reason = default_decision(recommended)
        curated_position = row.get("candidate_corrected_position", "") if recommended == "candidate_position_remap" else ""
        template_rows.append(
            {
                "dataset_name": row.get("dataset_name", ""),
                "mutation_record_id": row.get("mutation_record_id", ""),
                "uniprot_id": row.get("uniprot_id", ""),
                "protein_id": row.get("protein_id", ""),
                "pdb_id": row.get("pdb_id", ""),
                "chain_id": row.get("chain_id", ""),
                "mutation": row.get("mutation", ""),
                "wild_type_aa": row.get("wild_type_aa", ""),
                "position": row.get("position", ""),
                "mutant_aa": row.get("mutant_aa", ""),
                "structure_wt_aa": row.get("structure_wt_aa", ""),
                "candidate_corrected_position": row.get("candidate_corrected_position", ""),
                "candidate_reason": row.get("candidate_reason", ""),
                "recommended_action_from_audit": recommended,
                "curator_decision": curator_decision,
                "curated_position": curated_position,
                "curated_chain_id": "",
                "curated_structure_path": "",
                "exclude_from_experimental_structure_jobs": exclude_jobs,
                "allow_experimental_structure_jobs": allow_jobs,
                "decision_reason": decision_reason,
                "curator": "",
                "decision_date": "",
                "notes": row.get("notes", ""),
            }
        )
    return template_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--datasets-config", type=Path, default=DEFAULT_DATASETS_CONFIG)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.dataset != "current_internal":
        raise SystemExit("This template builder currently supports only --dataset current_internal")

    root = project_root_from(args.datasets_config)
    config = load_yaml(args.datasets_config)
    dataset_config(config, args.dataset)
    audit_path = root / "data" / "manifests" / "current_internal_structure_mismatch_audit.csv"
    output_path = root / "data" / "manifests" / "current_internal_structure_override_template.csv"
    if not audit_path.exists():
        raise SystemExit(f"Missing required audit CSV: {audit_path.relative_to(root).as_posix()}")

    audit_rows = read_csv(audit_path)
    template_rows = build_template_rows(audit_rows)
    write_csv(output_path, template_rows, TEMPLATE_FIELDS, args.force)

    print(f"Dataset: {args.dataset}")
    print(f"Audit rows read: {len(audit_rows)}")
    print(f"Override template rows written: {len(template_rows)}")
    print(f"Output: {output_path.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
