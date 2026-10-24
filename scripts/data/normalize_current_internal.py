#!/usr/bin/env python3
"""Normalize current_internal raw DDG rows into mutation and recalc job tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


DEFAULT_DATASETS_CONFIG = Path("configs/datasets.yaml")
DEFAULT_ALIASES_CONFIG = Path("configs/schemas/ddg_column_aliases.yaml")
DEFAULT_METHODS = ("foldx", "rosetta", "esm2")
DEFAULT_STRUCTURE_SOURCES = ("experimental", "openfold", "sequence_only")
METHOD_STRUCTURE_SOURCES = {
    "foldx": ("experimental", "openfold"),
    "rosetta": ("experimental", "openfold"),
    "esm2": ("sequence_only",),
}
KNOWN_METHODS = {"foldx", "rosetta", "esm2", "openfold"}
KNOWN_STRUCTURE_SOURCES = {"experimental", "openfold", "sequence_only"}
MUTATION_RE = re.compile(r"^([A-Z])([0-9]+)([A-Z])$")
PROTEIN_ID_RE = re.compile(r"^([^_]+)_([0-9][A-Za-z0-9]{3})_([^_]+)_([A-Z][0-9]+[A-Z])$")


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


def expected_raw_path(dataset: dict[str, Any]) -> Path:
    for source in dataset.get("sources", []):
        download = source.get("download", {}) if isinstance(source, dict) else {}
        if download.get("mode") == "local_file" and download.get("expected_path"):
            return Path(download["expected_path"])
    return Path(dataset["raw_dir"]) / "results.csv"


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load_aliases(path: Path) -> dict[str, list[str]]:
    data = load_yaml(path)
    columns = data.get("columns", {})
    aliases: dict[str, list[str]] = {}
    for canonical, spec in columns.items():
        values = [canonical]
        if isinstance(spec, dict):
            values.extend(str(item) for item in spec.get("aliases", []))
        aliases[canonical] = values
    return aliases


def detect_columns(header: list[str], aliases: dict[str, list[str]]) -> tuple[dict[str, str], list[str]]:
    normalized_to_original = {normalize_name(column): column for column in header}
    mapping: dict[str, str] = {}
    for canonical, names in aliases.items():
        for name in names:
            normalized = normalize_name(name)
            if normalized in normalized_to_original:
                mapping[canonical] = normalized_to_original[normalized]
                break
    missing = [key for key in ("protein_id", "pdb_id", "chain_id", "mutation", "experimental_ddg") if key not in mapping]
    return mapping, missing


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get(row: dict[str, str], mapping: dict[str, str], canonical: str) -> str:
    source = mapping.get(canonical)
    return clean(row.get(source, "")) if source else ""


def parse_float(value: str) -> float | None:
    value = clean(value)
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def parse_mutation(value: str) -> tuple[str, int, str] | None:
    match = MUTATION_RE.match(clean(value).upper())
    if not match:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


def parse_compound_protein_id(value: str) -> dict[str, str]:
    match = PROTEIN_ID_RE.match(clean(value).upper())
    if not match:
        return {}
    return {
        "uniprot_id": match.group(1),
        "pdb_id": match.group(2),
        "chain_id": match.group(3),
        "mutation": match.group(4),
    }


def normalize_method(value: str) -> str:
    value = normalize_name(value)
    aliases = {"fold_x": "foldx", "esm_2": "esm2", "open_fold": "openfold"}
    return aliases.get(value, value)


def normalize_structure_source(value: str) -> str:
    value = normalize_name(value)
    aliases = {"exp": "experimental", "pdb": "experimental", "structure": "experimental", "seq": "sequence_only"}
    return aliases.get(value, value)


def read_raw_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return header, rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing output without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def ddg_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def stable_record_id(index: int) -> str:
    return f"current_internal_mut_{index:06d}"


def job_id(method: str, structure_source: str, mutation_record_id: str) -> str:
    return f"current_internal__{method}__{structure_source}__{mutation_record_id}"


def build_outputs(
    dataset_name: str,
    raw_rel_path: Path,
    raw_rows: list[dict[str, str]],
    header: list[str],
    mapping: dict[str, str],
    missing_expected_columns: list[str],
    selected_methods: tuple[str, ...],
    selected_structure_sources: tuple[str, ...],
    results_dir: Path,
) -> dict[str, Any]:
    parsed_rows: list[dict[str, Any]] = []
    legacy_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    legacy_failures: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    extracted_structure_rows = 0
    protein_id_mutation_mismatch_rows = 0

    for idx, row in enumerate(raw_rows, start=1):
        source_row_id = f"raw_row_{idx:06d}"
        protein_id = get(row, mapping, "protein_id")
        parsed_protein_id = parse_compound_protein_id(protein_id)
        uniprot_id = parsed_protein_id.get("uniprot_id", "")
        pdb_id = get(row, mapping, "pdb_id")
        chain_id = get(row, mapping, "chain_id")
        mutation = get(row, mapping, "mutation").upper()
        mutation_from_protein_id = parsed_protein_id.get("mutation", "")
        if not pdb_id and parsed_protein_id.get("pdb_id"):
            pdb_id = parsed_protein_id["pdb_id"]
        if not chain_id and parsed_protein_id.get("chain_id"):
            chain_id = parsed_protein_id["chain_id"]
        if not mutation and mutation_from_protein_id:
            mutation = mutation_from_protein_id
        if parsed_protein_id.get("pdb_id") or parsed_protein_id.get("chain_id"):
            extracted_structure_rows += 1
        exp_text = get(row, mapping, "experimental_ddg")
        exp_ddg = parse_float(exp_text)
        legacy_unit = get(row, mapping, "unit")
        experimental_ddg_unit = "kcal/mol"
        method = normalize_method(get(row, mapping, "method"))
        structure_source = normalize_structure_source(get(row, mapping, "structure_source"))
        status = normalize_name(get(row, mapping, "status")) or "unknown"
        predicted_score = get(row, mapping, "predicted_score")
        error = get(row, mapping, "error")
        raw_output = get(row, mapping, "raw_output")
        output_path = get(row, mapping, "output_path")
        parsed_mutation = parse_mutation(mutation)
        flags: list[str] = []

        if mutation and mutation_from_protein_id and mutation != mutation_from_protein_id:
            flags.append("protein_id_mutation_mismatch")
            protein_id_mutation_mismatch_rows += 1
        if not mutation:
            flags.append("missing_mutation")
        elif parsed_mutation is None:
            flags.append("invalid_mutation_format")
        if exp_ddg is None:
            flags.append("missing_experimental_ddg")
        missing_key_fields = [
            name
            for name, value in (("protein_id", protein_id), ("pdb_id", pdb_id), ("chain_id", chain_id), ("mutation", mutation))
            if not value
        ]
        if missing_key_fields:
            flags.append("weak_mutation_key")
        if method and method not in KNOWN_METHODS:
            flags.append("unknown_method")
        if structure_source and structure_source not in KNOWN_STRUCTURE_SOURCES:
            flags.append("unknown_structure_source")

        parsed_row = {
            "source_row_id": source_row_id,
            "protein_id": protein_id,
            "uniprot_id": uniprot_id,
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "mutation": mutation,
            "mutation_from_protein_id": mutation_from_protein_id,
            "experimental_ddg": "" if exp_ddg is None else exp_ddg,
            "experimental_ddg_unit": experimental_ddg_unit,
            "method": method,
            "structure_source": structure_source,
            "status": status,
            "predicted_score": predicted_score,
            "quality_flags": ";".join(flags),
            "raw_columns_json": json.dumps(row, ensure_ascii=False, sort_keys=True),
        }
        parsed_rows.append(parsed_row)

        for flag in flags:
            issues.append(
                {
                    "source_row_id": source_row_id,
                    "issue_type": flag,
                    "protein_id": protein_id,
                    "uniprot_id": uniprot_id,
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "mutation": mutation,
                    "message": f"{flag} in source row",
                    "raw_columns_json": parsed_row["raw_columns_json"],
                }
            )

        if method or predicted_score or status != "unknown" or error or raw_output or output_path:
            legacy_row = {
                "dataset_name": dataset_name,
                "legacy_row_id": source_row_id,
                "mutation_record_id": "",
                "protein_id": protein_id,
                "uniprot_id": uniprot_id,
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "mutation": mutation,
                "experimental_ddg": "" if exp_ddg is None else exp_ddg,
                "legacy_method": method,
                "legacy_structure_source": structure_source,
                "legacy_status": status,
                "legacy_predicted_score": predicted_score,
                "legacy_unit": legacy_unit,
                "legacy_error": error,
                "legacy_raw_output_path": output_path or raw_output,
                "quality_flags": ";".join(flags),
                "notes": "Legacy output archived for regression/debug comparison only.",
            }
            legacy_rows.append(legacy_row)
            if status in {"failed", "failure", "error"}:
                legacy_failures.append(legacy_row.copy())

        if parsed_mutation is not None and exp_ddg is not None:
            grouped[(protein_id, uniprot_id, pdb_id, chain_id, mutation)].append(
                {
                    "source_row_id": source_row_id,
                    "protein_id": protein_id,
                    "uniprot_id": uniprot_id,
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "mutation": mutation,
                    "experimental_ddg": exp_ddg,
                    "experimental_ddg_unit": experimental_ddg_unit,
                    "method": method,
                    "flags": flags,
                }
            )

    mutation_rows: list[dict[str, Any]] = []
    mutation_id_by_key: dict[tuple[str, str, str, str, str], str] = {}
    for index, (key, rows) in enumerate(sorted(grouped.items()), start=1):
        protein_id, uniprot_id, pdb_id, chain_id, mutation = key
        record_id = stable_record_id(index)
        mutation_id_by_key[key] = record_id
        parsed_mutation = parse_mutation(mutation)
        assert parsed_mutation is not None
        wt, position, mut = parsed_mutation
        values = [row["experimental_ddg"] for row in rows]
        flags = sorted({flag for row in rows for flag in row["flags"] if flag})
        notes = []
        if max(values) - min(values) > 1e-6:
            flags.append("conflicting_experimental_ddg")
            notes.append(f"experimental_ddg min={min(values)} max={max(values)} count={len(values)}; median used")
            issues.append(
                {
                    "source_row_id": ";".join(row["source_row_id"] for row in rows),
                    "issue_type": "conflicting_experimental_ddg",
                    "protein_id": protein_id,
                    "uniprot_id": uniprot_id,
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "mutation": mutation,
                    "message": notes[-1],
                    "raw_columns_json": "",
                }
            )
        methods = sorted({row["method"] for row in rows if row["method"]})
        mutation_rows.append(
            {
                "dataset_name": dataset_name,
                "mutation_record_id": record_id,
                "source_dataset": raw_rel_path.as_posix(),
                "source_record_ids": ";".join(row["source_row_id"] for row in rows),
                "protein_id": protein_id,
                "uniprot_id": uniprot_id,
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "wild_type_aa": wt,
                "position": position,
                "mutant_aa": mut,
                "mutation": mutation,
                "experimental_ddg": median(values),
                "experimental_ddg_unit": "kcal/mol",
                "ddg_sign_convention": "ddG = G_mutant - G_wildtype",
                "has_legacy_method_outputs": bool(methods),
                "legacy_methods_available": ";".join(methods),
                "n_legacy_rows": len(rows),
                "quality_flags": ";".join(sorted(set(flags))),
                "notes": "; ".join(notes),
            }
        )

    for legacy in legacy_rows:
        key = (legacy["protein_id"], legacy["uniprot_id"], legacy["pdb_id"], legacy["chain_id"], legacy["mutation"])
        legacy["mutation_record_id"] = mutation_id_by_key.get(key, "")

    recalc_jobs: list[dict[str, Any]] = []
    for mutation in mutation_rows:
        for method in selected_methods:
            for structure_source in METHOD_STRUCTURE_SOURCES[method]:
                if structure_source not in selected_structure_sources:
                    continue
                requires_structure = method in {"foldx", "rosetta"}
                requires_sequence = method == "esm2" or structure_source == "openfold"
                mutation_record_id = mutation["mutation_record_id"]
                recalc_jobs.append(
                    {
                        "dataset_name": dataset_name,
                        "job_id": job_id(method, structure_source, mutation_record_id),
                        "mutation_record_id": mutation_record_id,
                        "protein_id": mutation["protein_id"],
                        "uniprot_id": mutation["uniprot_id"],
                        "pdb_id": mutation["pdb_id"],
                        "chain_id": mutation["chain_id"],
                        "mutation": mutation["mutation"],
                        "experimental_ddg": mutation["experimental_ddg"],
                        "method": method,
                        "structure_source": structure_source,
                        "requires_structure": str(requires_structure).lower(),
                        "requires_sequence": str(requires_sequence).lower(),
                        "job_status": "pending",
                        "expected_output_dir": (results_dir / method / structure_source / mutation_record_id).as_posix() + "/",
                        "notes": "",
                    }
                )

    ddg_values = [float(row["experimental_ddg"]) for row in mutation_rows if row["experimental_ddg"] != ""]
    qc = {
        "input_summary": {
            "input_csv": raw_rel_path.as_posix(),
            "raw_rows": len(raw_rows),
            "detected_columns": header,
            "column_mapping": mapping,
            "missing_expected_columns": missing_expected_columns,
        },
        "mutation_extraction": {
            "unique_mutation_records": len(mutation_rows),
            "valid_single_site_mutations": len(mutation_rows),
            "excluded_rows": len(raw_rows) - sum(len(rows) for rows in grouped.values()),
            "n_proteins": len({row["protein_id"] for row in mutation_rows if row["protein_id"]}),
            "n_pdb_ids": len({row["pdb_id"] for row in mutation_rows if row["pdb_id"]}),
            "top_proteins_by_mutation_count": Counter(row["protein_id"] for row in mutation_rows if row["protein_id"]).most_common(10),
            "pdb_id_distribution": Counter(row["pdb_id"] for row in mutation_rows if row["pdb_id"]).most_common(),
            "chain_distribution": Counter(row["chain_id"] for row in mutation_rows if row["chain_id"]).most_common(),
            "pdb_chain_extracted_from_protein_id_rows": extracted_structure_rows,
            "weak_mutation_key_count": sum(1 for row in mutation_rows if "weak_mutation_key" in row["quality_flags"].split(";")),
            "protein_id_mutation_mismatch_count": protein_id_mutation_mismatch_rows,
            "ddg_distribution": ddg_summary(ddg_values),
            "stabilizing_count": sum(1 for value in ddg_values if value < 0),
            "destabilizing_count": sum(1 for value in ddg_values if value > 0),
            "neutral_count": sum(1 for value in ddg_values if value == 0),
        },
        "legacy_outputs": {
            "legacy_rows": len(legacy_rows),
            "methods_detected": sorted({row["legacy_method"] for row in legacy_rows if row["legacy_method"]}),
            "structure_sources_detected": sorted({row["legacy_structure_source"] for row in legacy_rows if row["legacy_structure_source"]}),
            "statuses_detected": dict(Counter(row["legacy_status"] for row in legacy_rows if row["legacy_status"])),
            "failed_rows_count": len(legacy_failures),
        },
        "recalc_jobs": {
            "total_jobs": len(recalc_jobs),
            "jobs_by_method": dict(Counter(row["method"] for row in recalc_jobs)),
            "jobs_by_structure_source": dict(Counter(row["structure_source"] for row in recalc_jobs)),
            "jobs_requiring_structure": sum(1 for row in recalc_jobs if row["requires_structure"] == "true"),
            "jobs_requiring_sequence": sum(1 for row in recalc_jobs if row["requires_sequence"] == "true"),
        },
        "issues": {
            "counts_by_issue_type": dict(Counter(row["issue_type"] for row in issues)),
            "top_examples": issues[:10],
            "normalization_issues_csv": "data/interim/current_internal/current_internal_normalization_issues.csv",
        },
        "warning": "Legacy results are archived for comparison only. New article metrics must be computed from newly generated results.",
    }

    return {
        "parsed_rows": parsed_rows,
        "legacy_rows": legacy_rows,
        "issues": issues,
        "legacy_failures": legacy_failures,
        "mutation_rows": mutation_rows,
        "recalc_jobs": recalc_jobs,
        "qc": qc,
    }


def write_qc_markdown(path: Path, qc: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing output without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Current internal normalization QC",
        "",
        "## Input Summary",
        f"- Input CSV: `{qc['input_summary']['input_csv']}`",
        f"- Raw rows: {qc['input_summary']['raw_rows']}",
        f"- Detected columns: {', '.join(qc['input_summary']['detected_columns'])}",
        f"- Missing expected columns: {', '.join(qc['input_summary']['missing_expected_columns']) or 'none'}",
        "",
        "## Mutation Extraction",
        f"- Unique mutation records: {qc['mutation_extraction']['unique_mutation_records']}",
        f"- Valid single-site mutations: {qc['mutation_extraction']['valid_single_site_mutations']}",
        f"- Excluded rows: {qc['mutation_extraction']['excluded_rows']}",
        f"- Proteins: {qc['mutation_extraction']['n_proteins']}",
        f"- PDB IDs: {qc['mutation_extraction']['n_pdb_ids']}",
        f"- Rows with PDB/chain extracted from protein_id: {qc['mutation_extraction']['pdb_chain_extracted_from_protein_id_rows']}",
        f"- Weak mutation keys: {qc['mutation_extraction']['weak_mutation_key_count']}",
        f"- protein_id/mutation mismatches: {qc['mutation_extraction']['protein_id_mutation_mismatch_count']}",
        f"- PDB ID distribution: `{json.dumps(qc['mutation_extraction']['pdb_id_distribution'])}`",
        f"- Chain distribution: `{json.dumps(qc['mutation_extraction']['chain_distribution'])}`",
        f"- DDG distribution: `{json.dumps(qc['mutation_extraction']['ddg_distribution'], sort_keys=True)}`",
        f"- Stabilizing / destabilizing / neutral: {qc['mutation_extraction']['stabilizing_count']} / {qc['mutation_extraction']['destabilizing_count']} / {qc['mutation_extraction']['neutral_count']}",
        "",
        "## Legacy Outputs",
        f"- Legacy rows: {qc['legacy_outputs']['legacy_rows']}",
        f"- Methods detected: {', '.join(qc['legacy_outputs']['methods_detected']) or 'none'}",
        f"- Structure sources detected: {', '.join(qc['legacy_outputs']['structure_sources_detected']) or 'none'}",
        f"- Statuses detected: `{json.dumps(qc['legacy_outputs']['statuses_detected'], sort_keys=True)}`",
        f"- Failed rows count: {qc['legacy_outputs']['failed_rows_count']}",
        "",
        "## Recalc Jobs",
        f"- Total jobs: {qc['recalc_jobs']['total_jobs']}",
        f"- Jobs by method: `{json.dumps(qc['recalc_jobs']['jobs_by_method'], sort_keys=True)}`",
        f"- Jobs by structure source: `{json.dumps(qc['recalc_jobs']['jobs_by_structure_source'], sort_keys=True)}`",
        f"- Jobs requiring structure: {qc['recalc_jobs']['jobs_requiring_structure']}",
        f"- Jobs requiring sequence: {qc['recalc_jobs']['jobs_requiring_sequence']}",
        "",
        "## Issues",
        f"- Counts by issue type: `{json.dumps(qc['issues']['counts_by_issue_type'], sort_keys=True)}`",
        f"- Issues CSV: `{qc['issues']['normalization_issues_csv']}`",
        "",
        "## Warning",
        qc["warning"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--datasets-config", type=Path, default=DEFAULT_DATASETS_CONFIG)
    parser.add_argument("--aliases-config", type=Path, default=DEFAULT_ALIASES_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--structure-sources", nargs="+", default=list(DEFAULT_STRUCTURE_SOURCES))
    args = parser.parse_args()

    if args.dataset != "current_internal":
        raise SystemExit("This normalizer currently supports only --dataset current_internal")

    methods = tuple(normalize_method(method) for method in args.methods)
    invalid_methods = [method for method in methods if method not in METHOD_STRUCTURE_SOURCES]
    if invalid_methods:
        raise SystemExit(f"Unsupported methods: {', '.join(invalid_methods)}")

    structure_sources = tuple(normalize_structure_source(source) for source in args.structure_sources)
    invalid_sources = [source for source in structure_sources if source not in DEFAULT_STRUCTURE_SOURCES]
    if invalid_sources:
        raise SystemExit(f"Unsupported structure sources: {', '.join(invalid_sources)}")

    config = load_yaml(args.datasets_config)
    root = project_root_from(args.datasets_config)
    dataset = dataset_config(config, args.dataset)
    raw_rel_path = expected_raw_path(dataset)
    raw_path = root / raw_rel_path
    interim_dir = root / dataset["interim_dir"]
    processed_dir = root / dataset["processed_dir"]
    reports_dir = root / dataset["reports_dir"]
    results_dir = Path(dataset["results_dir"])

    output_paths = {
        "raw_parsed": interim_dir / "current_internal_raw_parsed.csv",
        "legacy_rows": interim_dir / "current_internal_legacy_rows.csv",
        "issues": interim_dir / "current_internal_normalization_issues.csv",
        "legacy_failures": interim_dir / "current_internal_legacy_failures.csv",
        "mutations": processed_dir / "current_internal_mutations.csv",
        "recalc_jobs": processed_dir / "current_internal_recalc_jobs.csv",
        "legacy_results": processed_dir / "current_internal_legacy_results.csv",
        "qc_md": reports_dir / "current_internal_normalization_qc.md",
        "qc_json": reports_dir / "current_internal_normalization_qc.json",
    }

    print(f"Dataset: {args.dataset}")
    print(f"Input CSV: {raw_rel_path.as_posix()}")
    print("Outputs:")
    for path in output_paths.values():
        print(f"  {path.relative_to(root).as_posix()}")

    if not raw_path.exists():
        raise SystemExit(f"Raw current_internal CSV is missing: {raw_rel_path.as_posix()}")

    header, raw_rows = read_raw_csv(raw_path)
    aliases = load_aliases(root / args.aliases_config if not args.aliases_config.is_absolute() else args.aliases_config)
    mapping, missing_expected_columns = detect_columns(header, aliases)
    outputs = build_outputs(
        args.dataset,
        raw_rel_path,
        raw_rows,
        header,
        mapping,
        missing_expected_columns,
        methods,
        structure_sources,
        results_dir,
    )

    print(f"Raw rows: {len(raw_rows)}")
    print(f"Detected columns: {', '.join(header)}")
    print(f"Column mapping: {json.dumps(mapping, sort_keys=True)}")
    print(f"Unique mutations: {len(outputs['mutation_rows'])}")
    print(f"Recalc jobs: {len(outputs['recalc_jobs'])}")
    print(f"Issues: {len(outputs['issues'])}")

    if args.dry_run:
        print("Dry run: no output files were written.")
        return 0

    write_csv(output_paths["raw_parsed"], outputs["parsed_rows"], ["source_row_id", "protein_id", "uniprot_id", "pdb_id", "chain_id", "mutation", "mutation_from_protein_id", "experimental_ddg", "experimental_ddg_unit", "method", "structure_source", "status", "predicted_score", "quality_flags", "raw_columns_json"], args.force)
    write_csv(output_paths["legacy_rows"], outputs["legacy_rows"], ["dataset_name", "legacy_row_id", "mutation_record_id", "protein_id", "uniprot_id", "pdb_id", "chain_id", "mutation", "experimental_ddg", "legacy_method", "legacy_structure_source", "legacy_status", "legacy_predicted_score", "legacy_unit", "legacy_error", "legacy_raw_output_path", "quality_flags", "notes"], args.force)
    write_csv(output_paths["issues"], outputs["issues"], ["source_row_id", "issue_type", "protein_id", "uniprot_id", "pdb_id", "chain_id", "mutation", "message", "raw_columns_json"], args.force)
    write_csv(output_paths["legacy_failures"], outputs["legacy_failures"], ["dataset_name", "legacy_row_id", "mutation_record_id", "protein_id", "uniprot_id", "pdb_id", "chain_id", "mutation", "experimental_ddg", "legacy_method", "legacy_structure_source", "legacy_status", "legacy_predicted_score", "legacy_unit", "legacy_error", "legacy_raw_output_path", "quality_flags", "notes"], args.force)
    write_csv(output_paths["mutations"], outputs["mutation_rows"], ["dataset_name", "mutation_record_id", "source_dataset", "source_record_ids", "protein_id", "uniprot_id", "pdb_id", "chain_id", "wild_type_aa", "position", "mutant_aa", "mutation", "experimental_ddg", "experimental_ddg_unit", "ddg_sign_convention", "has_legacy_method_outputs", "legacy_methods_available", "n_legacy_rows", "quality_flags", "notes"], args.force)
    write_csv(output_paths["recalc_jobs"], outputs["recalc_jobs"], ["dataset_name", "job_id", "mutation_record_id", "protein_id", "uniprot_id", "pdb_id", "chain_id", "mutation", "experimental_ddg", "method", "structure_source", "requires_structure", "requires_sequence", "job_status", "expected_output_dir", "notes"], args.force)
    write_csv(output_paths["legacy_results"], outputs["legacy_rows"], ["dataset_name", "legacy_row_id", "mutation_record_id", "protein_id", "uniprot_id", "pdb_id", "chain_id", "mutation", "experimental_ddg", "legacy_method", "legacy_structure_source", "legacy_status", "legacy_predicted_score", "legacy_unit", "legacy_error", "legacy_raw_output_path", "quality_flags", "notes"], args.force)
    write_qc_markdown(output_paths["qc_md"], outputs["qc"], args.force)
    if output_paths["qc_json"].exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing output without --force: {output_paths['qc_json']}")
    output_paths["qc_json"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["qc_json"].write_text(json.dumps(outputs["qc"], indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print("Normalization outputs written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
