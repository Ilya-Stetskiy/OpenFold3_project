#!/usr/bin/env python3
"""Build future evaluation task definitions for current_internal."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("PyYAML is required: install pyyaml or use the project environment.") from exc


DEFAULT_DATASETS_CONFIG = Path("configs/datasets.yaml")
DEFAULT_METRICS_CONFIG = Path("configs/evaluation_metrics.yaml")
TASK_FIELDS = [
    "dataset_name",
    "evaluation_task_id",
    "metric_name",
    "metric_group",
    "metric_type",
    "mutation_record_id",
    "protein_id",
    "uniprot_id",
    "pdb_id",
    "chain_id",
    "mutation",
    "wild_type_aa",
    "position",
    "mutant_aa",
    "method",
    "structure_source",
    "reference_structure_path",
    "predicted_structure_path",
    "reference_score_path",
    "predicted_score_path",
    "experimental_ddg",
    "local_radius_angstrom",
    "requires_reference_structure",
    "requires_predicted_structure",
    "requires_mutation_site",
    "requires_plddt",
    "input_ready",
    "blocking_reason",
    "expected_output_path",
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


def bool_text(value: bool) -> str:
    return str(bool(value)).lower()


def bool_from_text(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def rel_exists(root: Path, path_value: str) -> bool:
    if not path_value:
        return False
    path = Path(path_value)
    return (path if path.is_absolute() else root / path).exists()


def openfold_confidence_blocking_reason(root: Path, predicted_structure_path: str, confidence_path: str) -> str:
    if not rel_exists(root, predicted_structure_path):
        return "pending_openfold_structure"
    if not rel_exists(root, confidence_path):
        return "pending_confidence_file"
    return ""


def metric_entries(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    groups = config.get("metric_groups", {})
    if not isinstance(groups, dict):
        raise SystemExit("configs/evaluation_metrics.yaml must define metric_groups")
    for group_name, group in groups.items():
        for metric in group.get("metrics", []):
            metric = dict(metric)
            metric.setdefault("metric_group", group_name)
            entries[metric["metric_name"]] = metric
    return entries


def selected_metric_names(
    metrics_config: dict[str, Any],
    requested_groups: list[str] | None,
    requested_names: list[str] | None,
    include_future_blocked: bool,
) -> set[str]:
    selected: set[str] = set()
    groups = metrics_config.get("metric_groups", {})
    for group_name, group in groups.items():
        if requested_groups and group_name not in requested_groups:
            continue
        if not include_future_blocked and not group.get("active_for_current_internal", False):
            continue
        for metric in group.get("metrics", []):
            name = metric["metric_name"]
            if requested_names and name not in requested_names:
                continue
            selected.add(name)
    unknown_groups = sorted(set(requested_groups or []) - set(groups))
    if unknown_groups:
        raise SystemExit(f"Unknown metric group(s): {', '.join(unknown_groups)}")
    unknown_names = sorted(set(requested_names or []) - set(metric_entries(metrics_config)))
    if unknown_names:
        raise SystemExit(f"Unknown metric name(s): {', '.join(unknown_names)}")
    return selected


def sequence_id_for(row: dict[str, str]) -> str:
    return f"{row.get('dataset_name', 'current_internal')}__{row.get('uniprot_id', '')}__{row.get('pdb_id', '').upper()}_{row.get('chain_id', '')}"


def openfold_structure_path(dataset_name: str, sequence_id: str) -> str:
    return f"results/{dataset_name}/openfold/wt/{sequence_id}/model.cif"


def openfold_confidence_path(dataset_name: str, sequence_id: str) -> str:
    return f"results/{dataset_name}/openfold/wt/{sequence_id}/confidence.json"


def parsed_result_path(dataset_name: str, method: str, structure_source: str, mutation_record_id: str) -> str:
    return f"results/{dataset_name}/{method}/{structure_source}/{mutation_record_id}/parsed_result.json"


def base_task(
    dataset_name: str,
    metric: dict[str, Any],
    mutation: dict[str, str] | None = None,
    method: str = "",
    structure_source: str = "",
) -> dict[str, Any]:
    mutation = mutation or {}
    return {
        "dataset_name": dataset_name,
        "evaluation_task_id": "",
        "metric_name": metric["metric_name"],
        "metric_group": metric["metric_group"],
        "metric_type": metric["metric_type"],
        "mutation_record_id": mutation.get("mutation_record_id", ""),
        "protein_id": mutation.get("protein_id", ""),
        "uniprot_id": mutation.get("uniprot_id", ""),
        "pdb_id": mutation.get("pdb_id", ""),
        "chain_id": mutation.get("chain_id", ""),
        "mutation": mutation.get("mutation", ""),
        "wild_type_aa": mutation.get("wild_type_aa", ""),
        "position": mutation.get("position", ""),
        "mutant_aa": mutation.get("mutant_aa", ""),
        "method": method,
        "structure_source": structure_source,
        "reference_structure_path": "",
        "predicted_structure_path": "",
        "reference_score_path": "",
        "predicted_score_path": "",
        "experimental_ddg": mutation.get("experimental_ddg", ""),
        "local_radius_angstrom": "",
        "requires_reference_structure": bool_text(metric.get("requires_reference_structure", False)),
        "requires_predicted_structure": bool_text(metric.get("requires_predicted_structure", False)),
        "requires_mutation_site": bool_text(metric.get("requires_mutation_site", False)),
        "requires_plddt": bool_text(metric.get("requires_plddt", False)),
        "input_ready": "false",
        "blocking_reason": "",
        "expected_output_path": "",
        "notes": "",
    }


def add_task(tasks: list[dict[str, Any]], task: dict[str, Any]) -> None:
    tasks.append(task)


def structure_by_mutation(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["mutation_record_id"]: row for row in rows}


def mutations_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["mutation_record_id"]: row for row in rows}


def enrich_jobs_with_mutations(jobs: list[dict[str, str]], mutations_lookup: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []
    for job in jobs:
        merged = dict(mutations_lookup.get(job.get("mutation_record_id", ""), {}))
        merged.update(job)
        enriched.append(merged)
    return enriched


def sequence_by_chain(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row.get("pdb_id", "").upper(), row.get("chain_id", "")): row for row in rows}


def create_ddg_tasks(
    root: Path,
    dataset_name: str,
    metrics: dict[str, dict[str, Any]],
    selected: set[str],
    jobs: list[dict[str, str]],
    tasks: list[dict[str, Any]],
) -> None:
    if "ddg_regression" not in selected:
        return
    metric = metrics["ddg_regression"]
    for job in jobs:
        method = job.get("method", "")
        source = job.get("structure_source", "")
        predicted_path = parsed_result_path(dataset_name, method, source, job["mutation_record_id"])
        task = base_task(dataset_name, metric, job, method, source)
        task["evaluation_task_id"] = f"{dataset_name}__ddg_regression__{method}__{source}__{job['mutation_record_id']}"
        task["predicted_score_path"] = predicted_path
        task["input_ready"] = bool_text(bool(job.get("experimental_ddg")) and rel_exists(root, predicted_path))
        task["blocking_reason"] = "" if bool_from_text(task["input_ready"]) else "pending_method_result"
        if not job.get("experimental_ddg"):
            task["blocking_reason"] = "missing_experimental_ddg"
        task["expected_output_path"] = (
            f"results/evaluation/{dataset_name}/scalar_ddg/ddg_regression/{method}/{source}/{job['mutation_record_id']}.json"
        )
        task["notes"] = "Future scalar DDG evaluation task; no metrics are computed by the builder."
        add_task(tasks, task)


def create_paired_tasks(
    root: Path,
    dataset_name: str,
    metrics: dict[str, dict[str, Any]],
    selected: set[str],
    mutations: list[dict[str, str]],
    tasks: list[dict[str, Any]],
) -> None:
    if "paired_score_comparison" not in selected:
        return
    metric = metrics["paired_score_comparison"]
    for method in ("foldx", "rosetta"):
        for mutation in mutations:
            mutation_id = mutation["mutation_record_id"]
            ref_path = parsed_result_path(dataset_name, method, "experimental", mutation_id)
            pred_path = parsed_result_path(dataset_name, method, "openfold", mutation_id)
            ready = rel_exists(root, ref_path) and rel_exists(root, pred_path)
            task = base_task(dataset_name, metric, mutation, method, "experimental_vs_openfold")
            task["evaluation_task_id"] = f"{dataset_name}__paired_score_comparison__{method}__experimental_vs_openfold__{mutation_id}"
            task["reference_score_path"] = ref_path
            task["predicted_score_path"] = pred_path
            task["input_ready"] = bool_text(ready)
            task["blocking_reason"] = "" if ready else "pending_paired_method_results"
            task["expected_output_path"] = f"results/evaluation/{dataset_name}/scalar_ddg/paired_score_comparison/{method}/{mutation_id}.json"
            task["notes"] = "Future paired score comparison between experimental-structure and OpenFold-structure method outputs."
            add_task(tasks, task)


def unique_chain_rows(
    structure_rows: list[dict[str, str]],
    sequence_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    seq_by_chain = sequence_by_chain(sequence_rows)
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for row in structure_rows:
        key = (row.get("pdb_id", "").upper(), row.get("chain_id", ""))
        if key in seen:
            continue
        seen.add(key)
        merged = dict(row)
        seq_row = seq_by_chain.get(key, {})
        merged["sequence_id"] = seq_row.get("sequence_id", sequence_id_for(row))
        merged["protein_id"] = seq_row.get("protein_id", row.get("protein_id", ""))
        merged["mutation_record_id"] = ""
        merged["mutation"] = ""
        merged["wild_type_aa"] = ""
        merged["position"] = ""
        merged["mutant_aa"] = ""
        merged["experimental_ddg"] = ""
        unique.append(merged)
    return unique


def create_global_structure_tasks(
    root: Path,
    dataset_name: str,
    metrics: dict[str, dict[str, Any]],
    selected: set[str],
    chain_rows: list[dict[str, str]],
    tasks: list[dict[str, Any]],
) -> None:
    for metric_name in ("ca_rmsd", "backbone_rmsd", "all_atom_rmsd", "tm_score", "gdt_ts"):
        if metric_name not in selected:
            continue
        metric = metrics[metric_name]
        for chain in chain_rows:
            sequence_id = chain["sequence_id"]
            reference_path = chain.get("resolved_chain_structure_path") or chain.get("original_chain_structure_path", "")
            predicted_path = openfold_structure_path(dataset_name, sequence_id)
            ready = rel_exists(root, reference_path) and rel_exists(root, predicted_path)
            task = base_task(dataset_name, metric, chain)
            task["evaluation_task_id"] = f"{dataset_name}__{metric_name}__{sequence_id}"
            task["reference_structure_path"] = reference_path
            task["predicted_structure_path"] = predicted_path
            task["input_ready"] = bool_text(ready)
            task["blocking_reason"] = "" if ready else ("missing_reference_structure" if not rel_exists(root, reference_path) else "pending_openfold_structure")
            task["expected_output_path"] = f"results/evaluation/{dataset_name}/global_structure/{metric_name}/{sequence_id}.json"
            task["notes"] = "Global structure metric task is per unique PDB chain/sequence."
            add_task(tasks, task)


def create_local_structure_tasks(
    root: Path,
    dataset_name: str,
    metrics_config: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    selected: set[str],
    mutations: list[dict[str, str]],
    structure_by_id: dict[str, dict[str, str]],
    sequence_rows: list[dict[str, str]],
    tasks: list[dict[str, Any]],
) -> None:
    radius = str(metrics_config.get("defaults", {}).get("local_radius_angstrom", 5.0))
    seq_by_chain = sequence_by_chain(sequence_rows)
    for metric_name in ("local_ca_rmsd", "local_backbone_rmsd", "local_lddt", "local_contact_precision"):
        if metric_name not in selected:
            continue
        metric = metrics[metric_name]
        for mutation in mutations:
            structure = structure_by_id.get(mutation["mutation_record_id"], {})
            key = (mutation.get("pdb_id", "").upper(), mutation.get("chain_id", ""))
            sequence_id = seq_by_chain.get(key, {}).get("sequence_id", sequence_id_for(mutation))
            reference_path = structure.get("resolved_chain_structure_path") or structure.get("original_chain_structure_path", "")
            predicted_path = openfold_structure_path(dataset_name, sequence_id)
            ready = bool(mutation.get("position")) and rel_exists(root, reference_path) and rel_exists(root, predicted_path)
            task = base_task(dataset_name, metric, mutation)
            task["evaluation_task_id"] = f"{dataset_name}__{metric_name}__r{radius}__{mutation['mutation_record_id']}"
            task["reference_structure_path"] = reference_path
            task["predicted_structure_path"] = predicted_path
            task["local_radius_angstrom"] = radius
            task["input_ready"] = bool_text(ready)
            if ready:
                task["blocking_reason"] = ""
            elif not mutation.get("position"):
                task["blocking_reason"] = "missing_mutation_site"
            elif not rel_exists(root, reference_path):
                task["blocking_reason"] = "missing_reference_structure"
            else:
                task["blocking_reason"] = "pending_openfold_structure"
            task["expected_output_path"] = f"results/evaluation/{dataset_name}/local_mutation_structure/{metric_name}/{mutation['mutation_record_id']}.json"
            task["notes"] = "Local mutation-site structure task uses the default 5 Angstrom radius."
            add_task(tasks, task)


def create_confidence_tasks(
    root: Path,
    dataset_name: str,
    metrics_config: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    selected: set[str],
    mutations: list[dict[str, str]],
    jobs: list[dict[str, str]],
    chain_rows: list[dict[str, str]],
    sequence_rows: list[dict[str, str]],
    tasks: list[dict[str, Any]],
) -> None:
    radius = str(metrics_config.get("defaults", {}).get("local_radius_angstrom", 5.0))
    seq_by_chain = sequence_by_chain(sequence_rows)

    if "global_plddt" in selected:
        metric = metrics["global_plddt"]
        for chain in chain_rows:
            sequence_id = chain["sequence_id"]
            predicted_path = openfold_structure_path(dataset_name, sequence_id)
            confidence_path = openfold_confidence_path(dataset_name, sequence_id)
            blocking_reason = openfold_confidence_blocking_reason(root, predicted_path, confidence_path)
            ready = not blocking_reason
            task = base_task(dataset_name, metric, chain)
            task["evaluation_task_id"] = f"{dataset_name}__global_plddt__{sequence_id}"
            task["predicted_structure_path"] = predicted_path
            task["predicted_score_path"] = confidence_path
            task["input_ready"] = bool_text(ready)
            task["blocking_reason"] = blocking_reason
            task["expected_output_path"] = f"results/evaluation/{dataset_name}/confidence_error/global_plddt/{sequence_id}.json"
            task["notes"] = "Future pLDDT summary for OpenFold WT prediction."
            add_task(tasks, task)

    for metric_name in ("mutation_site_plddt", "local_plddt"):
        if metric_name not in selected:
            continue
        metric = metrics[metric_name]
        for mutation in mutations:
            key = (mutation.get("pdb_id", "").upper(), mutation.get("chain_id", ""))
            sequence_id = seq_by_chain.get(key, {}).get("sequence_id", sequence_id_for(mutation))
            predicted_path = openfold_structure_path(dataset_name, sequence_id)
            confidence_path = openfold_confidence_path(dataset_name, sequence_id)
            blocking_reason = openfold_confidence_blocking_reason(root, predicted_path, confidence_path)
            if not mutation.get("position"):
                blocking_reason = "missing_mutation_site"
            ready = not blocking_reason
            task = base_task(dataset_name, metric, mutation)
            task["evaluation_task_id"] = f"{dataset_name}__{metric_name}__{mutation['mutation_record_id']}"
            task["predicted_structure_path"] = predicted_path
            task["predicted_score_path"] = confidence_path
            task["local_radius_angstrom"] = radius if metric_name == "local_plddt" else ""
            task["input_ready"] = bool_text(ready)
            task["blocking_reason"] = blocking_reason
            task["expected_output_path"] = f"results/evaluation/{dataset_name}/confidence_error/{metric_name}/{mutation['mutation_record_id']}.json"
            task["notes"] = "Future mutation-site confidence summary."
            add_task(tasks, task)

    if "plddt_vs_global_error" in selected:
        metric = metrics["plddt_vs_global_error"]
        for chain in chain_rows:
            sequence_id = chain["sequence_id"]
            task = base_task(dataset_name, metric, chain)
            task["evaluation_task_id"] = f"{dataset_name}__plddt_vs_global_error__{sequence_id}"
            task["predicted_structure_path"] = openfold_structure_path(dataset_name, sequence_id)
            task["predicted_score_path"] = openfold_confidence_path(dataset_name, sequence_id)
            task["blocking_reason"] = "pending_upstream_metrics"
            task["expected_output_path"] = f"results/evaluation/{dataset_name}/confidence_error/plddt_vs_global_error/{sequence_id}.json"
            task["notes"] = "Relation task requires global pLDDT and global structure metrics first."
            add_task(tasks, task)

    if "plddt_vs_local_error" in selected:
        metric = metrics["plddt_vs_local_error"]
        for mutation in mutations:
            key = (mutation.get("pdb_id", "").upper(), mutation.get("chain_id", ""))
            sequence_id = seq_by_chain.get(key, {}).get("sequence_id", sequence_id_for(mutation))
            task = base_task(dataset_name, metric, mutation)
            task["evaluation_task_id"] = f"{dataset_name}__plddt_vs_local_error__{mutation['mutation_record_id']}"
            task["predicted_structure_path"] = openfold_structure_path(dataset_name, sequence_id)
            task["predicted_score_path"] = openfold_confidence_path(dataset_name, sequence_id)
            task["local_radius_angstrom"] = radius
            task["blocking_reason"] = "pending_upstream_metrics"
            task["expected_output_path"] = f"results/evaluation/{dataset_name}/confidence_error/plddt_vs_local_error/{mutation['mutation_record_id']}.json"
            task["notes"] = "Relation task requires local pLDDT and local structure metrics first."
            add_task(tasks, task)

    if "plddt_vs_ddg_error" in selected:
        metric = metrics["plddt_vs_ddg_error"]
        for job in jobs:
            key = (job.get("pdb_id", "").upper(), job.get("chain_id", ""))
            sequence_id = seq_by_chain.get(key, {}).get("sequence_id", sequence_id_for(job))
            task = base_task(dataset_name, metric, job, job.get("method", ""), job.get("structure_source", ""))
            task["evaluation_task_id"] = (
                f"{dataset_name}__plddt_vs_ddg_error__{job.get('method', '')}__{job.get('structure_source', '')}__{job['mutation_record_id']}"
            )
            task["predicted_structure_path"] = openfold_structure_path(dataset_name, sequence_id)
            task["predicted_score_path"] = openfold_confidence_path(dataset_name, sequence_id)
            task["blocking_reason"] = "pending_upstream_metrics"
            task["expected_output_path"] = (
                f"results/evaluation/{dataset_name}/confidence_error/plddt_vs_ddg_error/{job.get('method', '')}/{job.get('structure_source', '')}/{job['mutation_record_id']}.json"
            )
            task["notes"] = "Relation task requires pLDDT summaries and DDG error metrics first."
            add_task(tasks, task)


def create_future_complex_tasks(
    dataset_name: str,
    metrics: dict[str, dict[str, Any]],
    selected: set[str],
    tasks: list[dict[str, Any]],
) -> None:
    for metric_name in ("dockq", "interface_rmsd", "interface_contact_precision"):
        if metric_name not in selected:
            continue
        metric = metrics[metric_name]
        task = base_task(dataset_name, metric)
        task["evaluation_task_id"] = f"{dataset_name}__future_blocked__{metric_name}"
        task["input_ready"] = "false"
        task["blocking_reason"] = "dataset_not_complex_interface"
        task["expected_output_path"] = f"results/evaluation/{dataset_name}/future_complex_interface/{metric_name}/blocked_placeholder.json"
        task["notes"] = "Future complex/interface extension; not active for current_internal single-chain dataset."
        add_task(tasks, task)


def build_qc(
    dataset_name: str,
    paths: dict[str, str],
    tasks: list[dict[str, Any]],
    chain_rows: list[dict[str, str]],
) -> dict[str, Any]:
    by_group = Counter(row["metric_group"] for row in tasks)
    by_name = Counter(row["metric_name"] for row in tasks)
    by_type = Counter(row["metric_type"] for row in tasks)
    blocking = Counter(row["blocking_reason"] or "none" for row in tasks if not bool_from_text(row.get("input_ready", "")))
    structure_tasks = [row for row in tasks if row["metric_group"] in {"global_structure", "local_mutation_structure"}]
    scalar_tasks = [row for row in tasks if row["metric_group"] == "scalar_ddg"]
    confidence_tasks = [row for row in tasks if row["metric_group"] == "confidence_error"]
    return {
        "dataset_name": dataset_name,
        "input_summary": paths,
        "task_summary": {
            "total_evaluation_tasks": len(tasks),
            "tasks_by_metric_group": dict(by_group),
            "tasks_by_metric_name": dict(by_name),
            "tasks_by_metric_type": dict(by_type),
        },
        "readiness": {
            "input_ready_count": sum(1 for row in tasks if bool_from_text(row.get("input_ready", ""))),
            "blocked_count": sum(1 for row in tasks if not bool_from_text(row.get("input_ready", ""))),
            "blocking_reasons": dict(blocking),
        },
        "structure_tasks": {
            "unique_global_structure_comparison_tasks": by_group.get("global_structure", 0),
            "local_mutation_site_tasks": by_group.get("local_mutation_structure", 0),
            "reference_structures_available": sum(1 for row in chain_rows if row.get("resolved_chain_structure_path")),
            "predicted_structures_pending": sum(1 for row in structure_tasks if row.get("blocking_reason") == "pending_openfold_structure"),
        },
        "scalar_tasks": {
            "ddg_regression_tasks": by_name.get("ddg_regression", 0),
            "paired_score_comparison_tasks": by_name.get("paired_score_comparison", 0),
            "pending_method_outputs": sum(1 for row in scalar_tasks if row.get("blocking_reason") in {"pending_method_result", "pending_paired_method_results"}),
        },
        "confidence_tasks": {
            "plddt_tasks": by_name.get("global_plddt", 0) + by_name.get("mutation_site_plddt", 0) + by_name.get("local_plddt", 0),
            "relation_tasks": by_name.get("plddt_vs_global_error", 0) + by_name.get("plddt_vs_local_error", 0) + by_name.get("plddt_vs_ddg_error", 0),
            "pending_upstream_dependencies": sum(1 for row in confidence_tasks if row.get("blocking_reason") == "pending_upstream_metrics"),
        },
        "warnings": [
            "RMSD/TM-score/lDDT/contact-precision tasks are future task definitions until OpenFold structures exist.",
            "Global metrics are per unique chain/sequence; local metrics are per mutation.",
            "pLDDT-vs-error tasks require upstream structural and DDG metrics.",
            "Complex/interface metrics are not active for current_internal single-chain dataset.",
        ],
    }


def markdown_report(qc: dict[str, Any]) -> str:
    lines = [
        "# Current Internal Evaluation Tasks QC",
        "",
        "This report describes task definitions only. No RMSD, TM-score, lDDT, contact precision, pLDDT relation, or DDG metric is computed here.",
        "",
        "## Input Summary",
        "",
    ]
    for label, path in qc["input_summary"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Task Summary",
            "",
            f"- Total evaluation tasks: {qc['task_summary']['total_evaluation_tasks']}",
            f"- Tasks by metric_group: `{json.dumps(qc['task_summary']['tasks_by_metric_group'], sort_keys=True)}`",
            f"- Tasks by metric_name: `{json.dumps(qc['task_summary']['tasks_by_metric_name'], sort_keys=True)}`",
            f"- Tasks by metric_type: `{json.dumps(qc['task_summary']['tasks_by_metric_type'], sort_keys=True)}`",
            "",
            "## Readiness",
            "",
            f"- Input-ready tasks: {qc['readiness']['input_ready_count']}",
            f"- Blocked tasks: {qc['readiness']['blocked_count']}",
            f"- Blocking reasons: `{json.dumps(qc['readiness']['blocking_reasons'], sort_keys=True)}`",
            "",
            "## Structure Tasks",
            "",
            f"- Unique global structure comparison tasks: {qc['structure_tasks']['unique_global_structure_comparison_tasks']}",
            f"- Local mutation-site tasks: {qc['structure_tasks']['local_mutation_site_tasks']}",
            f"- Reference structures available: {qc['structure_tasks']['reference_structures_available']}",
            f"- Predicted structures pending: {qc['structure_tasks']['predicted_structures_pending']}",
            "",
            "## Scalar Tasks",
            "",
            f"- ddg_regression tasks: {qc['scalar_tasks']['ddg_regression_tasks']}",
            f"- paired_score_comparison tasks: {qc['scalar_tasks']['paired_score_comparison_tasks']}",
            f"- Pending method outputs: {qc['scalar_tasks']['pending_method_outputs']}",
            "",
            "## Confidence Tasks",
            "",
            f"- pLDDT tasks: {qc['confidence_tasks']['plddt_tasks']}",
            f"- Relation tasks: {qc['confidence_tasks']['relation_tasks']}",
            f"- Pending upstream dependencies: {qc['confidence_tasks']['pending_upstream_dependencies']}",
            "",
            "## Warnings",
            "",
        ]
    )
    for warning in qc["warnings"]:
        lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--datasets-config", type=Path, default=DEFAULT_DATASETS_CONFIG)
    parser.add_argument("--metrics-config", type=Path, default=DEFAULT_METRICS_CONFIG)
    parser.add_argument("--metric-group", action="append", dest="metric_groups")
    parser.add_argument("--metric-name", action="append", dest="metric_names")
    parser.add_argument("--include-future-blocked", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.dataset != "current_internal":
        raise SystemExit("This evaluation task builder currently supports only --dataset current_internal")

    root = project_root_from(args.datasets_config)
    datasets_config = load_yaml(args.datasets_config)
    metrics_config = load_yaml(args.metrics_config)
    dataset = dataset_config(datasets_config, args.dataset)
    processed_dir = root / dataset["processed_dir"]
    reports_dir = root / dataset["reports_dir"]
    manifests_dir = root / "data" / "manifests"

    mutations_path = processed_dir / "current_internal_mutations.csv"
    jobs_path = processed_dir / "current_internal_recalc_jobs_resolved.csv"
    structure_path = manifests_dir / "current_internal_structure_manifest_resolved.csv"
    sequence_path = manifests_dir / "current_internal_sequence_manifest.csv"
    metrics_path = root / args.metrics_config
    for path in (mutations_path, jobs_path, structure_path, sequence_path, metrics_path):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path.relative_to(root).as_posix()}")

    metrics = metric_entries(metrics_config)
    selected = selected_metric_names(metrics_config, args.metric_groups, args.metric_names, args.include_future_blocked)
    mutations = read_csv(mutations_path)
    jobs = read_csv(jobs_path)
    structure_rows = read_csv(structure_path)
    sequence_rows = read_csv(sequence_path)
    mutation_lookup = mutations_by_id(mutations)
    enriched_jobs = enrich_jobs_with_mutations(jobs, mutation_lookup)
    structure_lookup = structure_by_mutation(structure_rows)
    chain_rows = unique_chain_rows(structure_rows, sequence_rows)

    tasks: list[dict[str, Any]] = []
    create_ddg_tasks(root, args.dataset, metrics, selected, enriched_jobs, tasks)
    create_paired_tasks(root, args.dataset, metrics, selected, mutations, tasks)
    create_global_structure_tasks(root, args.dataset, metrics, selected, chain_rows, tasks)
    create_local_structure_tasks(root, args.dataset, metrics_config, metrics, selected, mutations, structure_lookup, sequence_rows, tasks)
    create_confidence_tasks(root, args.dataset, metrics_config, metrics, selected, mutations, enriched_jobs, chain_rows, sequence_rows, tasks)
    if args.include_future_blocked:
        create_future_complex_tasks(args.dataset, metrics, selected, tasks)

    paths_for_qc = {
        "mutations_path": mutations_path.relative_to(root).as_posix(),
        "recalc_jobs_resolved_path": jobs_path.relative_to(root).as_posix(),
        "structure_manifest_resolved_path": structure_path.relative_to(root).as_posix(),
        "sequence_manifest_path": sequence_path.relative_to(root).as_posix(),
        "metrics_registry_path": args.metrics_config.as_posix(),
    }
    qc = build_qc(args.dataset, paths_for_qc, tasks, chain_rows)

    output_tasks_path = processed_dir / "current_internal_evaluation_tasks.csv"
    report_md_path = reports_dir / "current_internal_evaluation_tasks_qc.md"
    report_json_path = reports_dir / "current_internal_evaluation_tasks_qc.json"

    print(f"Dataset: {args.dataset}")
    print(f"Selected metrics: {len(selected)}")
    print(f"Evaluation tasks: {len(tasks)}")
    print(f"Tasks by metric_group: {json.dumps(qc['task_summary']['tasks_by_metric_group'], sort_keys=True)}")
    print(f"Input-ready tasks: {qc['readiness']['input_ready_count']}")
    print(f"Blocking reasons: {json.dumps(qc['readiness']['blocking_reasons'], sort_keys=True)}")
    if args.dry_run:
        print("Dry run: evaluation task outputs were not written.")
        return 0

    write_csv(output_tasks_path, tasks, TASK_FIELDS, args.force)
    write_text(report_json_path, json.dumps(qc, indent=2, ensure_ascii=False, sort_keys=True) + "\n", args.force)
    write_text(report_md_path, markdown_report(qc), args.force)
    print(f"Evaluation tasks written: {output_tasks_path.relative_to(root).as_posix()}")
    print(f"QC report written: {report_md_path.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
