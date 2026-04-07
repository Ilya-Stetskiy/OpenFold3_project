from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests

from openfold3.testbench.evaluation import EvaluationSummary, evaluate_reports

from .cif_utils import parse_structure_records, summarize_structure, write_pdb_atom_records
from .harness import DdgBenchmarkHarness, HarnessReport
from .methods import FoldXBuildModelMethod
from .models import BenchmarkCase, MutationInput
from .structure_source import (
    CANONICAL_AA_3_TO_1,
    ResolvedStructureSource,
    resolve_structure_source,
    validate_mutation_site,
)


@dataclass(frozen=True, slots=True)
class LocalEditResult:
    case_id: str
    mutation: MutationInput
    source_kind: str
    source_path: Path
    source_cache_hit: bool
    mutant_structure_path: Path | None
    local_edit_status: str
    prepared_from_cif: bool | None
    runtime_seconds: float
    failure_reason: str | None
    report_path: Path
    harness_report: HarnessReport
    mutant_structure_summary: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class LocalEditSuiteResult:
    output_root: Path
    results: tuple[LocalEditResult, ...]
    evaluation_summary: EvaluationSummary
    evaluation_summary_path: Path
    manifest_path: Path


def _slug_case_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "case"


def _ensure_single_mutation(mutations: Iterable[MutationInput]) -> MutationInput:
    mutations = tuple(mutations)
    if len(mutations) != 1:
        raise ValueError(
            f"Local edit v1 expects exactly one mutation, got {len(mutations)}"
        )
    mutation = mutations[0]
    if str(mutation.from_residue).upper() == str(mutation.to_residue).upper():
        raise ValueError(f"No-op mutation is unsupported in v1: {mutation.mutation_id}")
    return mutation


def _structure_summary_payload(structure_path: Path) -> dict[str, object]:
    summary = summarize_structure(structure_path)
    return {
        "atom_count": summary.atom_count,
        "residue_count": summary.residue_count,
        "chain_ids": list(summary.chain_ids),
        "chain_lengths": {
            chain_id: len(residues)
            for chain_id, residues in summary.residues_by_chain.items()
        },
        "inferred_chain_groups": [
            list(group) for group in summary.inferred_chain_groups
        ],
        "min_inter_chain_atom_distance": summary.min_inter_chain_atom_distance,
        "interface_atom_contacts_5a": summary.interface_atom_contacts_5a,
        "interface_ca_contacts_8a": summary.interface_ca_contacts_8a,
        "chain_pair_min_distances": summary.chain_pair_min_distances,
    }


def _default_harness() -> DdgBenchmarkHarness:
    return DdgBenchmarkHarness(methods=[FoldXBuildModelMethod()])


def _write_mutant_structure_with_reference_hetero_atoms(
    *,
    source_path: Path,
    mutant_model_path: Path,
    output_path: Path,
) -> None:
    mutant_atoms = [
        atom
        for atom in parse_structure_records(mutant_model_path)
        if str(atom.group_pdb).upper() != "HETATM"
        and atom.residue_name.upper() in CANONICAL_AA_3_TO_1
    ]
    reference_hetero_atoms = [
        atom
        for atom in parse_structure_records(source_path)
        if str(atom.group_pdb).upper() == "HETATM"
        or atom.residue_name.upper() not in CANONICAL_AA_3_TO_1
    ]
    write_pdb_atom_records(output_path, [*mutant_atoms, *reference_hetero_atoms])


def _build_benchmark_case(
    *,
    case_id: str,
    source: ResolvedStructureSource,
    mutation: MutationInput,
    confidence_path: str | Path | None,
    experimental_ddg: float | None,
    notes: str | None,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        structure_path=source.source_path,
        confidence_path=(
            None
            if confidence_path is None
            else Path(confidence_path).expanduser().resolve()
        ),
        mutations=(mutation,),
        experimental_ddg=experimental_ddg,
        notes=notes,
        pdb_id=source.pdb_id,
    )


def _extract_foldx_result(report: HarnessReport) -> dict[str, object] | None:
    for result in report.results:
        if result.method == "foldx":
            return {
                "status": result.status,
                "details": result.details,
            }
    return None


def _persist_result_payload(path: Path, result: LocalEditResult) -> None:
    payload = {
        **asdict(result),
        "mutation": asdict(result.mutation),
        "source_path": str(result.source_path),
        "mutant_structure_path": (
            None
            if result.mutant_structure_path is None
            else str(result.mutant_structure_path)
        ),
        "report_path": str(result.report_path),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_local_mutation_case(
    *,
    mutation: MutationInput,
    work_dir: str | Path,
    structure_path: str | Path | None = None,
    pdb_id: str | None = None,
    case_id: str | None = None,
    confidence_path: str | Path | None = None,
    experimental_ddg: float | None = None,
    notes: str | None = None,
    cache_dir: str | Path | None = None,
    session: requests.Session | None = None,
    harness: DdgBenchmarkHarness | None = None,
) -> LocalEditResult:
    started_at = time.perf_counter()
    mutation = _ensure_single_mutation((mutation,))
    source = resolve_structure_source(
        structure_path=structure_path,
        pdb_id=pdb_id,
        cache_dir=cache_dir,
        session=session,
    )
    validate_mutation_site(source.source_path, mutation)

    resolved_case_id = case_id or f"{source.pdb_id or source.source_path.stem}_{mutation.mutation_id}"
    case_root = Path(work_dir).expanduser().resolve() / _slug_case_id(resolved_case_id)
    case_root.mkdir(parents=True, exist_ok=True)
    report_path = case_root / "local_edit_report.json"
    result_path = case_root / "local_edit_result.json"

    benchmark_case = _build_benchmark_case(
        case_id=resolved_case_id,
        source=source,
        mutation=mutation,
        confidence_path=confidence_path,
        experimental_ddg=experimental_ddg,
        notes=notes,
    )
    resolved_harness = harness or _default_harness()
    report = resolved_harness.run_case(benchmark_case)
    DdgBenchmarkHarness.write_report(report, report_path)

    foldx_result = _extract_foldx_result(report)
    local_edit_status = "failed"
    prepared_from_cif = None
    failure_reason = "foldx_method_missing"
    mutant_structure_path: Path | None = None
    mutant_structure_summary: dict[str, object] | None = None

    if foldx_result is not None:
        local_edit_status = str(foldx_result["status"])
        details = (
            foldx_result["details"]
            if isinstance(foldx_result["details"], dict)
            else {}
        )
        prepared_from_cif_raw = details.get("prepared_from_cif")
        if prepared_from_cif_raw is not None:
            prepared_from_cif = bool(prepared_from_cif_raw)
        failure_reason = (
            None if local_edit_status == "ok" else str(details.get("reason") or local_edit_status)
        )
        mutant_model_path = details.get("mutant_model_path")
        if local_edit_status == "ok" and mutant_model_path is not None:
            candidate_path = Path(str(mutant_model_path)).expanduser().resolve()
            if not candidate_path.exists():
                local_edit_status = "failed"
                failure_reason = f"mutant_model_missing:{candidate_path}"
            else:
                mutant_structure_path = case_root / candidate_path.name
                _write_mutant_structure_with_reference_hetero_atoms(
                    source_path=source.source_path,
                    mutant_model_path=candidate_path,
                    output_path=mutant_structure_path,
                )
                mutant_structure_summary = _structure_summary_payload(
                    mutant_structure_path
                )
                failure_reason = None

    runtime_seconds = time.perf_counter() - started_at
    result = LocalEditResult(
        case_id=resolved_case_id,
        mutation=mutation,
        source_kind=source.source_kind,
        source_path=source.source_path,
        source_cache_hit=source.cache_hit,
        mutant_structure_path=mutant_structure_path,
        local_edit_status=local_edit_status,
        prepared_from_cif=prepared_from_cif,
        runtime_seconds=runtime_seconds,
        failure_reason=failure_reason,
        report_path=report_path,
        harness_report=report,
        mutant_structure_summary=mutant_structure_summary,
    )
    _persist_result_payload(result_path, result)
    return result


def run_local_mutation_suite(
    *,
    output_root: str | Path,
    cases: Iterable[BenchmarkCase] | None = None,
    cases_json: str | Path | None = None,
    dataset_kind: str = "exploratory",
    cache_dir: str | Path | None = None,
    session: requests.Session | None = None,
    harness: DdgBenchmarkHarness | None = None,
) -> LocalEditSuiteResult:
    if (cases is None) == (cases_json is None):
        raise ValueError("Provide exactly one of cases or cases_json")

    if cases_json is not None:
        from openfold3.testbench.runner import load_cases_from_json

        cases = load_cases_from_json(
            Path(cases_json),
            structure_cache_dir=cache_dir,
            session=session,
        )
    assert cases is not None
    resolved_cases = tuple(cases)

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases_root = output_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)

    results: list[LocalEditResult] = []
    reports: list[HarnessReport] = []
    for case in resolved_cases:
        mutation = _ensure_single_mutation(case.mutations)
        result = run_local_mutation_case(
            mutation=mutation,
            work_dir=cases_root,
            structure_path=None if case.pdb_id is not None else case.structure_path,
            pdb_id=case.pdb_id,
            case_id=case.case_id,
            confidence_path=case.confidence_path,
            experimental_ddg=case.experimental_ddg,
            notes=case.notes,
            cache_dir=cache_dir,
            session=session,
            harness=harness,
        )
        results.append(result)
        reports.append(result.harness_report)

    summary = evaluate_reports(reports)
    evaluation_summary_path = output_root / "evaluation_summary.json"
    evaluation_summary_path.write_text(summary.to_json(), encoding="utf-8")

    manifest_path = output_root / "local_edit_manifest.json"
    manifest = {
        "output_root": str(output_root),
        "dataset_kind": dataset_kind,
        "num_cases": len(results),
        "successful_edits": sum(result.local_edit_status == "ok" for result in results),
        "report_paths": [str(result.report_path) for result in results],
        "evaluation_summary_path": str(evaluation_summary_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return LocalEditSuiteResult(
        output_root=output_root,
        results=tuple(results),
        evaluation_summary=summary,
        evaluation_summary_path=evaluation_summary_path,
        manifest_path=manifest_path,
    )
