from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import requests

from .cif_utils import parse_structure_records, summarize_structure
from .harness import DdgBenchmarkHarness
from .local_edit import LocalEditResult, run_local_mutation_case
from .models import MutationInput
from .structure_source import (
    CANONICAL_AA_3_TO_1,
    extract_protein_sequence,
    resolve_structure_source,
)


@dataclass(frozen=True, slots=True)
class ReferenceMutationCase:
    case_id: str
    mutation: MutationInput
    source_pdb_id: str | None = None
    source_structure_path: Path | None = None
    reference_pdb_id: str | None = None
    reference_structure_path: Path | None = None
    focus_chain_ids: tuple[str, ...] = ()
    category: str = "reference_mutant"
    description: str | None = None


@dataclass(frozen=True, slots=True)
class CyclicMutationCase:
    case_id: str
    steps: tuple[MutationInput, ...]
    source_pdb_id: str | None = None
    source_structure_path: Path | None = None
    focus_chain_ids: tuple[str, ...] = ()
    category: str = "cyclic_round_trip"
    description: str | None = None


@dataclass(frozen=True, slots=True)
class LocalEditBenchmarkCaseResult:
    case_id: str
    benchmark_kind: str
    category: str
    row: dict[str, object]
    step_results: tuple[LocalEditResult, ...]


@dataclass(frozen=True, slots=True)
class LocalEditBenchmarkSuiteResult:
    output_root: Path
    results: tuple[LocalEditBenchmarkCaseResult, ...]
    rows_csv_path: Path
    summary_json_path: Path


def _resolve_structure_path(
    *,
    structure_path: Path | None,
    pdb_id: str | None,
    cache_dir: str | Path | None,
    session: requests.Session | None,
) -> Path:
    resolved = resolve_structure_source(
        structure_path=structure_path,
        pdb_id=pdb_id,
        cache_dir=cache_dir,
        session=session,
    )
    return resolved.source_path


def _protein_atom_index(
    structure_path: Path,
    *,
    atom_names: set[str],
    focus_chain_ids: tuple[str, ...] = (),
) -> dict[tuple[str, str, str], np.ndarray]:
    focus = set(focus_chain_ids)
    index: dict[tuple[str, str, str], np.ndarray] = {}
    for atom in parse_structure_records(structure_path):
        if focus and atom.chain_id not in focus:
            continue
        if atom.atom_name not in atom_names:
            continue
        if atom.residue_name.upper() not in CANONICAL_AA_3_TO_1:
            continue
        index[(atom.chain_id, atom.residue_id, atom.atom_name)] = np.array(
            [atom.x, atom.y, atom.z],
            dtype=float,
        )
    return index


def _rmsd_after_superposition(
    left_path: Path,
    right_path: Path,
    *,
    atom_names: set[str],
    focus_chain_ids: tuple[str, ...] = (),
) -> float:
    left_index = _protein_atom_index(
        left_path,
        atom_names=atom_names,
        focus_chain_ids=focus_chain_ids,
    )
    right_index = _protein_atom_index(
        right_path,
        atom_names=atom_names,
        focus_chain_ids=focus_chain_ids,
    )
    common_keys = sorted(set(left_index) & set(right_index))
    if not common_keys:
        raise ValueError(
            f"No matched atoms found for RMSD between {left_path} and {right_path}"
        )

    mobile = np.stack([left_index[key] for key in common_keys], axis=0)
    reference = np.stack([right_index[key] for key in common_keys], axis=0)
    mobile_centroid = mobile.mean(axis=0)
    reference_centroid = reference.mean(axis=0)
    mobile_centered = mobile - mobile_centroid
    reference_centered = reference - reference_centroid
    covariance = mobile_centered.T @ reference_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    aligned = mobile_centered @ rotation + reference_centroid
    deltas = aligned - reference
    return float(np.sqrt(np.mean(np.sum(deltas * deltas, axis=1))))


def _protein_sequences_by_chain(
    structure_path: Path,
    *,
    focus_chain_ids: tuple[str, ...] = (),
) -> dict[str, str]:
    summary = summarize_structure(structure_path)
    focus = list(focus_chain_ids) if focus_chain_ids else list(summary.chain_ids)
    sequences: dict[str, str] = {}
    for chain_id in focus:
        try:
            sequences[chain_id] = extract_protein_sequence(structure_path, chain_id)
        except ValueError:
            continue
    return sequences


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    reference_rows = [
        row for row in rows if row.get("benchmark_kind") == "reference_mutant"
    ]
    cycle_rows = [row for row in rows if row.get("benchmark_kind") == "cyclic_round_trip"]

    def _mean(values: list[float]) -> float | None:
        return None if not values else float(statistics.mean(values))

    def _max(values: list[float]) -> float | None:
        return None if not values else float(max(values))

    reference_ca = [
        float(row["predicted_vs_reference_ca_rmsd_angstrom"])
        for row in reference_rows
        if row.get("predicted_vs_reference_ca_rmsd_angstrom") is not None
    ]
    reference_backbone = [
        float(row["predicted_vs_reference_backbone_rmsd_angstrom"])
        for row in reference_rows
        if row.get("predicted_vs_reference_backbone_rmsd_angstrom") is not None
    ]
    cycle_ca = [
        float(row["round_trip_ca_rmsd_angstrom"])
        for row in cycle_rows
        if row.get("round_trip_ca_rmsd_angstrom") is not None
    ]
    cycle_backbone = [
        float(row["round_trip_backbone_rmsd_angstrom"])
        for row in cycle_rows
        if row.get("round_trip_backbone_rmsd_angstrom") is not None
    ]
    return {
        "total_cases": len(rows),
        "reference_cases": len(reference_rows),
        "cycle_cases": len(cycle_rows),
        "successful_cases": sum(row.get("local_edit_status") == "ok" for row in rows),
        "reference_quality_ok_cases": sum(
            row.get("quality_status") == "ok" for row in reference_rows
        ),
        "cycle_quality_ok_cases": sum(
            row.get("quality_status") == "ok" for row in cycle_rows
        ),
        "reference_ca_rmsd_mean_angstrom": _mean(reference_ca),
        "reference_ca_rmsd_max_angstrom": _max(reference_ca),
        "reference_backbone_rmsd_mean_angstrom": _mean(reference_backbone),
        "reference_backbone_rmsd_max_angstrom": _max(reference_backbone),
        "cycle_ca_rmsd_mean_angstrom": _mean(cycle_ca),
        "cycle_ca_rmsd_max_angstrom": _max(cycle_ca),
        "cycle_backbone_rmsd_mean_angstrom": _mean(cycle_backbone),
        "cycle_backbone_rmsd_max_angstrom": _max(cycle_backbone),
    }


def _local_edit_result_payload(result: LocalEditResult) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "mutation": asdict(result.mutation),
        "source_kind": result.source_kind,
        "source_path": str(result.source_path),
        "source_cache_hit": result.source_cache_hit,
        "mutant_structure_path": (
            None
            if result.mutant_structure_path is None
            else str(result.mutant_structure_path)
        ),
        "local_edit_status": result.local_edit_status,
        "prepared_from_cif": result.prepared_from_cif,
        "runtime_seconds": result.runtime_seconds,
        "failure_reason": result.failure_reason,
        "report_path": str(result.report_path),
        "mutant_structure_summary": result.mutant_structure_summary,
    }


def run_reference_mutation_case(
    case: ReferenceMutationCase,
    *,
    work_dir: str | Path,
    cache_dir: str | Path | None = None,
    session: requests.Session | None = None,
    harness: DdgBenchmarkHarness | None = None,
) -> LocalEditBenchmarkCaseResult:
    source_path = _resolve_structure_path(
        structure_path=case.source_structure_path,
        pdb_id=case.source_pdb_id,
        cache_dir=cache_dir,
        session=session,
    )
    reference_path = _resolve_structure_path(
        structure_path=case.reference_structure_path,
        pdb_id=case.reference_pdb_id,
        cache_dir=cache_dir,
        session=session,
    )

    local_result = run_local_mutation_case(
        mutation=case.mutation,
        structure_path=source_path,
        work_dir=work_dir,
        case_id=case.case_id,
        cache_dir=cache_dir,
        session=session,
        harness=harness,
        notes=case.description,
    )

    row: dict[str, object] = {
        "case_id": case.case_id,
        "benchmark_kind": "reference_mutant",
        "category": case.category,
        "description": case.description,
        "source_pdb_id": case.source_pdb_id,
        "reference_pdb_id": case.reference_pdb_id,
        "mutation_id": case.mutation.mutation_id,
        "focus_chain_ids": ",".join(case.focus_chain_ids),
        "local_edit_status": local_result.local_edit_status,
        "failure_reason": local_result.failure_reason,
        "runtime_seconds": local_result.runtime_seconds,
    }
    if (
        local_result.local_edit_status != "ok"
        or local_result.mutant_structure_path is None
    ):
        row["quality_status"] = "failed"
        return LocalEditBenchmarkCaseResult(
            case_id=case.case_id,
            benchmark_kind="reference_mutant",
            category=case.category,
            row=row,
            step_results=(local_result,),
        )

    predicted_path = local_result.mutant_structure_path
    predicted_vs_reference_ca = _rmsd_after_superposition(
        predicted_path,
        reference_path,
        atom_names={"CA"},
        focus_chain_ids=case.focus_chain_ids,
    )
    predicted_vs_reference_backbone = _rmsd_after_superposition(
        predicted_path,
        reference_path,
        atom_names={"N", "CA", "C", "O"},
        focus_chain_ids=case.focus_chain_ids,
    )
    reference_vs_wt_ca = _rmsd_after_superposition(
        reference_path,
        source_path,
        atom_names={"CA"},
        focus_chain_ids=case.focus_chain_ids,
    )
    reference_vs_wt_backbone = _rmsd_after_superposition(
        reference_path,
        source_path,
        atom_names={"N", "CA", "C", "O"},
        focus_chain_ids=case.focus_chain_ids,
    )
    predicted_vs_wt_ca = _rmsd_after_superposition(
        predicted_path,
        source_path,
        atom_names={"CA"},
        focus_chain_ids=case.focus_chain_ids,
    )
    predicted_vs_wt_backbone = _rmsd_after_superposition(
        predicted_path,
        source_path,
        atom_names={"N", "CA", "C", "O"},
        focus_chain_ids=case.focus_chain_ids,
    )

    reference_sequences = _protein_sequences_by_chain(
        reference_path,
        focus_chain_ids=case.focus_chain_ids,
    )
    predicted_sequences = _protein_sequences_by_chain(
        predicted_path,
        focus_chain_ids=case.focus_chain_ids,
    )
    source_summary = summarize_structure(source_path)
    predicted_summary = summarize_structure(predicted_path)
    focus_chain_lengths_preserved = {
        chain_id: len(predicted_sequences.get(chain_id, ""))
        == len(reference_sequences.get(chain_id, ""))
        for chain_id in (case.focus_chain_ids or tuple(reference_sequences))
    }
    quality_ok = (
        predicted_vs_reference_ca <= max(1.5, reference_vs_wt_ca + 0.5)
        and predicted_vs_reference_backbone <= max(1.25, reference_vs_wt_backbone + 0.5)
        and predicted_sequences == reference_sequences
        and all(focus_chain_lengths_preserved.values())
    )
    row.update(
        {
            "quality_status": "ok" if quality_ok else "warn",
            "predicted_structure_path": str(predicted_path),
            "reference_structure_path": str(reference_path),
            "predicted_vs_reference_ca_rmsd_angstrom": predicted_vs_reference_ca,
            "predicted_vs_reference_backbone_rmsd_angstrom": predicted_vs_reference_backbone,
            "reference_vs_wt_ca_rmsd_angstrom": reference_vs_wt_ca,
            "reference_vs_wt_backbone_rmsd_angstrom": reference_vs_wt_backbone,
            "predicted_vs_wt_ca_rmsd_angstrom": predicted_vs_wt_ca,
            "predicted_vs_wt_backbone_rmsd_angstrom": predicted_vs_wt_backbone,
            "predicted_sequences_match_reference": predicted_sequences == reference_sequences,
            "predicted_focus_chain_lengths_preserved": all(
                focus_chain_lengths_preserved.values()
            ),
            "predicted_summary_chain_ids": ",".join(predicted_summary.chain_ids),
            "source_summary_chain_ids": ",".join(source_summary.chain_ids),
        }
    )
    return LocalEditBenchmarkCaseResult(
        case_id=case.case_id,
        benchmark_kind="reference_mutant",
        category=case.category,
        row=row,
        step_results=(local_result,),
    )


def run_cyclic_mutation_case(
    case: CyclicMutationCase,
    *,
    work_dir: str | Path,
    cache_dir: str | Path | None = None,
    session: requests.Session | None = None,
    harness: DdgBenchmarkHarness | None = None,
) -> LocalEditBenchmarkCaseResult:
    source_path = _resolve_structure_path(
        structure_path=case.source_structure_path,
        pdb_id=case.source_pdb_id,
        cache_dir=cache_dir,
        session=session,
    )
    source_sequences = _protein_sequences_by_chain(
        source_path,
        focus_chain_ids=case.focus_chain_ids,
    )
    current_structure_path = source_path
    step_results: list[LocalEditResult] = []
    for index, mutation in enumerate(case.steps, start=1):
        result = run_local_mutation_case(
            mutation=mutation,
            structure_path=current_structure_path,
            work_dir=work_dir,
            case_id=f"{case.case_id}_step_{index}",
            cache_dir=cache_dir,
            session=session,
            harness=harness,
            notes=case.description,
        )
        step_results.append(result)
        if result.local_edit_status != "ok" or result.mutant_structure_path is None:
            row = {
                "case_id": case.case_id,
                "benchmark_kind": "cyclic_round_trip",
                "category": case.category,
                "description": case.description,
                "source_pdb_id": case.source_pdb_id,
                "focus_chain_ids": ",".join(case.focus_chain_ids),
                "step_count": len(case.steps),
                "local_edit_status": result.local_edit_status,
                "failure_reason": result.failure_reason,
                "quality_status": "failed",
            }
            return LocalEditBenchmarkCaseResult(
                case_id=case.case_id,
                benchmark_kind="cyclic_round_trip",
                category=case.category,
                row=row,
                step_results=tuple(step_results),
            )
        current_structure_path = result.mutant_structure_path

    final_path = current_structure_path
    final_summary = summarize_structure(final_path)
    source_summary = summarize_structure(source_path)
    final_sequences = _protein_sequences_by_chain(
        final_path,
        focus_chain_ids=case.focus_chain_ids,
    )
    focus_chain_lengths_preserved = {
        chain_id: len(final_sequences.get(chain_id, ""))
        == len(source_sequences.get(chain_id, ""))
        for chain_id in (case.focus_chain_ids or tuple(source_sequences))
    }
    ca_rmsd = _rmsd_after_superposition(
        final_path,
        source_path,
        atom_names={"CA"},
        focus_chain_ids=case.focus_chain_ids,
    )
    backbone_rmsd = _rmsd_after_superposition(
        final_path,
        source_path,
        atom_names={"N", "CA", "C", "O"},
        focus_chain_ids=case.focus_chain_ids,
    )
    quality_ok = (
        final_sequences == source_sequences
        and all(focus_chain_lengths_preserved.values())
        and ca_rmsd <= 1.0
        and backbone_rmsd <= 0.75
    )
    row = {
        "case_id": case.case_id,
        "benchmark_kind": "cyclic_round_trip",
        "category": case.category,
        "description": case.description,
        "source_pdb_id": case.source_pdb_id,
        "focus_chain_ids": ",".join(case.focus_chain_ids),
        "step_count": len(case.steps),
        "local_edit_status": "ok",
        "failure_reason": None,
        "quality_status": "ok" if quality_ok else "warn",
        "final_structure_path": str(final_path),
        "round_trip_ca_rmsd_angstrom": ca_rmsd,
        "round_trip_backbone_rmsd_angstrom": backbone_rmsd,
        "final_sequences_match_source": final_sequences == source_sequences,
        "final_focus_chain_lengths_preserved": all(
            focus_chain_lengths_preserved.values()
        ),
        "final_summary_chain_ids": ",".join(final_summary.chain_ids),
        "source_summary_chain_ids": ",".join(source_summary.chain_ids),
        "runtime_seconds": sum(result.runtime_seconds for result in step_results),
    }
    return LocalEditBenchmarkCaseResult(
        case_id=case.case_id,
        benchmark_kind="cyclic_round_trip",
        category=case.category,
        row=row,
        step_results=tuple(step_results),
    )


def default_reference_mutation_cases() -> tuple[ReferenceMutationCase, ...]:
    return (
        ReferenceMutationCase(
            case_id="t4_lysozyme_l99a",
            source_pdb_id="1L63",
            reference_pdb_id="181L",
            mutation=MutationInput("A", "L", 99, "A"),
            focus_chain_ids=("A",),
            category="strong_single_mutation_change",
            description="T4 lysozyme cavity mutation L99A against the 181L reference structure.",
        ),
    )


def default_cyclic_mutation_cases() -> tuple[CyclicMutationCase, ...]:
    return (
        CyclicMutationCase(
            case_id="barnase_barstar_cycle_d39",
            source_pdb_id="1BRS",
            steps=(
                MutationInput("D", "D", 39, "A"),
                MutationInput("D", "A", 39, "N"),
                MutationInput("D", "N", 39, "D"),
            ),
            focus_chain_ids=("A", "D"),
            category="cyclic_round_trip_single_site",
            description="Single-site cycle D39A -> D39N -> D39 on barstar chain D.",
        ),
        CyclicMutationCase(
            case_id="barnase_barstar_cycle_d39_e80",
            source_pdb_id="1BRS",
            steps=(
                MutationInput("D", "D", 39, "A"),
                MutationInput("D", "E", 80, "G"),
                MutationInput("D", "A", 39, "D"),
                MutationInput("D", "G", 80, "E"),
            ),
            focus_chain_ids=("A", "D"),
            category="cyclic_round_trip_two_site",
            description="Two-site swap cycle D39A, E80G, A39D, G80E on barstar chain D.",
        ),
    )


def benchmark_cases_for_preset(
    preset: str,
) -> tuple[tuple[ReferenceMutationCase, ...], tuple[CyclicMutationCase, ...]]:
    reference_cases = default_reference_mutation_cases()
    cyclic_cases = default_cyclic_mutation_cases()
    if preset == "smoke":
        return reference_cases[:1], cyclic_cases[:1]
    if preset == "reference_only":
        return reference_cases, ()
    if preset == "strong_change":
        return reference_cases[1:], ()
    if preset == "full":
        return reference_cases, cyclic_cases
    raise ValueError(f"Unsupported benchmark preset: {preset}")


def run_local_edit_benchmark(
    *,
    output_root: str | Path,
    preset: str = "full",
    cache_dir: str | Path | None = None,
    session: requests.Session | None = None,
    harness: DdgBenchmarkHarness | None = None,
) -> LocalEditBenchmarkSuiteResult:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases_root = output_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)

    reference_cases, cyclic_cases = benchmark_cases_for_preset(preset)
    results: list[LocalEditBenchmarkCaseResult] = []
    for case in reference_cases:
        results.append(
            run_reference_mutation_case(
                case,
                work_dir=cases_root,
                cache_dir=cache_dir,
                session=session,
                harness=harness,
            )
        )
    for case in cyclic_cases:
        results.append(
            run_cyclic_mutation_case(
                case,
                work_dir=cases_root,
                cache_dir=cache_dir,
                session=session,
                harness=harness,
            )
        )

    rows = [result.row for result in results]
    summary_payload = {
        "preset": preset,
        "aggregate": _aggregate(rows),
        "rows": rows,
        "results": [
            {
                "case_id": result.case_id,
                "benchmark_kind": result.benchmark_kind,
                "category": result.category,
                "row": result.row,
                "step_results": [
                    _local_edit_result_payload(step) for step in result.step_results
                ],
            }
            for result in results
        ],
    }
    rows_csv_path = output_root / "rows.csv"
    summary_json_path = output_root / "summary.json"
    _write_rows_csv(rows_csv_path, rows)
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    return LocalEditBenchmarkSuiteResult(
        output_root=output_root,
        results=tuple(results),
        rows_csv_path=rows_csv_path,
        summary_json_path=summary_json_path,
    )
