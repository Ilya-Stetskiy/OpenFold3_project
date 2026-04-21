from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .harness import DdgBenchmarkHarness, MethodResult
from .methods import (
    FoldXBuildModelMethod,
    HeliXonBindingDdgMethod,
    OpenFoldConfidenceMethod,
    PromptDdgMethod,
    RosettaScoreMethod,
    Saambe3DMethod,
    StructureInterfaceMethod,
)
from .models import BenchmarkCase, MutationInput


@dataclass(frozen=True)
class MutationAnalysisInput:
    structure_path: Path
    mutation: MutationInput | None = None
    case_id: str | None = None
    confidence_path: Path | None = None
    wt_structure_path: Path | None = None
    wt_confidence_path: Path | None = None
    methods: tuple[str, ...] | None = None
    output_dir: Path | None = None
    mode: str = "mutant_only"
    max_workers: int | None = None


@dataclass(frozen=True)
class MutationAnalysisResult:
    case_id: str
    mode: str
    structure_path: Path
    wt_structure_path: Path | None
    mutation: MutationInput | None
    methods: tuple[str, ...]
    max_workers: int
    result_json_path: Path | None
    result_csv_path: Path | None
    report: dict[str, object]


def _method_factories() -> dict[str, type[object]]:
    return {
        "structure_interface": StructureInterfaceMethod,
        "openfold_confidence": OpenFoldConfidenceMethod,
        "foldx": FoldXBuildModelMethod,
        "rosetta_score": RosettaScoreMethod,
        "saambe_3d": Saambe3DMethod,
        "helixon_binding_ddg": HeliXonBindingDdgMethod,
        "prompt_ddg": PromptDdgMethod,
    }


def available_method_names() -> tuple[str, ...]:
    return tuple(_method_factories())


def _normalize_mutation(mutation: MutationInput | dict[str, object] | None) -> MutationInput | None:
    if mutation is None or isinstance(mutation, MutationInput):
        return mutation
    return MutationInput(
        chain_id=str(mutation["chain_id"]),
        from_residue=str(mutation["from_residue"]),
        position_1based=int(mutation["position_1based"]),
        to_residue=str(mutation["to_residue"]),
    )


def _normalize_methods(
    methods: list[str] | tuple[str, ...] | None,
    *,
    has_mutation: bool,
    has_confidence: bool,
) -> tuple[str, ...]:
    if methods is None:
        default = [
            "foldx",
            "rosetta_score",
            "saambe_3d",
            "helixon_binding_ddg",
            "prompt_ddg",
        ]
        if has_confidence:
            default.append("openfold_confidence")
        if has_mutation:
            return tuple(default)
        fallback = ["rosetta_score"]
        if has_confidence:
            fallback.append("openfold_confidence")
        return tuple(fallback)

    normalized = tuple(str(name).strip() for name in methods if str(name).strip())
    unknown = sorted(set(normalized) - set(_method_factories()))
    if unknown:
        raise ValueError(
            f"Unsupported method names: {unknown}. Supported: {sorted(_method_factories())}"
        )
    return normalized


def _resolve_max_workers(max_workers: int | None, task_count: int) -> int:
    if task_count <= 0:
        return 1
    if max_workers is not None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        return min(max_workers, task_count)
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, task_count))


def _benchmark_case(
    *,
    case_id: str,
    structure_path: str | Path,
    mutation: MutationInput | None,
    confidence_path: str | Path | None,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        structure_path=Path(structure_path).expanduser().resolve(),
        confidence_path=(
            None if confidence_path is None else Path(confidence_path).expanduser().resolve()
        ),
        mutations=(() if mutation is None else (mutation,)),
    )


def _method_output_row(result: MethodResult, runtime_seconds: float) -> dict[str, object]:
    return {
        "method": result.method,
        "status": result.status,
        "score": result.score,
        "units": result.units,
        "runtime_seconds": runtime_seconds,
        "details": result.details,
    }


def _run_methods_for_case(
    case: BenchmarkCase,
    *,
    method_names: tuple[str, ...],
    max_workers: int | None,
) -> tuple[dict[str, object], list[dict[str, object]], int]:
    harness = DdgBenchmarkHarness(methods=[])
    context = harness.build_context(case)
    selected_methods = [(_method_factories()[name](), name) for name in method_names]
    resolved_workers = _resolve_max_workers(max_workers, len(selected_methods))

    def _invoke(method: object) -> tuple[MethodResult, float]:
        started = time.perf_counter()
        result = method.run(context)  # type: ignore[attr-defined]
        return result, time.perf_counter() - started

    rows_by_name: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=resolved_workers, thread_name_prefix="mutation-analysis") as executor:
        futures = {
            executor.submit(_invoke, method): name
            for method, name in selected_methods
        }
        for future in as_completed(futures):
            name = futures[future]
            result, runtime_seconds = future.result()
            rows_by_name[name] = _method_output_row(result, runtime_seconds)

    ordered_rows = [rows_by_name[name] for name in method_names]
    structure_summary = {
        "atom_count": context.structure_summary.atom_count,
        "residue_count": context.structure_summary.residue_count,
        "chain_ids": list(context.structure_summary.chain_ids),
        "chain_lengths": {
            chain_id: len(residues)
            for chain_id, residues in context.structure_summary.residues_by_chain.items()
        },
        "inferred_chain_groups": [
            list(group) for group in context.structure_summary.inferred_chain_groups
        ],
        "min_inter_chain_atom_distance": context.structure_summary.min_inter_chain_atom_distance,
        "interface_atom_contacts_5a": context.structure_summary.interface_atom_contacts_5a,
        "interface_ca_contacts_8a": context.structure_summary.interface_ca_contacts_8a,
        "chain_pair_min_distances": context.structure_summary.chain_pair_min_distances,
    }
    return structure_summary, ordered_rows, resolved_workers


def _delta_vs_wt(
    mutant_rows: list[dict[str, object]],
    wt_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    wt_by_method = {str(row["method"]): row for row in wt_rows}
    comparison: dict[str, dict[str, object]] = {}
    for mutant_row in mutant_rows:
        method = str(mutant_row["method"])
        wt_row = wt_by_method.get(method)
        if wt_row is None:
            continue
        mutant_score = mutant_row.get("score")
        wt_score = wt_row.get("score")
        if mutant_row.get("status") != "ok" or wt_row.get("status") != "ok":
            continue
        if mutant_score is None or wt_score is None:
            continue
        units = mutant_row.get("units")
        if units != wt_row.get("units"):
            continue
        comparison[method] = {
            "mutant_score": float(mutant_score),
            "wt_score": float(wt_score),
            "delta": float(mutant_score) - float(wt_score),
            "units": units,
        }
    return comparison


def _flat_results(
    *,
    case_id: str,
    structure_path: Path,
    wt_structure_path: Path | None,
    mutation: MutationInput | None,
    mutant_rows: list[dict[str, object]],
    comparison: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    mutation_id = None if mutation is None else mutation.mutation_id
    rows: list[dict[str, object]] = []
    for row in mutant_rows:
        method = str(row["method"])
        delta_payload = comparison.get(method)
        rows.append(
            {
                "case_id": case_id,
                "mutation_id": mutation_id,
                "method": method,
                "status": row.get("status"),
                "score": row.get("score"),
                "units": row.get("units"),
                "runtime_seconds": row.get("runtime_seconds"),
                "delta_vs_wt": (
                    None if delta_payload is None else delta_payload["delta"]
                ),
                "structure_path": str(structure_path),
                "wt_structure_path": (
                    None if wt_structure_path is None else str(wt_structure_path)
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_mutation_analysis(
    *,
    structure_path: str | Path,
    mutation: MutationInput | dict[str, object] | None = None,
    case_id: str | None = None,
    confidence_path: str | Path | None = None,
    wt_structure_path: str | Path | None = None,
    wt_confidence_path: str | Path | None = None,
    methods: list[str] | tuple[str, ...] | None = None,
    output_dir: str | Path | None = None,
    mode: str = "mutant_only",
    max_workers: int | None = None,
    write_outputs: bool = True,
) -> MutationAnalysisResult:
    normalized_mutation = _normalize_mutation(mutation)
    normalized_structure_path = Path(structure_path).expanduser().resolve()
    normalized_wt_structure_path = (
        None
        if wt_structure_path is None
        else Path(wt_structure_path).expanduser().resolve()
    )
    if mode not in {"mutant_only", "mutant_vs_wt"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == "mutant_vs_wt" and normalized_wt_structure_path is None:
        raise ValueError("wt_structure_path is required for mode='mutant_vs_wt'")

    resolved_case_id = case_id or normalized_structure_path.stem
    method_names = _normalize_methods(
        methods,
        has_mutation=normalized_mutation is not None,
        has_confidence=confidence_path is not None,
    )
    started_at = time.perf_counter()

    mutant_case = _benchmark_case(
        case_id=resolved_case_id,
        structure_path=normalized_structure_path,
        mutation=normalized_mutation,
        confidence_path=confidence_path,
    )
    mutant_summary, mutant_rows, resolved_workers = _run_methods_for_case(
        mutant_case,
        method_names=method_names,
        max_workers=max_workers,
    )

    wt_summary: dict[str, object] | None = None
    wt_rows: list[dict[str, object]] = []
    comparison: dict[str, dict[str, object]] = {}
    if normalized_wt_structure_path is not None:
        wt_case = _benchmark_case(
            case_id=f"{resolved_case_id}_WT",
            structure_path=normalized_wt_structure_path,
            mutation=None,
            confidence_path=wt_confidence_path,
        )
        wt_summary, wt_rows, _ = _run_methods_for_case(
            wt_case,
            method_names=method_names,
            max_workers=max_workers,
        )
        comparison = _delta_vs_wt(mutant_rows, wt_rows)

    flat_results = _flat_results(
        case_id=resolved_case_id,
        structure_path=normalized_structure_path,
        wt_structure_path=normalized_wt_structure_path,
        mutation=normalized_mutation,
        mutant_rows=mutant_rows,
        comparison=comparison,
    )
    elapsed_seconds = time.perf_counter() - started_at

    payload: dict[str, object] = {
        "case_id": resolved_case_id,
        "mode": mode,
        "inputs": {
            "structure_path": str(normalized_structure_path),
            "wt_structure_path": (
                None if normalized_wt_structure_path is None else str(normalized_wt_structure_path)
            ),
            "confidence_path": None if confidence_path is None else str(Path(confidence_path).expanduser().resolve()),
            "wt_confidence_path": None if wt_confidence_path is None else str(Path(wt_confidence_path).expanduser().resolve()),
            "mutation": None if normalized_mutation is None else asdict(normalized_mutation),
            "methods": list(method_names),
        },
        "runtime": {
            "max_workers": resolved_workers,
            "elapsed_seconds": elapsed_seconds,
        },
        "mutant": {
            "structure_summary": mutant_summary,
            "results": mutant_rows,
        },
        "wt": None if wt_summary is None else {
            "structure_summary": wt_summary,
            "results": wt_rows,
        },
        "comparison": {
            "delta_vs_wt": comparison,
        },
        "flat_results": flat_results,
    }

    result_json_path: Path | None = None
    result_csv_path: Path | None = None
    if write_outputs:
        resolved_output_dir = (
            normalized_structure_path.parent / "mutation_analysis"
            if output_dir is None
            else Path(output_dir).expanduser().resolve()
        )
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        result_json_path = resolved_output_dir / f"{resolved_case_id}.json"
        result_csv_path = resolved_output_dir / f"{resolved_case_id}.csv"
        result_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_csv(result_csv_path, flat_results)

    return MutationAnalysisResult(
        case_id=resolved_case_id,
        mode=mode,
        structure_path=normalized_structure_path,
        wt_structure_path=normalized_wt_structure_path,
        mutation=normalized_mutation,
        methods=method_names,
        max_workers=resolved_workers,
        result_json_path=result_json_path,
        result_csv_path=result_csv_path,
        report=payload,
    )


def run_mutation_analysis_batch(
    *,
    cases: list[MutationAnalysisInput],
    output_dir: str | Path | None = None,
    max_workers: int | None = None,
    write_outputs: bool = True,
) -> list[MutationAnalysisResult]:
    resolved_output_dir = (
        None if output_dir is None else Path(output_dir).expanduser().resolve()
    )
    results: list[MutationAnalysisResult] = []
    for case in cases:
        case_output_dir = resolved_output_dir
        if case_output_dir is None and case.output_dir is not None:
            case_output_dir = case.output_dir
        results.append(
            run_mutation_analysis(
                structure_path=case.structure_path,
                mutation=case.mutation,
                case_id=case.case_id,
                confidence_path=case.confidence_path,
                wt_structure_path=case.wt_structure_path,
                wt_confidence_path=case.wt_confidence_path,
                methods=None if case.methods is None else list(case.methods),
                output_dir=case_output_dir,
                mode=case.mode,
                max_workers=max_workers if max_workers is not None else case.max_workers,
                write_outputs=write_outputs,
            )
        )
    return results
