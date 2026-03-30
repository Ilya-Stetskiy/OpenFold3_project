from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from .cif_utils import summarize_structure
from .confidence import load_confidence_json
from .models import BenchmarkCase, StructureSummary


@dataclass(frozen=True)
class MethodResult:
    method: str
    status: str
    score: float | None = None
    units: str | None = None
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessContext:
    case: BenchmarkCase
    structure_summary: StructureSummary
    confidence_payload: dict[str, object] | None


@dataclass(frozen=True)
class HarnessReport:
    case_id: str
    structure_path: str
    confidence_path: str | None
    structure_summary: dict[str, object]
    results: tuple[MethodResult, ...]
    experimental_ddg: float | None = None
    notes: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class BenchmarkMethod(Protocol):
    name: str

    def run(self, context: HarnessContext) -> MethodResult: ...


class DdgBenchmarkHarness:
    def __init__(self, methods: list[BenchmarkMethod]):
        self.methods = methods

    def build_context(self, case: BenchmarkCase) -> HarnessContext:
        resolved = case.with_resolved_paths()
        structure_summary = summarize_structure(
            resolved.structure_path, chain_groups=resolved.chain_groups
        )
        confidence_payload = None
        if resolved.confidence_path is not None:
            confidence_payload = load_confidence_json(resolved.confidence_path)
        return HarnessContext(
            case=resolved,
            structure_summary=structure_summary,
            confidence_payload=confidence_payload,
        )

    def run_case(self, case: BenchmarkCase) -> HarnessReport:
        context = self.build_context(case)
        collected_results: list[MethodResult] = []
        for method in self.methods:
            print(f"  START {method.name}", flush=True)
            started_at = time.perf_counter()
            result = method.run(context)
            runtime_seconds = time.perf_counter() - started_at
            if result.score is None:
                summary = result.status
            else:
                summary = f"{result.status}({result.score:.4f})"
            print(
                f"  DONE  {method.name} {summary} sec={runtime_seconds:.2f}",
                flush=True,
            )
            collected_results.append(result)
        results = tuple(collected_results)
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
        return HarnessReport(
            case_id=context.case.case_id,
            structure_path=str(context.case.structure_path),
            confidence_path=(
                None
                if context.case.confidence_path is None
                else str(context.case.confidence_path)
            ),
            structure_summary=structure_summary,
            results=results,
            experimental_ddg=context.case.experimental_ddg,
            notes=context.case.notes,
        )

    @staticmethod
    def write_report(report: HarnessReport, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.to_json(), encoding="utf-8")
