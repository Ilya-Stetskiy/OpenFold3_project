#!/usr/bin/env python3
"""Run the mutation-centric five-scale ddG benchmark over one or more cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openfold3.benchmark.harness import DdgBenchmarkHarness
from openfold3.benchmark.methods import multiscale_methods
from openfold3.benchmark.models import BenchmarkCase, MutationInput
from openfold3.testbench.models import TestbenchConfig
from openfold3.testbench.runner import TestbenchRunner, load_cases_from_json


def _parse_mutation(value: str) -> MutationInput:
    chain_part, mutation_part = value.split(":", maxsplit=1)
    from_residue = mutation_part[0]
    to_residue = mutation_part[-1]
    position = int(mutation_part[1:-1])
    return MutationInput(
        chain_id=chain_part,
        from_residue=from_residue,
        position_1based=position,
        to_residue=to_residue,
    )


def _single_case_from_args(args) -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            case_id=args.case_id,
            structure_path=args.structure,
            confidence_path=args.confidence_json,
            mutations=tuple(args.mutation),
            experimental_ddg=args.experimental_ddg,
            notes=args.notes,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-kind",
        choices=("benchmark", "exploratory"),
        default="exploratory",
    )
    parser.add_argument("--cases-json", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--structure", type=Path)
    parser.add_argument("--confidence-json", type=Path)
    parser.add_argument("--mutation", action="append", type=_parse_mutation, default=[])
    parser.add_argument("--experimental-ddg", type=float)
    parser.add_argument("--notes")
    args = parser.parse_args()

    if args.cases_json is not None:
        cases = load_cases_from_json(args.cases_json)
    else:
        if not args.case_id or args.structure is None:
            raise ValueError("Either --cases-json or both --case-id and --structure are required")
        cases = _single_case_from_args(args)

    harness = DdgBenchmarkHarness(methods=multiscale_methods())
    config = TestbenchConfig(
        output_root=args.output_root,
        dataset_kind=args.dataset_kind,
        notes=args.notes,
    )
    run_id, reports, summary = TestbenchRunner(config, harness=harness).run_cases(cases)
    payload = {
        "run_id": run_id,
        "cases": [report.case_id for report in reports],
        "methods": [method.name for method in multiscale_methods()],
        "evaluation": json.loads(summary.to_json()),
    }
    print(json.dumps(payload, indent=2))
    print(f"run_root={config.output_root.resolve()}")


if __name__ == "__main__":
    main()
