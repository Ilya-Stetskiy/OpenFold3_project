#!/usr/bin/env python
"""Run a lightweight ddG benchmark harness over a single CIF/PDB case."""

from __future__ import annotations

import argparse
from pathlib import Path

from openfold3.benchmark.harness import DdgBenchmarkHarness
from openfold3.benchmark.methods import default_methods
from openfold3.benchmark.models import BenchmarkCase, MutationInput


def _parse_mutation(value: str) -> MutationInput:
    try:
        chain_part, mutation_part = value.split(":", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Mutation must look like A:W42F"
        ) from exc
    if len(mutation_part) < 3:
        raise argparse.ArgumentTypeError("Mutation must look like A:W42F")
    from_residue = mutation_part[0]
    to_residue = mutation_part[-1]
    try:
        position = int(mutation_part[1:-1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Mutation position must be numeric") from exc
    return MutationInput(
        chain_id=chain_part,
        from_residue=from_residue,
        position_1based=position,
        to_residue=to_residue,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--confidence-json", type=Path)
    parser.add_argument("--mutation", action="append", type=_parse_mutation, default=[])
    parser.add_argument("--experimental-ddg", type=float)
    parser.add_argument("--notes")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime_smoke/ddg_benchmark/report.json"),
    )
    args = parser.parse_args()

    case = BenchmarkCase(
        case_id=args.case_id,
        structure_path=args.structure,
        confidence_path=args.confidence_json,
        mutations=tuple(args.mutation),
        experimental_ddg=args.experimental_ddg,
        notes=args.notes,
    )
    harness = DdgBenchmarkHarness(methods=default_methods())
    report = harness.run_case(case)
    harness.write_report(report, args.output)
    print(report.to_json())
    print(f"report_path={args.output.resolve()}")


if __name__ == "__main__":
    main()
