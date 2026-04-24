from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import AVAILABLE_METHODS, ESM2Adapter, FoldXAdapter, RosettaDDGAdapter, normalize_ddg
from .data.canonical import demo_records, load_canonical_records, parse_record
from .data.mutation_engine import apply_mutation
from .runners.pipeline import PipelineConfig, ensure_layout, layout_tree, run_pipeline
from .structures.loader import extract_mutation_site, load_structure


def _demo_foldx_record() -> object:
    structure_path = (Path(__file__).resolve().parents[2] / "tests" / "test_data" / "ddg_pipeline" / "demo_input.pdb").resolve()
    return parse_record(
        {
            "protein_id": "demo-foldx",
            "sequence": "L",
            "mutation": "L1A",
            "position": 1,
            "wt": "L",
            "mut": "A",
            "structure_paths": {
                "experimental": str(structure_path),
                "openfold": str(structure_path),
                "foldx": str(structure_path),
            },
            "experimental_ddg": -0.1,
        }
    )


def run_demo_validation(output_root: str | Path) -> None:
    ensure_layout(output_root)
    print(layout_tree(output_root))
    records = demo_records()
    print(json.dumps([record.to_dict() for record in records], indent=2, sort_keys=True))
    for record in records:
        mutated = apply_mutation(record.sequence, record.mutation)
        assert mutated == record.mutated_sequence
        print(record.mutation, mutated)
    print(f"Normalized FoldX example: {normalize_ddg('foldx', -1.25)}")
    print(f"Normalized Rosetta example: {normalize_ddg('rosetta', -1.25)}")
    print(f"Normalized ESM2 example: {normalize_ddg('esm2', -1.25)}")

    foldx_record = _demo_foldx_record()
    structure = load_structure(foldx_record.structure_paths.experimental)
    site = extract_mutation_site(structure, foldx_record.position, expected_wt=foldx_record.wt)
    print(
        json.dumps(
            {
                "structure_path": str(foldx_record.structure_paths.experimental),
                "site": {
                    "chain_id": site.chain_id,
                    "residue_id": site.residue_id,
                    "residue_name_1": site.residue_name_1,
                    "ca_coord": site.ca_coord,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )

    run_pipeline(
        [foldx_record],
        [FoldXAdapter(), RosettaDDGAdapter(number_of_runs=1, top_k=1), ESM2Adapter()],
        PipelineConfig(
            output_root=Path(output_root).resolve(),
            methods=("foldx", "rosetta", "esm2"),
            structure_sources=("experimental",),
            seed=17,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the validated FoldX + Rosetta + ESM2 ddG pipeline.")
    parser.add_argument("--dataset-json", type=Path, default=None, help="Canonical dataset JSON list.")
    parser.add_argument("--output-root", type=Path, required=True, help="Artifact root for the pipeline.")
    parser.add_argument("--methods", nargs="*", default=None, help="Subset of methods to run.")
    parser.add_argument("--demo", action="store_true", help="Run stage validations with built-in demo records.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    requested_methods = tuple(args.methods or AVAILABLE_METHODS.keys())
    unsupported = [method for method in requested_methods if method not in AVAILABLE_METHODS]
    if unsupported:
        raise ValueError(f"Unsupported methods requested: {', '.join(unsupported)}")

    if args.demo:
        run_demo_validation(args.output_root.resolve())
        return 0

    if args.dataset_json is None:
        raise ValueError("--dataset-json is required unless --demo is used")
    records = load_canonical_records(args.dataset_json)
    run_pipeline(
        records,
        [AVAILABLE_METHODS[method] for method in requested_methods],
        PipelineConfig(
            output_root=args.output_root.resolve(),
            methods=requested_methods,
            structure_sources=("experimental", "openfold", "foldx"),
            seed=17,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
