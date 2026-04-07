#!/usr/bin/env python3
"""Run the panel-based WT-plus-19-mutants ddG stand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openfold3.panel_stand import PanelDdgStandRunner, PanelStandConfig
from openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet


def _parse_positions(raw: str, *, sequence_length: int | None = None) -> tuple[int, ...]:
    values: list[int] = []
    for token in raw.replace("\n", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
            if start > end:
                raise ValueError(f"Invalid descending range: {token}")
            values.extend(range(start, end + 1))
            continue
        values.append(int(token))
    if not values:
        raise ValueError("At least one position is required")
    unique_values = tuple(sorted(set(values)))
    if sequence_length is not None:
        for value in unique_values:
            if value < 1 or value > sequence_length:
                raise ValueError(
                    f"Position {value} is outside sequence length {sequence_length}"
                )
    return unique_values


def _load_positions_csv(path: Path) -> tuple[int, ...]:
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(int(line.split(",")[0].strip()))
    if not values:
        raise ValueError(f"No positions found in {path}")
    return tuple(sorted(set(values)))


def _find_chain_sequence_length(query_json: Path, mutable_chain_id: str) -> int:
    query_set = InferenceQuerySet.from_json(query_json)
    if len(query_set.queries) != 1:
        raise ValueError(f"Expected exactly one WT query in {query_json}")
    query = next(iter(query_set.queries.values()))
    for chain in query.chains:
        if mutable_chain_id in chain.chain_ids and chain.sequence is not None:
            return len(str(chain.sequence))
    raise ValueError(f"Could not find mutable protein chain {mutable_chain_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--wt-query-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mutable-chain-id", required=True)
    parser.add_argument("--positions")
    parser.add_argument("--positions-csv", type=Path)
    parser.add_argument(
        "--all-chain-positions",
        action="store_true",
        help="Use every residue position from the mutable chain",
    )
    parser.add_argument("--runner-yaml", type=Path)
    parser.add_argument("--inference-ckpt-path", type=Path)
    parser.add_argument("--inference-ckpt-name")
    parser.add_argument("--msa-computation-settings-yaml", type=Path)
    parser.add_argument("--num-diffusion-samples", type=int)
    parser.add_argument("--num-model-seeds", type=int)
    parser.add_argument("--msa-panel-workers", type=int, default=1)
    parser.add_argument("--analysis-workers", type=int, default=4)
    parser.add_argument(
        "--predict-strategy",
        choices=("adaptive", "chunked", "single_batch"),
        default="adaptive",
        help="Adaptive compares warmup GPU predict vs CPU analysis and then chooses chunked overlap or one remaining predict batch",
    )
    parser.add_argument(
        "--predict-panel-chunk-size",
        type=int,
        default=8,
        help="Warmup and chunk size in panels; used for chunked mode and the first adaptive probe batch",
    )
    parser.add_argument(
        "--enable-profiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect CPU/GPU/process telemetry under output_root/profiling",
    )
    parser.add_argument(
        "--profiling-sample-interval-seconds",
        type=float,
        default=1.0,
        help="Telemetry sampling interval in seconds",
    )
    parser.add_argument(
        "--cleanup-intermediates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    chain_length = _find_chain_sequence_length(args.wt_query_json, args.mutable_chain_id)

    if args.all_chain_positions:
        positions = tuple(range(1, chain_length + 1))
    elif args.positions_csv is not None:
        positions = _load_positions_csv(args.positions_csv)
    elif args.positions:
        positions = _parse_positions(args.positions, sequence_length=chain_length)
    else:
        raise ValueError(
            "Provide --positions, --positions-csv, or --all-chain-positions"
        )

    config = PanelStandConfig(
        target_id=args.target_id,
        wt_query_json=args.wt_query_json,
        output_root=args.output_root,
        mutable_chain_id=args.mutable_chain_id,
        positions=positions,
        runner_yaml=args.runner_yaml,
        inference_ckpt_path=args.inference_ckpt_path,
        inference_ckpt_name=args.inference_ckpt_name,
        msa_computation_settings_yaml=args.msa_computation_settings_yaml,
        num_diffusion_samples=args.num_diffusion_samples,
        num_model_seeds=args.num_model_seeds,
        msa_panel_workers=args.msa_panel_workers,
        analysis_workers=args.analysis_workers,
        predict_strategy=args.predict_strategy,
        predict_panel_chunk_size=args.predict_panel_chunk_size,
        enable_profiling=args.enable_profiling,
        profiling_sample_interval_seconds=args.profiling_sample_interval_seconds,
        cleanup_intermediates=args.cleanup_intermediates,
    )
    runner = PanelDdgStandRunner(config)
    try:
        payload = runner.run()
    finally:
        runner.close()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
