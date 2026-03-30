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


def _parse_positions(raw: str) -> tuple[int, ...]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("At least one position is required")
    return tuple(values)


def _load_positions_csv(path: Path) -> tuple[int, ...]:
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(int(line.split(",")[0].strip()))
    if not values:
        raise ValueError(f"No positions found in {path}")
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--wt-query-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mutable-chain-id", required=True)
    parser.add_argument("--positions")
    parser.add_argument("--positions-csv", type=Path)
    parser.add_argument("--runner-yaml", type=Path)
    parser.add_argument("--inference-ckpt-path", type=Path)
    parser.add_argument("--inference-ckpt-name")
    parser.add_argument("--msa-computation-settings-yaml", type=Path)
    parser.add_argument("--num-diffusion-samples", type=int)
    parser.add_argument("--num-model-seeds", type=int)
    parser.add_argument("--msa-panel-workers", type=int, default=1)
    parser.add_argument("--analysis-workers", type=int, default=4)
    parser.add_argument(
        "--cleanup-intermediates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.positions_csv is not None:
        positions = _load_positions_csv(args.positions_csv)
    elif args.positions:
        positions = _parse_positions(args.positions)
    else:
        raise ValueError("Either --positions or --positions-csv is required")

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
