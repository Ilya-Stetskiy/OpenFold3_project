#!/usr/bin/env python3
"""Summarize panel ddG stand results from state.sqlite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openfold3.panel_summary import summarize_panel_state_db, write_panel_summary_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-db", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-id")
    args = parser.parse_args()

    if args.state_db is None and args.run_root is None:
        raise ValueError("Either --state-db or --run-root is required")

    state_db = args.state_db
    if state_db is None:
        state_db = args.run_root / "state.sqlite"
    output_dir = args.output_dir
    if output_dir is None:
        root = args.run_root if args.run_root is not None else state_db.parent
        output_dir = root / "summary_export"

    summary = summarize_panel_state_db(state_db, target_id=args.target_id)
    outputs = write_panel_summary_outputs(summary, output_dir)
    payload = {
        "target_id": summary.target_id,
        "total_jobs": summary.total_jobs,
        "analyzed_jobs": summary.analyzed_jobs,
        "fully_scored_jobs": summary.fully_scored_jobs,
        "methods": list(summary.methods),
        "method_completion": summary.method_completion,
        "pairwise_spearman": summary.pairwise_spearman,
        "top_consensus": list(summary.top_consensus),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
