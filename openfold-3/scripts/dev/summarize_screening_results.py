#!/usr/bin/env python3
"""Summarize mutation_runner screening results into testbench-style aggregate metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from openfold3.testbench.screening_bridge import (
    load_screening_rows,
    summarize_screening_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    rows = load_screening_rows(args.results_jsonl)
    summary = summarize_screening_rows(rows, top_k=args.top_k)
    if args.output is not None:
        args.output.write_text(summary.to_json(), encoding="utf-8")
    print(summary.to_json())


if __name__ == "__main__":
    main()
