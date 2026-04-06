#!/usr/bin/env python3
"""Summarize ddG testbench reports from JSON report files."""

from __future__ import annotations

import argparse
from pathlib import Path

from openfold3.testbench.evaluation import evaluate_reports, load_reports_from_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports = load_reports_from_paths(args.report_paths)
    summary = evaluate_reports(reports)
    if args.output is not None:
        args.output.write_text(summary.to_json(), encoding="utf-8")
    print(summary.to_json())


if __name__ == "__main__":
    main()
