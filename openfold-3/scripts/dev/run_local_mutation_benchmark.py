#!/usr/bin/env python3
"""Run the reference-mutant and cyclic local FoldX mutation benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openfold3.benchmark.local_edit_benchmark import run_local_edit_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runtime_smoke/local_mutation_benchmark"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("runtime_smoke/local_mutation_benchmark_cache"),
    )
    parser.add_argument(
        "--preset",
        choices=("smoke", "reference_only", "strong_change", "full"),
        default="full",
    )
    args = parser.parse_args()

    result = run_local_edit_benchmark(
        output_root=args.output_root,
        cache_dir=args.cache_dir,
        preset=args.preset,
    )
    payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"summary_json={result.summary_json_path}")
    print(f"rows_csv={result.rows_csv_path}")


if __name__ == "__main__":
    main()
