#!/usr/bin/env python3
"""Inspect resolved current_internal execution jobs without running tools."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from execution_utils import ensure_dir, filter_jobs_for_stage, read_csv_dicts, write_json, safe_write_text


ROOT = Path(__file__).resolve().parents[2]


def resolved_jobs_path(dataset: str) -> Path:
    return ROOT / "data" / "processed" / dataset / f"{dataset}_recalc_jobs_resolved.csv"


def build_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    ready_counts = {
        "foldx_experimental": len(filter_jobs_for_stage(rows, "foldx_experimental")),
        "rosetta_experimental": len(filter_jobs_for_stage(rows, "rosetta_experimental")),
        "esm2_sequence_only": len(filter_jobs_for_stage(rows, "esm2_cpu_optional")),
        "foldx_openfold": len(filter_jobs_for_stage(rows, "foldx_openfold")),
        "rosetta_openfold": len(filter_jobs_for_stage(rows, "rosetta_openfold")),
    }
    by_method_source = Counter(f"{row.get('method', '')}/{row.get('structure_source', '')}" for row in rows)
    return {
        "total_jobs": len(rows),
        "jobs_by_method": dict(Counter(row.get("method", "") for row in rows)),
        "jobs_by_structure_source": dict(Counter(row.get("structure_source", "") for row in rows)),
        "jobs_by_method_source": dict(by_method_source),
        "jobs_by_input_ready": dict(Counter(row.get("input_ready", "") for row in rows)),
        "jobs_by_blocking_reason": dict(Counter(row.get("blocking_reason", "") or "none" for row in rows)),
        "ready_jobs": ready_counts,
    }


def render_markdown(dataset: str, filters: dict[str, str], summary: dict[str, Any]) -> str:
    lines = [
        f"# {dataset} resolved jobs inspection",
        "",
        "This report inspects resolved recalc jobs only. It does not run FoldX, Rosetta, OpenFold, or ESM2.",
        "",
        "## Filters",
        "",
    ]
    if filters:
        for key, value in filters.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Summary", "", f"- Total jobs: {summary['total_jobs']}"])
    for title, key in [
        ("Jobs by method", "jobs_by_method"),
        ("Jobs by structure source", "jobs_by_structure_source"),
        ("Jobs by method/source", "jobs_by_method_source"),
        ("Jobs by input_ready", "jobs_by_input_ready"),
        ("Jobs by blocking reason", "jobs_by_blocking_reason"),
        ("Ready jobs for executor stages", "ready_jobs"),
    ]:
        lines.extend(["", f"### {title}", ""])
        for name, count in sorted(summary[key].items()):
            lines.append(f"- `{name}`: {count}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="current_internal")
    parser.add_argument("--stage-group")
    parser.add_argument("--method")
    parser.add_argument("--structure-source")
    args = parser.parse_args(argv)

    rows = read_csv_dicts(resolved_jobs_path(args.dataset))
    filters: dict[str, str] = {}
    if args.method:
        filters["method"] = args.method
        rows = [row for row in rows if row.get("method") == args.method]
    if args.structure_source:
        filters["structure_source"] = args.structure_source
        rows = [row for row in rows if row.get("structure_source") == args.structure_source]
    if args.stage_group:
        filters["stage_group"] = args.stage_group
        if args.stage_group == "cpu_heavy":
            rows = [
                row
                for row in rows
                if (row.get("method"), row.get("structure_source"))
                in {("foldx", "experimental"), ("rosetta", "experimental"), ("esm2", "sequence_only")}
            ]
        elif args.stage_group == "postprocess":
            rows = [
                row
                for row in rows
                if (row.get("method"), row.get("structure_source")) in {("foldx", "openfold"), ("rosetta", "openfold")}
            ]

    summary = build_summary(rows)
    report_dir = ROOT / "reports" / args.dataset
    ensure_dir(report_dir)
    md_path = report_dir / f"{args.dataset}_resolved_jobs_inspection.md"
    json_path = report_dir / f"{args.dataset}_resolved_jobs_inspection.json"
    safe_write_text(md_path, render_markdown(args.dataset, filters, summary), force=True)
    write_json(json_path, {"dataset": args.dataset, "filters": filters, "summary": summary}, force=True)

    print(f"Dataset: {args.dataset}")
    print(f"Total jobs: {summary['total_jobs']}")
    print(f"Jobs by method: {json.dumps(summary['jobs_by_method'], sort_keys=True)}")
    print(f"Jobs by structure_source: {json.dumps(summary['jobs_by_structure_source'], sort_keys=True)}")
    print(f"Jobs by input_ready: {json.dumps(summary['jobs_by_input_ready'], sort_keys=True)}")
    print(f"Jobs by blocking_reason: {json.dumps(summary['jobs_by_blocking_reason'], sort_keys=True)}")
    print(f"Ready jobs: {json.dumps(summary['ready_jobs'], sort_keys=True)}")
    print(f"Report: {md_path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

