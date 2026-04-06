#!/usr/bin/env python
# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run RMSD benchmarking for a directory of OpenFold prediction outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    _path = Path(__file__).resolve()
    for _candidate in (_path.parent, *_path.parents):
        if (_candidate / "scripts").exists():
            sys.path.insert(0, str(_candidate))
            break

from scripts.dev.compute_rmsd_vs_pdb import (  # noqa: E402
    chain_summary,
    compute_rmsd,
    filter_atom_array,
    get_common_keys,
    kabsch_align,
    load_structure,
    parse_chain_list,
    select_matched_coordinates,
    summarize_match_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-root",
        type=Path,
        required=True,
        help="OpenFold output root with query/seed/*_model.(pdb|cif|bcif)",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        required=True,
        help="Flat directory with reference files named <query>.(cif|pdb|bcif)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for benchmark outputs",
    )
    parser.add_argument(
        "--atom-set",
        choices=("ca", "backbone", "all-heavy"),
        default="ca",
        help="Atoms to use for alignment and RMSD",
    )
    parser.add_argument(
        "--chains",
        type=str,
        default=None,
        help="Comma-separated chain IDs to use in both structures",
    )
    parser.add_argument(
        "--pred-chains",
        type=str,
        default=None,
        help="Comma-separated chain IDs to use only in the prediction",
    )
    parser.add_argument(
        "--ref-chains",
        type=str,
        default=None,
        help="Comma-separated chain IDs to use only in the reference",
    )
    parser.add_argument(
        "--fail-on-low-match",
        type=int,
        default=10,
        help="Fail a row if fewer than this many atoms are matched",
    )
    parser.add_argument(
        "--skip-missing-ref",
        action="store_true",
        help="Skip queries without matching reference structure instead of failing",
    )
    return parser.parse_args()


def find_prediction_files(pred_root: Path) -> list[Path]:
    prediction_files = []
    for suffix in ("*.pdb", "*.cif", "*.bcif"):
        prediction_files.extend(pred_root.rglob(f"*_model.{suffix.split('.')[-1]}"))
    return sorted(prediction_files)


def find_reference_path(ref_dir: Path, query_name: str) -> Path | None:
    for suffix in (".cif", ".pdb", ".bcif"):
        candidate = ref_dir / f"{query_name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def find_query_name(pred_root: Path, pred_path: Path) -> str:
    try:
        rel_parts = pred_path.relative_to(pred_root).parts
    except ValueError:
        return pred_path.parents[1].name
    if len(rel_parts) >= 3:
        return rel_parts[0]
    return pred_path.parents[1].name


def parse_seed_and_sample(pred_path: Path) -> tuple[str | None, str | None]:
    parts = pred_path.stem.split("_")
    try:
        seed_index = parts.index("seed")
        sample_index = parts.index("sample")
    except ValueError:
        return None, None
    return parts[seed_index + 1], parts[sample_index + 1]


def find_aggregated_confidence_path(pred_path: Path) -> Path | None:
    candidate = pred_path.with_name(
        pred_path.name.replace("_model.", "_confidences_aggregated.")
    )
    return candidate if candidate.exists() else None


def load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compute_row(
    pred_path: Path,
    ref_path: Path,
    atom_set: str,
    pred_chains: list[str] | None,
    ref_chains: list[str] | None,
    fail_on_low_match: int,
) -> dict[str, Any]:
    pred_structure = load_structure(pred_path)
    ref_structure = load_structure(ref_path)

    pred_filtered = filter_atom_array(pred_structure, atom_set=atom_set, chain_ids=pred_chains)
    ref_filtered = filter_atom_array(ref_structure, atom_set=atom_set, chain_ids=ref_chains)

    common_keys = get_common_keys(pred_filtered, ref_filtered)
    if len(common_keys) < fail_on_low_match:
        raise RuntimeError(
            "Too few matched atoms for a meaningful RMSD calculation: "
            f"{len(common_keys)} matched, threshold is {fail_on_low_match}"
        )

    pred_coords, ref_coords = select_matched_coordinates(pred_filtered, ref_filtered)
    pred_aligned = kabsch_align(pred_coords, ref_coords)

    return {
        "pred_available_chains": chain_summary(pred_structure),
        "ref_available_chains": chain_summary(ref_structure),
        "coverage": summarize_match_coverage(pred_filtered, ref_filtered, len(common_keys)),
        "rmsd_before_superposition": compute_rmsd(pred_coords, ref_coords),
        "rmsd_after_superposition": compute_rmsd(pred_aligned, ref_coords),
    }


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    aggregated = row.get("aggregated_confidence") or {}
    coverage = row["coverage"]
    return {
        "query": row["query"],
        "seed": row["seed"],
        "sample": row["sample"],
        "pred_path": row["pred_path"],
        "ref_path": row["ref_path"],
        "atom_set": row["atom_set"],
        "matched_atom_count": coverage["matched_atom_count"],
        "pred_filtered_atom_count": coverage["pred_filtered_atom_count"],
        "ref_filtered_atom_count": coverage["ref_filtered_atom_count"],
        "rmsd_before_superposition": row["rmsd_before_superposition"],
        "rmsd_after_superposition": row["rmsd_after_superposition"],
        "avg_plddt": aggregated.get("avg_plddt"),
        "ptm": aggregated.get("ptm"),
        "iptm": aggregated.get("iptm"),
        "sample_ranking_score": aggregated.get("sample_ranking_score"),
    }


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_best_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(row["query"], []).append(row)

    best_rows = {}
    for query, query_rows in by_query.items():
        best_rows[query] = min(query_rows, key=lambda item: item["rmsd_after_superposition"])
    return best_rows


def build_summary(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    best_by_query = select_best_rows(rows) if rows else {}

    rmsd_values = [row["rmsd_after_superposition"] for row in rows]
    ranking_pairs = [
        (row["rmsd_after_superposition"], row["aggregated_confidence"]["sample_ranking_score"])
        for row in rows
        if row.get("aggregated_confidence")
        and row["aggregated_confidence"].get("sample_ranking_score") is not None
    ]
    plddt_pairs = [
        (row["rmsd_after_superposition"], row["aggregated_confidence"]["avg_plddt"])
        for row in rows
        if row.get("aggregated_confidence")
        and row["aggregated_confidence"].get("avg_plddt") is not None
    ]

    summary = {
        "n_rows": len(rows),
        "n_failures": len(failures),
        "queries_evaluated": sorted(best_by_query),
        "rmsd_after_superposition": {
            "min": min(rmsd_values) if rmsd_values else None,
            "max": max(rmsd_values) if rmsd_values else None,
            "mean": float(np.mean(rmsd_values)) if rmsd_values else None,
        },
        "correlations": {
            "rmsd_vs_sample_ranking_score": correlation(
                [item[0] for item in ranking_pairs],
                [item[1] for item in ranking_pairs],
            ),
            "rmsd_vs_avg_plddt": correlation(
                [item[0] for item in plddt_pairs],
                [item[1] for item in plddt_pairs],
            ),
        },
        "best_by_query": {
            query: {
                "pred_path": row["pred_path"],
                "ref_path": row["ref_path"],
                "seed": row["seed"],
                "sample": row["sample"],
                "matched_atom_count": row["coverage"]["matched_atom_count"],
                "rmsd_after_superposition": row["rmsd_after_superposition"],
                "avg_plddt": (
                    row["aggregated_confidence"].get("avg_plddt")
                    if row.get("aggregated_confidence")
                    else None
                ),
                "sample_ranking_score": (
                    row["aggregated_confidence"].get("sample_ranking_score")
                    if row.get("aggregated_confidence")
                    else None
                ),
            }
            for query, row in best_by_query.items()
        },
        "failures": failures,
    }
    return summary


def main() -> None:
    args = parse_args()

    shared_chains = parse_chain_list(args.chains)
    pred_chains = parse_chain_list(args.pred_chains) or shared_chains
    ref_chains = parse_chain_list(args.ref_chains) or shared_chains

    prediction_files = find_prediction_files(args.pred_root)
    if not prediction_files:
        raise FileNotFoundError(
            f"No prediction files matching *_model.(pdb|cif|bcif) found under {args.pred_root}"
        )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for pred_path in prediction_files:
        query_name = find_query_name(args.pred_root, pred_path)
        ref_path = find_reference_path(args.ref_dir, query_name)
        if ref_path is None:
            failure = {
                "query": query_name,
                "pred_path": str(pred_path.resolve()),
                "reason": f"Reference file not found in {args.ref_dir}",
            }
            if args.skip_missing_ref:
                failures.append(failure)
                continue
            raise FileNotFoundError(failure["reason"] + f" for query {query_name}")

        try:
            result = compute_row(
                pred_path=pred_path,
                ref_path=ref_path,
                atom_set=args.atom_set,
                pred_chains=pred_chains,
                ref_chains=ref_chains,
                fail_on_low_match=args.fail_on_low_match,
            )
        except Exception as exc:
            failures.append(
                {
                    "query": query_name,
                    "pred_path": str(pred_path.resolve()),
                    "ref_path": str(ref_path.resolve()),
                    "reason": str(exc),
                }
            )
            continue

        aggregated_path = find_aggregated_confidence_path(pred_path)
        aggregated_confidence = load_json_if_exists(aggregated_path)
        seed, sample = parse_seed_and_sample(pred_path)

        rows.append(
            {
                "query": query_name,
                "seed": seed,
                "sample": sample,
                "pred_path": str(pred_path.resolve()),
                "ref_path": str(ref_path.resolve()),
                "atom_set": args.atom_set,
                "pred_chains": pred_chains,
                "ref_chains": ref_chains,
                "aggregated_confidence_path": (
                    str(aggregated_path.resolve()) if aggregated_path else None
                ),
                "aggregated_confidence": aggregated_confidence,
                **result,
            }
        )

    flat_rows = [flatten_row(row) for row in rows]
    summary = build_summary(rows, failures)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.output_dir / "rmsd_rows.jsonl", rows)
    dump_csv(args.output_dir / "rmsd_rows.csv", flat_rows)
    dump_json(args.output_dir / "rmsd_summary.json", summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
