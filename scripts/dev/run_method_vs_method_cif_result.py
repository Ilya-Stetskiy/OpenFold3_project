#!/usr/bin/env python3
"""Run a method-vs-method benchmark over OpenFold CIF outputs in cif_result."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from openfold3.benchmark.harness import DdgBenchmarkHarness, HarnessReport
from openfold3.benchmark.methods import FoldXBuildModelMethod, RosettaScoreMethod
from openfold3.benchmark.models import BenchmarkCase, MutationInput


@dataclass(frozen=True)
class MethodComparisonSummary:
    total_cases: int
    paired_ok_cases: int
    foldx_ok_cases: int
    rosetta_ok_cases: int
    foldx_runtime_mean_seconds: float | None
    rosetta_runtime_mean_seconds: float | None
    pearson_foldx_vs_rosetta: float | None
    spearman_foldx_vs_rosetta: float | None
    sign_agreement: float | None
    top_1_overlap_abs: float | None
    top_3_overlap_abs: float | None


def _safe_mean(values: list[float]) -> float | None:
    return None if not values else float(statistics.mean(values))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return float(cov / math.sqrt(var_x * var_y))


def _rank(values: list[float]) -> list[float]:
    sorted_pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(sorted_pairs):
        end = idx + 1
        while end < len(sorted_pairs) and sorted_pairs[end][1] == sorted_pairs[idx][1]:
            end += 1
        avg_rank = (idx + end - 1) / 2.0 + 1.0
        for original_index, _ in sorted_pairs[idx:end]:
            ranks[original_index] = avg_rank
        idx = end
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _top_k_overlap_abs(xs: list[float], ys: list[float], k: int) -> float | None:
    if len(xs) < k or len(ys) < k or k < 1:
        return None
    top_x = {
        index
        for index, _ in sorted(
            enumerate(xs), key=lambda item: abs(item[1]), reverse=True
        )[:k]
    }
    top_y = {
        index
        for index, _ in sorted(
            enumerate(ys), key=lambda item: abs(item[1]), reverse=True
        )[:k]
    }
    return float(len(top_x & top_y) / k)


def _sign_agreement(xs: list[float], ys: list[float]) -> float | None:
    if not xs:
        return None
    matches = sum((x >= 0) == (y >= 0) for x, y in zip(xs, ys, strict=True))
    return float(matches / len(xs))


def _parse_case_from_path(model_cif: Path) -> BenchmarkCase | None:
    case_token = model_cif.parts[-3]
    if "__WT" in case_token:
        return None
    if "__A_Q2" not in case_token:
        return None
    mutant = case_token.split("__A_Q2", maxsplit=1)[1]
    if len(mutant) != 1 or not mutant.isalpha():
        return None
    confidence_path = model_cif.with_name(
        model_cif.name.replace("_model.cif", "_confidences_aggregated.json")
    )
    return BenchmarkCase(
        case_id=case_token,
        structure_path=model_cif,
        confidence_path=confidence_path if confidence_path.exists() else None,
        mutations=(MutationInput("A", "Q", 2, mutant.upper()),),
        notes="OpenFold CIF result from cif_result corpus",
    )


def discover_cases(cif_result_root: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen_case_ids: set[str] = set()
    for model_cif in sorted(cif_result_root.rglob("*_model.cif")):
        if "/summary/" in model_cif.as_posix():
            continue
        if "/output/" not in model_cif.as_posix():
            continue
        case = _parse_case_from_path(model_cif)
        if case is None or case.case_id in seen_case_ids:
            continue
        seen_case_ids.add(case.case_id)
        cases.append(case)
    return cases


def summarize_reports(reports: list[HarnessReport]) -> dict[str, object]:
    foldx_scores: list[float] = []
    rosetta_scores: list[float] = []
    foldx_runtimes: list[float] = []
    rosetta_runtimes: list[float] = []
    per_case: list[dict[str, object]] = []

    for report in reports:
        foldx = next(result for result in report.results if result.method == "foldx")
        rosetta = next(
            result for result in report.results if result.method == "rosetta_score"
        )
        if foldx.status == "ok" and foldx.score is not None:
            foldx_runtimes.append(float(foldx.details.get("runtime_seconds", 0.0)))
        if rosetta.status == "ok" and rosetta.score is not None:
            rosetta_runtimes.append(float(rosetta.details.get("runtime_seconds", 0.0)))
        if (
            foldx.status == "ok"
            and foldx.score is not None
            and rosetta.status == "ok"
            and rosetta.score is not None
        ):
            foldx_scores.append(float(foldx.score))
            rosetta_scores.append(float(rosetta.score))
        per_case.append(
            {
                "case_id": report.case_id,
                "structure_path": report.structure_path,
                "mutation_ids": [
                    mutation["mutation_id"]
                    for mutation in foldx.details.get("mutation_ids", [])
                ]
                if isinstance(foldx.details.get("mutation_ids"), list)
                else [],
                "foldx": {
                    "status": foldx.status,
                    "score": foldx.score,
                    "units": foldx.units,
                    "runtime_seconds": foldx.details.get("runtime_seconds"),
                },
                "rosetta_score": {
                    "status": rosetta.status,
                    "score": rosetta.score,
                    "units": rosetta.units,
                    "runtime_seconds": rosetta.details.get("runtime_seconds"),
                },
            }
        )

    summary = MethodComparisonSummary(
        total_cases=len(reports),
        paired_ok_cases=len(foldx_scores),
        foldx_ok_cases=sum(
            next(result for result in report.results if result.method == "foldx").status
            == "ok"
            for report in reports
        ),
        rosetta_ok_cases=sum(
            next(
                result for result in report.results if result.method == "rosetta_score"
            ).status
            == "ok"
            for report in reports
        ),
        foldx_runtime_mean_seconds=_safe_mean(foldx_runtimes),
        rosetta_runtime_mean_seconds=_safe_mean(rosetta_runtimes),
        pearson_foldx_vs_rosetta=_pearson(foldx_scores, rosetta_scores),
        spearman_foldx_vs_rosetta=_spearman(foldx_scores, rosetta_scores),
        sign_agreement=_sign_agreement(foldx_scores, rosetta_scores),
        top_1_overlap_abs=_top_k_overlap_abs(foldx_scores, rosetta_scores, 1),
        top_3_overlap_abs=_top_k_overlap_abs(foldx_scores, rosetta_scores, 3),
    )
    return {
        "summary": asdict(summary),
        "cases": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cif-result-root",
        type=Path,
        default=Path("/mnt/d/Proga/OpenFold_codex/cif_result"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runtime_smoke/cif_result_method_benchmark"),
    )
    args = parser.parse_args()

    cases = discover_cases(args.cif_result_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "reports").mkdir(parents=True, exist_ok=True)
    (args.output_root / "cases_manifest.json").write_text(
        json.dumps(
            [
                {
                    "case_id": case.case_id,
                    "structure_path": str(case.structure_path),
                    "confidence_path": (
                        None if case.confidence_path is None else str(case.confidence_path)
                    ),
                    "mutations": [
                        {
                            "chain_id": mutation.chain_id,
                            "from_residue": mutation.from_residue,
                            "position_1based": mutation.position_1based,
                            "to_residue": mutation.to_residue,
                        }
                        for mutation in case.mutations
                    ],
                }
                for case in cases
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    harness = DdgBenchmarkHarness(
        methods=[FoldXBuildModelMethod(), RosettaScoreMethod()]
    )
    reports: list[HarnessReport] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
        report = harness.run_case(case)
        reports.append(report)
        output_path = args.output_root / "reports" / f"{case.case_id}.json"
        harness.write_report(report, output_path)
        print(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "results": [
                        {
                            "method": result.method,
                            "status": result.status,
                            "score": result.score,
                            "runtime_seconds": result.details.get("runtime_seconds"),
                        }
                        for result in report.results
                    ],
                }
            ),
            flush=True,
        )

    payload = summarize_reports(reports)
    summary_path = args.output_root / "method_vs_method_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"summary_path={summary_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
