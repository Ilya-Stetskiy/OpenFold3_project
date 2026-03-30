#!/usr/bin/env python3
"""Run a structural FoldX-vs-Rosetta benchmark over ready CIF/PDB models."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from openfold3.benchmark.harness import DdgBenchmarkHarness, HarnessContext, HarnessReport, MethodResult
from openfold3.benchmark.methods import (
    _infer_rosetta_database_from_binary,
    _parse_rosetta_scorefile_total,
    _prepare_local_pdb_copy,
    _resolve_executable_path,
    _resolve_rosetta_binary,
    _resolve_rosetta_database,
)
from openfold3.benchmark.models import BenchmarkCase


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


def _find_confidence_path(structure_path: Path) -> Path | None:
    candidates = [
        structure_path.with_name(
            structure_path.name.replace("_model.cif", "_confidences_aggregated.json")
        ),
        structure_path.with_name(
            structure_path.name.replace("_model.cif", "_confidences.json")
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _discover_cases(cif_root: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen_case_ids: set[str] = set()
    for model_path in sorted(cif_root.rglob("*_model.cif")):
        if "/summary/" in model_path.as_posix():
            continue
        case_id = model_path.stem.replace("_model", "")
        if case_id in seen_case_ids:
            continue
        seen_case_ids.add(case_id)
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                structure_path=model_path,
                confidence_path=_find_confidence_path(model_path),
                notes="Structural method-vs-method benchmark over ready model files",
            )
        )
    return cases


def _parse_foldx_stability_score(stability_path: Path) -> tuple[float, list[str]]:
    rows = [
        line.strip().split("\t")
        for line in stability_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Empty FoldX stability output: {stability_path}")
    row = rows[-1]
    if len(row) < 2:
        raise ValueError(f"Unexpected FoldX stability row: {row}")
    return float(row[1]), row


class FoldXStabilityMethod:
    name: str = "foldx_stability"
    executable: str = "foldx"
    env_var_name: str = "FOLDX_BINARY"

    def __init__(self, timeout_seconds: int = 600):
        self.timeout_seconds = timeout_seconds

    def run(self, context: HarnessContext) -> MethodResult:
        executable_path = _resolve_executable_path(
            self.executable, env_var_name=self.env_var_name
        )
        if executable_path is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "executable_not_found",
                    "expected_executable": self.executable,
                    "env_var_name": self.env_var_name,
                },
            )

        started_at = time.perf_counter()
        work_dir = Path(tempfile.mkdtemp(prefix="foldx-stability-"))
        structure_copy, prepared_from_cif, pdb_prepare_seconds = _prepare_local_pdb_copy(
            context.case.structure_path, work_dir
        )
        command = [
            executable_path,
            "--command",
            "Stability",
            "--pdb",
            structure_copy.name,
            "--output-dir",
            ".",
            "--screen",
            "false",
        ]
        try:
            process = subprocess.run(
                command,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "foldx_stability_timeout",
                    "timeout_seconds": self.timeout_seconds,
                    "resolved_executable": executable_path,
                    "work_dir": str(work_dir),
                    "stdout": (exc.stdout or "")[-4000:],
                    "stderr": (exc.stderr or "")[-4000:],
                },
            )
        runtime_seconds = time.perf_counter() - started_at
        stability_path = work_dir / f"{structure_copy.stem}_0_ST.fxout"
        if process.returncode != 0:
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "foldx_stability_failed",
                    "returncode": process.returncode,
                    "resolved_executable": executable_path,
                    "work_dir": str(work_dir),
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                },
            )
        if not stability_path.exists():
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "foldx_stability_output_missing",
                    "resolved_executable": executable_path,
                    "work_dir": str(work_dir),
                    "expected_output": str(stability_path),
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                },
            )
        try:
            score, raw_row = _parse_foldx_stability_score(stability_path)
        except ValueError as exc:
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "foldx_stability_parse_failed",
                    "error": str(exc),
                    "resolved_executable": executable_path,
                    "work_dir": str(work_dir),
                    "stability_path": str(stability_path),
                },
            )
        return MethodResult(
            method=self.name,
            status="ok",
            score=score,
            units="kcal/mol",
            details={
                "protocol": "Stability",
                "resolved_executable": executable_path,
                "work_dir": str(work_dir),
                "prepared_input_pdb_path": str(structure_copy),
                "prepared_from_cif": prepared_from_cif,
                "pdb_prepare_runtime_seconds": pdb_prepare_seconds,
                "runtime_seconds": runtime_seconds,
                "stability_path": str(stability_path),
                "raw_row": raw_row,
                "stdout_tail": process.stdout[-2000:],
                "stderr_tail": process.stderr[-2000:],
            },
        )


class RosettaScoreTimeoutMethod:
    name: str = "rosetta_score"
    executable: str = "score_jd2"
    env_var_name: str = "ROSETTA_SCORE_JD2_BINARY"
    database_env_var_name: str = "ROSETTA_DATABASE"

    def __init__(self, timeout_seconds: int = 900):
        self.timeout_seconds = timeout_seconds

    def run(self, context: HarnessContext) -> MethodResult:
        executable_path = _resolve_rosetta_binary(
            self.executable, env_var_name=self.env_var_name
        )
        if executable_path is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "executable_not_found",
                    "expected_executable": self.executable,
                    "env_var_name": self.env_var_name,
                },
            )
        database_path = _resolve_rosetta_database()
        if database_path is None:
            database_path = _infer_rosetta_database_from_binary(executable_path)
        if database_path is None:
            return MethodResult(
                method=self.name,
                status="unavailable",
                details={
                    "reason": "rosetta_database_not_found",
                    "database_env_var_name": self.database_env_var_name,
                    "resolved_executable": executable_path,
                    "database_inferred_from_binary": False,
                },
            )

        started_at = time.perf_counter()
        work_dir = Path(tempfile.mkdtemp(prefix="rosetta-score-"))
        output_dir = work_dir / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        structure_copy, prepared_from_cif, pdb_prepare_seconds = _prepare_local_pdb_copy(
            context.case.structure_path, work_dir
        )
        scorefile_path = output_dir / "score.sc"
        command = [
            executable_path,
            "-database",
            database_path,
            "-in:file:s",
            str(structure_copy),
            "-out:file:scorefile",
            str(scorefile_path),
            "-out:path:all",
            str(output_dir),
            "-overwrite",
        ]
        try:
            process = subprocess.run(
                command,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "rosetta_process_timeout",
                    "timeout_seconds": self.timeout_seconds,
                    "resolved_executable": executable_path,
                    "database_path": database_path,
                    "work_dir": str(work_dir),
                    "stdout": (exc.stdout or "")[-4000:],
                    "stderr": (exc.stderr or "")[-4000:],
                },
            )
        runtime_seconds = time.perf_counter() - started_at
        if process.returncode != 0:
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "rosetta_process_failed",
                    "returncode": process.returncode,
                    "resolved_executable": executable_path,
                    "database_path": database_path,
                    "work_dir": str(work_dir),
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                },
            )
        if not scorefile_path.exists():
            return MethodResult(
                method=self.name,
                status="failed",
                details={
                    "reason": "rosetta_scorefile_missing",
                    "resolved_executable": executable_path,
                    "database_path": database_path,
                    "work_dir": str(work_dir),
                    "expected_scorefile_path": str(scorefile_path),
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                },
            )
        score = _parse_rosetta_scorefile_total(scorefile_path)
        return MethodResult(
            method=self.name,
            status="ok",
            score=score,
            units="rosetta_energy",
            details={
                "protocol": "score_jd2",
                "resolved_executable": executable_path,
                "database_path": database_path,
                "database_inferred_from_binary": (
                    "ROSETTA_DATABASE" not in os.environ
                    and _resolve_rosetta_database() != database_path
                ),
                "work_dir": str(work_dir),
                "prepared_input_pdb_path": str(structure_copy),
                "prepared_from_cif": prepared_from_cif,
                "pdb_prepare_runtime_seconds": pdb_prepare_seconds,
                "runtime_seconds": runtime_seconds,
                "scorefile_path": str(scorefile_path),
                "mutation_ids": [mutation.mutation_id for mutation in context.case.mutations],
                "stdout_tail": process.stdout[-2000:],
            },
        )


def _method_result_map(report: HarnessReport) -> dict[str, MethodResult]:
    return {result.method: result for result in report.results}


def _summarize_reports(reports: list[HarnessReport]) -> dict[str, object]:
    foldx_scores: list[float] = []
    rosetta_scores: list[float] = []
    foldx_runtimes: list[float] = []
    rosetta_runtimes: list[float] = []
    per_case: list[dict[str, object]] = []

    for report in reports:
        result_map = _method_result_map(report)
        foldx = result_map["foldx_stability"]
        rosetta = result_map["rosetta_score"]
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
                "chain_ids": report.structure_summary["chain_ids"],
                "chain_lengths": report.structure_summary["chain_lengths"],
                "foldx_stability": {
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
            _method_result_map(report)["foldx_stability"].status == "ok"
            for report in reports
        ),
        rosetta_ok_cases=sum(
            _method_result_map(report)["rosetta_score"].status == "ok"
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
        "--cif-root",
        type=Path,
        default=Path("/mnt/d/Proga/OpenFold_codex/cif_result"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runtime_smoke/structural_method_benchmark"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--foldx-timeout-seconds", type=int, default=600)
    parser.add_argument("--rosetta-timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    cases = _discover_cases(args.cif_root)
    if args.limit is not None:
        cases = cases[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports_dir = args.output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (args.output_root / "cases_manifest.json").write_text(
        json.dumps(
            [
                {
                    "case_id": case.case_id,
                    "structure_path": str(case.structure_path),
                    "confidence_path": (
                        None if case.confidence_path is None else str(case.confidence_path)
                    ),
                }
                for case in cases
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    foldx_method = FoldXStabilityMethod(timeout_seconds=args.foldx_timeout_seconds)
    rosetta_method = RosettaScoreTimeoutMethod(
        timeout_seconds=args.rosetta_timeout_seconds
    )
    harness = DdgBenchmarkHarness(methods=[foldx_method, rosetta_method])
    reports: list[HarnessReport] = []
    case_index = {case.case_id: index for index, case in enumerate(cases, start=1)}
    completed_case_ids: set[str] = set()
    if args.resume:
        for report_path in reports_dir.glob("*.json"):
            try:
                report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            case_id = report_payload.get("case_id")
            if isinstance(case_id, str):
                completed_case_ids.add(case_id)
    pending_cases = [case for case in cases if case.case_id not in completed_case_ids]

    def _run_single(case: BenchmarkCase) -> HarnessReport:
        local_harness = DdgBenchmarkHarness(
            methods=[
                FoldXStabilityMethod(timeout_seconds=args.foldx_timeout_seconds),
                RosettaScoreTimeoutMethod(
                    timeout_seconds=args.rosetta_timeout_seconds
                ),
            ]
        )
        report = local_harness.run_case(case)
        report_path = reports_dir / f"{case.case_id}.json"
        local_harness.write_report(report, report_path)
        return report

    if args.jobs <= 1:
        for case in pending_cases:
            print(
                f"[{case_index[case.case_id]}/{len(cases)}] {case.case_id}",
                flush=True,
            )
            report = _run_single(case)
            reports.append(report)
            result_map = _method_result_map(report)
            print(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "foldx_status": result_map["foldx_stability"].status,
                        "foldx_score": result_map["foldx_stability"].score,
                        "rosetta_status": result_map["rosetta_score"].status,
                        "rosetta_score": result_map["rosetta_score"].score,
                    }
                ),
                flush=True,
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_case = {
                executor.submit(_run_single, case): case for case in pending_cases
            }
            for future in concurrent.futures.as_completed(future_to_case):
                case = future_to_case[future]
                report = future.result()
                reports.append(report)
                result_map = _method_result_map(report)
                print(
                    f"[{case_index[case.case_id]}/{len(cases)}] {case.case_id}",
                    flush=True,
                )
                print(
                    json.dumps(
                        {
                            "case_id": case.case_id,
                            "foldx_status": result_map["foldx_stability"].status,
                            "foldx_score": result_map["foldx_stability"].score,
                            "rosetta_status": result_map["rosetta_score"].status,
                            "rosetta_score": result_map["rosetta_score"].score,
                        }
                    ),
                    flush=True,
                )
    reports.sort(key=lambda report: case_index[report.case_id])

    if args.resume:
        for case in cases:
            if case.case_id in completed_case_ids:
                report_path = reports_dir / f"{case.case_id}.json"
                try:
                    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                restored_results = tuple(
                    MethodResult(
                        method=result["method"],
                        status=result["status"],
                        score=result.get("score"),
                        units=result.get("units"),
                        details=result.get("details", {}),
                    )
                    for result in report_payload.get("results", [])
                    if isinstance(result, dict)
                )
                if not restored_results:
                    continue
                reports.append(
                    HarnessReport(
                        case_id=report_payload["case_id"],
                        structure_path=report_payload["structure_path"],
                        confidence_path=report_payload.get("confidence_path"),
                        structure_summary=report_payload["structure_summary"],
                        results=restored_results,
                        experimental_ddg=report_payload.get("experimental_ddg"),
                        notes=report_payload.get("notes"),
                    )
                )
        reports.sort(key=lambda report: case_index[report.case_id])

    payload = _summarize_reports(reports)
    summary_path = args.output_root / "method_vs_method_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"summary_path={summary_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
