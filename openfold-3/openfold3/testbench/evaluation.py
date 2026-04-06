from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from openfold3.benchmark.harness import HarnessReport, MethodResult


def _safe_mean(values: list[float]) -> float | None:
    return None if not values else float(statistics.mean(values))


def _safe_median(values: list[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
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
    if len(xs) < 2 or len(ys) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _kendall(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    concordant = 0
    discordant = 0
    for left in range(len(xs)):
        for right in range(left + 1, len(xs)):
            x_delta = xs[left] - xs[right]
            y_delta = ys[left] - ys[right]
            if x_delta == 0 or y_delta == 0:
                continue
            if x_delta * y_delta > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return None
    return float((concordant - discordant) / total)


def _mae(xs: list[float], ys: list[float]) -> float | None:
    if not xs or not ys:
        return None
    return float(statistics.mean(abs(x - y) for x, y in zip(xs, ys, strict=True)))


def _rmse(xs: list[float], ys: list[float]) -> float | None:
    if not xs or not ys:
        return None
    return float(
        math.sqrt(statistics.mean((x - y) ** 2 for x, y in zip(xs, ys, strict=True)))
    )


def _sign_accuracy(xs: list[float], ys: list[float]) -> float | None:
    if not xs or not ys:
        return None
    matches = 0
    for x, y in zip(xs, ys, strict=True):
        if (x >= 0 and y >= 0) or (x < 0 and y < 0):
            matches += 1
    return float(matches / len(xs))


def _top_k_overlap(xs: list[float], ys: list[float], k: int) -> float | None:
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


@dataclass(frozen=True)
class MethodEvaluation:
    method: str
    ok_count: int
    unavailable_count: int
    failed_count: int
    score_mean: float | None
    score_median: float | None
    runtime_mean_seconds: float | None
    pearson_vs_experimental: float | None
    spearman_vs_experimental: float | None
    kendall_vs_experimental: float | None
    mae_vs_experimental: float | None
    rmse_vs_experimental: float | None
    sign_accuracy_vs_experimental: float | None
    top_1_overlap_vs_experimental: float | None
    top_3_overlap_vs_experimental: float | None


@dataclass(frozen=True)
class EvaluationSummary:
    total_cases: int
    benchmark_cases: int
    exploratory_cases: int
    methods: tuple[MethodEvaluation, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def evaluate_reports(reports: list[HarnessReport]) -> EvaluationSummary:
    method_names = sorted({result.method for report in reports for result in report.results})
    methods: list[MethodEvaluation] = []
    benchmark_reports = [report for report in reports if report.experimental_ddg is not None]

    for method_name in method_names:
        method_results = [
            result
            for report in reports
            for result in report.results
            if result.method == method_name
        ]
        scores = [float(result.score) for result in method_results if result.score is not None]
        runtimes = [
            float(result.details["runtime_seconds"])
            for result in method_results
            if isinstance(result.details, dict) and "runtime_seconds" in result.details
        ]
        paired_scores: list[float] = []
        paired_truth: list[float] = []
        for report in benchmark_reports:
            result = next(
                (item for item in report.results if item.method == method_name and item.score is not None),
                None,
            )
            if result is None:
                continue
            paired_scores.append(float(result.score))
            paired_truth.append(float(report.experimental_ddg))
        methods.append(
            MethodEvaluation(
                method=method_name,
                ok_count=sum(result.status == "ok" for result in method_results),
                unavailable_count=sum(result.status == "unavailable" for result in method_results),
                failed_count=sum(result.status == "failed" for result in method_results),
                score_mean=_safe_mean(scores),
                score_median=_safe_median(scores),
                runtime_mean_seconds=_safe_mean(runtimes),
                pearson_vs_experimental=_pearson(paired_scores, paired_truth),
                spearman_vs_experimental=_spearman(paired_scores, paired_truth),
                kendall_vs_experimental=_kendall(paired_scores, paired_truth),
                mae_vs_experimental=_mae(paired_scores, paired_truth),
                rmse_vs_experimental=_rmse(paired_scores, paired_truth),
                sign_accuracy_vs_experimental=_sign_accuracy(paired_scores, paired_truth),
                top_1_overlap_vs_experimental=_top_k_overlap(paired_scores, paired_truth, 1),
                top_3_overlap_vs_experimental=_top_k_overlap(paired_scores, paired_truth, 3),
            )
        )

    return EvaluationSummary(
        total_cases=len(reports),
        benchmark_cases=len(benchmark_reports),
        exploratory_cases=len(reports) - len(benchmark_reports),
        methods=tuple(methods),
    )


def load_reports_from_paths(paths: list[Path]) -> list[HarnessReport]:
    reports: list[HarnessReport] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["results"] = tuple(MethodResult(**result) for result in payload["results"])
        reports.append(HarnessReport(**payload))
    return reports
