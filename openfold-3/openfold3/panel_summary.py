from __future__ import annotations

import csv
import json
import math
import sqlite3
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _rank_desc(values: list[float]) -> list[float]:
    sorted_pairs = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
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


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson(_rank_desc(xs), _rank_desc(ys))


def _z_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [0.0]
    mean = statistics.mean(values)
    pstdev = statistics.pstdev(values)
    if pstdev == 0:
        return [0.0 for _ in values]
    return [(value - mean) / pstdev for value in values]


@dataclass(frozen=True)
class PanelJobSummaryRow:
    job_id: str
    panel_id: str
    target_id: str
    chain_id: str
    position_1based: int
    from_residue: str
    to_residue: str
    predict_status: str
    analysis_status: str
    structure_path: str | None
    confidence_path: str | None
    report_path: str | None
    rosetta_delta_vs_wt: float | None
    method_scores: dict[str, float | None]
    method_statuses: dict[str, str]
    consensus_z: float | None = None
    consensus_rank_desc: float | None = None


@dataclass(frozen=True)
class PanelStandSummary:
    target_id: str
    total_jobs: int
    analyzed_jobs: int
    fully_scored_jobs: int
    methods: tuple[str, ...]
    method_completion: dict[str, int]
    pairwise_spearman: dict[str, float | None]
    top_consensus: tuple[dict[str, Any], ...]
    rows: tuple[PanelJobSummaryRow, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def summarize_panel_state_db(
    state_db_path: Path,
    *,
    target_id: str | None = None,
) -> PanelStandSummary:
    conn = sqlite3.connect(state_db_path)
    conn.row_factory = sqlite3.Row
    try:
        wt_row = conn.execute(
            "SELECT * FROM wt_baseline LIMIT 1" if target_id is None else
            "SELECT * FROM wt_baseline WHERE target_id = ?",
            () if target_id is None else (target_id,),
        ).fetchone()
        if wt_row is None:
            raise ValueError(f"No wt_baseline row found in {state_db_path}")
        resolved_target_id = str(wt_row["target_id"])

        job_rows = list(
            conn.execute(
                """
                SELECT *
                FROM jobs
                WHERE target_id = ?
                ORDER BY position_1based, to_residue, job_id
                """,
                (resolved_target_id,),
            )
        )
        method_rows = list(
            conn.execute(
                """
                SELECT mr.job_id, mr.method, mr.status, mr.score, mr.units, mr.details_json
                FROM method_results mr
                JOIN jobs j ON j.job_id = mr.job_id
                WHERE j.target_id = ?
                ORDER BY mr.job_id, mr.method
                """,
                (resolved_target_id,),
            )
        )
    finally:
        conn.close()

    method_by_job: dict[str, dict[str, sqlite3.Row]] = {}
    methods: list[str] = []
    for row in method_rows:
        job_id = str(row["job_id"])
        method = str(row["method"])
        method_by_job.setdefault(job_id, {})[method] = row
        if method not in methods:
            methods.append(method)

    if "rosetta_delta_vs_wt" not in methods:
        methods.append("rosetta_delta_vs_wt")

    completion = {method: 0 for method in methods}
    summary_rows: list[PanelJobSummaryRow] = []
    for job_row in job_rows:
        job_id = str(job_row["job_id"])
        method_scores: dict[str, float | None] = {}
        method_statuses: dict[str, str] = {}
        for method in methods:
            if method == "rosetta_delta_vs_wt":
                score = _safe_float(job_row["rosetta_delta_vs_wt"])
                method_scores[method] = score
                status = "ok" if score is not None else "missing"
                method_statuses[method] = status
                if score is not None:
                    completion[method] += 1
                continue
            stored = method_by_job.get(job_id, {}).get(method)
            if stored is None:
                method_scores[method] = None
                method_statuses[method] = "missing"
                continue
            method_scores[method] = _safe_float(stored["score"])
            status = str(stored["status"])
            method_statuses[method] = status
            if status == "ok" and method_scores[method] is not None:
                completion[method] += 1

        summary_rows.append(
            PanelJobSummaryRow(
                job_id=job_id,
                panel_id=str(job_row["panel_id"]),
                target_id=str(job_row["target_id"]),
                chain_id=str(job_row["chain_id"]),
                position_1based=int(job_row["position_1based"]),
                from_residue=str(job_row["from_residue"]),
                to_residue=str(job_row["to_residue"]),
                predict_status=str(job_row["predict_status"]),
                analysis_status=str(job_row["analysis_status"]),
                structure_path=job_row["structure_path"],
                confidence_path=job_row["confidence_path"],
                report_path=job_row["report_path"],
                rosetta_delta_vs_wt=_safe_float(job_row["rosetta_delta_vs_wt"]),
                method_scores=method_scores,
                method_statuses=method_statuses,
            )
        )

    informative_methods = [
        method
        for method in methods
        if method != "rosetta_score"
    ]
    per_method_z: dict[str, dict[str, float]] = {method: {} for method in informative_methods}
    for method in informative_methods:
        available = [
            (row.job_id, row.method_scores.get(method))
            for row in summary_rows
            if row.method_scores.get(method) is not None
        ]
        scores = [float(score) for _, score in available if score is not None]
        z_values = _z_scores(scores)
        for (job_id, _), z_value in zip(available, z_values, strict=True):
            per_method_z[method][job_id] = z_value

    hydrated_rows: list[PanelJobSummaryRow] = []
    consensus_values: list[float] = []
    consensus_job_ids: list[str] = []
    for row in summary_rows:
        z_components = [
            per_method_z[method][row.job_id]
            for method in informative_methods
            if row.job_id in per_method_z[method]
        ]
        consensus_z = None
        if z_components:
            consensus_z = float(statistics.mean(z_components))
            consensus_values.append(consensus_z)
            consensus_job_ids.append(row.job_id)
        hydrated_rows.append(
            PanelJobSummaryRow(
                **{
                    **asdict(row),
                    "consensus_z": consensus_z,
                    "consensus_rank_desc": None,
                }
            )
        )

    consensus_ranks = _rank_desc(consensus_values)
    consensus_rank_by_job = {
        job_id: rank for job_id, rank in zip(consensus_job_ids, consensus_ranks, strict=True)
    }
    final_rows: list[PanelJobSummaryRow] = []
    for row in hydrated_rows:
        final_rows.append(
            PanelJobSummaryRow(
                **{
                    **asdict(row),
                    "consensus_rank_desc": consensus_rank_by_job.get(row.job_id),
                }
            )
        )

    pairwise_spearman: dict[str, float | None] = {}
    for idx, left_method in enumerate(informative_methods):
        for right_method in informative_methods[idx + 1 :]:
            paired = [
                (row.method_scores.get(left_method), row.method_scores.get(right_method))
                for row in final_rows
                if row.method_scores.get(left_method) is not None
                and row.method_scores.get(right_method) is not None
            ]
            xs = [float(left) for left, _ in paired if left is not None]
            ys = [float(right) for _, right in paired if right is not None]
            pairwise_spearman[f"{left_method}__vs__{right_method}"] = _spearman(xs, ys)

    top_consensus = tuple(
        {
            "job_id": row.job_id,
            "panel_id": row.panel_id,
            "mutation": (
                f"{row.chain_id}:{row.from_residue}{row.position_1based}{row.to_residue}"
            ),
            "consensus_z": row.consensus_z,
            "consensus_rank_desc": row.consensus_rank_desc,
        }
        for row in sorted(
            [row for row in final_rows if row.consensus_z is not None],
            key=lambda item: item.consensus_z,
            reverse=True,
        )[:10]
    )

    analyzed_jobs = sum(row.analysis_status == "done" for row in final_rows)
    fully_scored_jobs = sum(
        all(
            row.method_scores.get(method) is not None
            for method in informative_methods
        )
        for row in final_rows
    )
    return PanelStandSummary(
        target_id=resolved_target_id,
        total_jobs=len(final_rows),
        analyzed_jobs=analyzed_jobs,
        fully_scored_jobs=fully_scored_jobs,
        methods=tuple(methods),
        method_completion=completion,
        pairwise_spearman=pairwise_spearman,
        top_consensus=top_consensus,
        rows=tuple(final_rows),
    )


def write_panel_summary_outputs(summary: PanelStandSummary, output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "panel_summary.json"
    rows_csv = output_root / "panel_summary_rows.csv"
    summary_json.write_text(summary.to_json(), encoding="utf-8")

    fieldnames = [
        "job_id",
        "panel_id",
        "target_id",
        "chain_id",
        "position_1based",
        "from_residue",
        "to_residue",
        "predict_status",
        "analysis_status",
        "structure_path",
        "confidence_path",
        "report_path",
        "rosetta_delta_vs_wt",
        "consensus_z",
        "consensus_rank_desc",
    ]
    method_score_fields = [f"score__{method}" for method in summary.methods]
    method_status_fields = [f"status__{method}" for method in summary.methods]
    with rows_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames + method_score_fields + method_status_fields,
        )
        writer.writeheader()
        for row in summary.rows:
            payload = {
                key: getattr(row, key)
                for key in fieldnames
            }
            for method in summary.methods:
                payload[f"score__{method}"] = row.method_scores.get(method)
                payload[f"status__{method}"] = row.method_statuses.get(method)
            writer.writerow(payload)
    return {"summary_json": summary_json, "rows_csv": rows_csv}
