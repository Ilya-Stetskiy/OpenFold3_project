from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScreeningRow:
    mutation_id: str
    query_id: str
    query_hash: str | None
    sample_index: int | None
    seed: int | None
    sample_ranking_score: float | None
    iptm: float | None
    ptm: float | None
    avg_plddt: float | None
    gpde: float | None
    has_clash: float | None
    cache_hit: bool
    sequence_cache_hits: int
    query_result_cache_hit: bool
    cpu_prep_seconds: float
    gpu_inference_seconds: float
    total_seconds: float
    output_dir: str
    mutation_spec: dict[str, object] | None = None
    aggregated_confidence_path: str | None = None
    derived_interface_metrics: dict[str, object] | None = None
    query_output_cleaned: bool = False


@dataclass(frozen=True)
class ScreeningBridgeSummary:
    total_rows: int
    wt_rows: int
    mutated_rows: int
    cache_hit_rows: int
    query_result_cache_hits: int
    cpu_prep_seconds_mean: float
    gpu_inference_seconds_mean: float
    total_seconds_mean: float
    top_candidates_by_ranking_score: tuple[dict[str, object], ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def load_screening_rows(results_jsonl: Path) -> list[ScreeningRow]:
    rows: list[ScreeningRow] = []
    for line in results_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(ScreeningRow(**payload))
    return rows


def summarize_screening_rows(
    rows: list[ScreeningRow], top_k: int = 5
) -> ScreeningBridgeSummary:
    if not rows:
        raise ValueError("rows must not be empty")
    ranked_rows = [
        row
        for row in rows
        if row.sample_ranking_score is not None
    ]
    ranked_rows.sort(
        key=lambda row: (
            float(row.sample_ranking_score),
            float(row.avg_plddt or 0.0),
        ),
        reverse=True,
    )
    top_candidates = tuple(
        {
            "mutation_id": row.mutation_id,
            "query_id": row.query_id,
            "sample_ranking_score": row.sample_ranking_score,
            "avg_plddt": row.avg_plddt,
            "iptm": row.iptm,
            "ptm": row.ptm,
            "gpde": row.gpde,
            "total_seconds": row.total_seconds,
        }
        for row in ranked_rows[:top_k]
    )
    return ScreeningBridgeSummary(
        total_rows=len(rows),
        wt_rows=sum(row.mutation_id == "WT" for row in rows),
        mutated_rows=sum(row.mutation_id != "WT" for row in rows),
        cache_hit_rows=sum(row.cache_hit for row in rows),
        query_result_cache_hits=sum(row.query_result_cache_hit for row in rows),
        cpu_prep_seconds_mean=float(statistics.mean(row.cpu_prep_seconds for row in rows)),
        gpu_inference_seconds_mean=float(
            statistics.mean(row.gpu_inference_seconds for row in rows)
        ),
        total_seconds_mean=float(statistics.mean(row.total_seconds for row in rows)),
        top_candidates_by_ranking_score=top_candidates,
    )
