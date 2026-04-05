from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROFILE_EVENT_PREFIX = "OF3_PROFILE_EVENT"
PREDICT_TIMINGS_RE = re.compile(
    r"Predict timings for query_id\(s\)\s+(?P<query_ids>.+?):\s+batch_size=(?P<batch_size>\d+)\s+forward=(?P<forward>[0-9.]+)s\s+confidence=(?P<confidence>[0-9.]+)s"
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def parse_profile_event_from_line(line: str) -> dict[str, Any] | None:
    index = line.find(PROFILE_EVENT_PREFIX)
    if index < 0:
        return None
    payload_text = line[index + len(PROFILE_EVENT_PREFIX) :].strip()
    if not payload_text:
        return None
    try:
        return json.loads(payload_text)
    except json.JSONDecodeError:
        return None


def derive_events_from_log_line(
    line: str,
    *,
    run_id: str,
    case_id: str,
    benchmark_mode: str,
    run_mode: str,
    pid: int,
    timestamp: float,
    relative_seconds: float,
) -> list[dict[str, Any]]:
    base = {
        "run_id": run_id,
        "case_id": case_id,
        "benchmark_mode": benchmark_mode,
        "run_mode": run_mode,
        "pid": pid,
        "timestamp": timestamp,
        "relative_seconds": relative_seconds,
        "source": "stdout",
        "raw_line": line.rstrip(),
    }

    payload = parse_profile_event_from_line(line)
    if payload is not None:
        return [{**base, **payload}]

    if "Loading weights from" in line:
        return [
            {
                **base,
                "stage": "checkpoint_load",
                "event": "start",
                "inferred": True,
            }
        ]

    if "Beginning inference prediction" in line:
        return [
            {
                **base,
                "stage": "checkpoint_load",
                "event": "end",
                "inferred": True,
            },
            {
                **base,
                "stage": "predict_total",
                "event": "start",
                "inferred": True,
            },
        ]

    match = PREDICT_TIMINGS_RE.search(line)
    if match:
        query_ids = [item.strip() for item in match.group("query_ids").split(",")]
        forward_seconds = float(match.group("forward"))
        confidence_seconds = float(match.group("confidence"))
        return [
            {
                **base,
                "stage": "forward",
                "event": "end",
                "query_ids": query_ids,
                "duration_seconds": forward_seconds,
                "inferred": True,
            },
            {
                **base,
                "stage": "confidence",
                "event": "end",
                "query_ids": query_ids,
                "duration_seconds": confidence_seconds,
                "inferred": True,
            },
        ]

    return []


def event_rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "case_id",
                "benchmark_mode",
                "run_mode",
                "stage",
                "event",
                "timestamp",
                "relative_seconds",
                "pid",
            ]
        )
    frame = pd.DataFrame(rows)
    sort_columns = [
        column
        for column in ["timestamp", "relative_seconds", "stage", "event"]
        if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(by=sort_columns, ascending=[True] * len(sort_columns))
    return frame


def sample_rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "case_id",
                "benchmark_mode",
                "run_mode",
                "sample_seq",
                "timestamp",
                "relative_seconds",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        by=["sample_seq", "pid"],
        ascending=[True, True],
    )


def _stage_duration_seconds(events_df: pd.DataFrame, stage: str) -> float | None:
    if events_df.empty or "stage" not in events_df.columns:
        return None

    stage_df = events_df[events_df["stage"] == stage].copy()
    if stage_df.empty:
        return None

    if "duration_seconds" in stage_df.columns:
        durations = stage_df.loc[
            stage_df["event"] == "end",
            "duration_seconds",
        ].dropna()
        if not durations.empty:
            return float(durations.astype(float).sum())

    starts = stage_df.loc[stage_df["event"] == "start", "relative_seconds"].tolist()
    ends = stage_df.loc[stage_df["event"] == "end", "relative_seconds"].tolist()
    if not starts or not ends:
        return None

    total = 0.0
    for start, end in zip(starts, ends, strict=False):
        total += max(0.0, float(end) - float(start))
    return total


def summarize_case_metrics(
    samples_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    wall_seconds: float,
) -> dict[str, Any]:
    tick_df = pd.DataFrame()
    if not samples_df.empty:
        tick_df = samples_df.sort_values(by=["sample_seq", "pid"]).drop_duplicates(
            subset=["sample_seq"],
            keep="first",
        )

    checkpoint_load_seconds = _stage_duration_seconds(events_df, "checkpoint_load")
    predict_seconds = _stage_duration_seconds(events_df, "predict_total")
    if predict_seconds is None:
        predict_seconds = _stage_duration_seconds(events_df, "batch_total")
    forward_seconds = _stage_duration_seconds(events_df, "forward")
    confidence_seconds = _stage_duration_seconds(events_df, "confidence")

    if tick_df.empty:
        return {
            "wall_seconds": wall_seconds,
            "checkpoint_load_seconds": checkpoint_load_seconds,
            "predict_seconds": predict_seconds,
            "forward_seconds": forward_seconds,
            "confidence_seconds": confidence_seconds,
            "peak_rss_gb": None,
            "peak_cpu_percent": None,
            "peak_gpu_memory_gb": None,
            "mean_gpu_util_percent": None,
            "max_gpu_util_percent": None,
            "process_count_peak": None,
            "gpu_metrics_available": False,
            "gpu_error": None,
            "sample_count": 0,
            "event_count": int(len(events_df)),
        }

    gpu_available = bool(
        tick_df.get("gpu_metrics_available", pd.Series(dtype=bool)).fillna(False).any()
    )
    gpu_error = None
    if "gpu_error" in tick_df.columns:
        non_empty = tick_df["gpu_error"].dropna()
        if not non_empty.empty:
            gpu_error = str(non_empty.iloc[-1])

    return {
        "wall_seconds": wall_seconds,
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "predict_seconds": predict_seconds,
        "forward_seconds": forward_seconds,
        "confidence_seconds": confidence_seconds,
        "peak_rss_gb": float(tick_df["tree_total_rss_bytes"].max() / (1024**3)),
        "peak_cpu_percent": float(tick_df["tree_total_cpu_percent"].max()),
        "peak_gpu_memory_gb": (
            None
            if not gpu_available
            or "gpu_memory_used_total_mb" not in tick_df.columns
            or tick_df["gpu_memory_used_total_mb"].dropna().empty
            else float(tick_df["gpu_memory_used_total_mb"].max() / 1024.0)
        ),
        "mean_gpu_util_percent": (
            None
            if not gpu_available
            or "gpu_util_percent_max" not in tick_df.columns
            or tick_df["gpu_util_percent_max"].dropna().empty
            else float(tick_df["gpu_util_percent_max"].mean())
        ),
        "max_gpu_util_percent": (
            None
            if not gpu_available
            or "gpu_util_percent_max" not in tick_df.columns
            or tick_df["gpu_util_percent_max"].dropna().empty
            else float(tick_df["gpu_util_percent_max"].max())
        ),
        "process_count_peak": int(tick_df["process_count"].max()),
        "gpu_metrics_available": gpu_available,
        "gpu_error": gpu_error,
        "sample_count": int(len(tick_df)),
        "event_count": int(len(events_df)),
    }


def summarize_run(
    preview_df: pd.DataFrame,
    case_results_df: pd.DataFrame,
) -> dict[str, Any]:
    successful = case_results_df[case_results_df["status"] == "ok"].copy()
    failed = case_results_df[case_results_df["status"] == "failed"].copy()
    return {
        "n_requested_entries": int(len(preview_df)),
        "n_case_runs": int(len(case_results_df)),
        "n_successful_case_runs": int(len(successful)),
        "n_failed_case_runs": int(len(failed)),
        "successful_by_mode": (
            successful.groupby("run_mode")["pdb_id"].count().to_dict()
            if not successful.empty
            else {}
        ),
        "failed_by_mode": (
            failed.groupby("run_mode")["pdb_id"].count().to_dict()
            if not failed.empty
            else {}
        ),
        "successful_by_benchmark_mode": (
            successful.groupby("benchmark_mode")["pdb_id"].count().to_dict()
            if not successful.empty
            else {}
        ),
        "failed_case_ids": failed["case_id"].tolist() if not failed.empty else [],
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            "| "
            + " | ".join("" if row[column] is None else str(row[column]) for column in columns)
            + " |"
        )
    return "\n".join(lines)


def write_summary_markdown(
    path: str | Path,
    *,
    summary: dict[str, Any],
    case_results_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    plot_paths: dict[str, Path],
) -> None:
    path = Path(path)
    lines = [
        "# OpenFold3 Runtime Benchmark",
        "",
        f"- Requested entries: {summary['n_requested_entries']}",
        f"- Case runs: {summary['n_case_runs']}",
        f"- Successful case runs: {summary['n_successful_case_runs']}",
        f"- Failed case runs: {summary['n_failed_case_runs']}",
        "",
        "## Plot files",
    ]
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")

    lines.extend(
        [
            "",
            "## Cold Runs",
            _markdown_table(
                case_results_df[
                    case_results_df["run_mode"] == "cold"
                ][
                    [
                        "benchmark_mode",
                        "pdb_id",
                        "total_protein_length",
                        "wall_seconds",
                        "checkpoint_load_seconds",
                        "predict_seconds",
                        "peak_rss_gb",
                        "peak_gpu_memory_gb",
                        "max_gpu_util_percent",
                        "failure_reason",
                    ]
                ]
            ),
            "",
            "## Warm Runs",
            _markdown_table(
                case_results_df[
                    case_results_df["run_mode"] == "warm"
                ][
                    [
                        "benchmark_mode",
                        "pdb_id",
                        "total_protein_length",
                        "wall_seconds",
                        "checkpoint_load_seconds",
                        "predict_seconds",
                        "peak_rss_gb",
                        "peak_gpu_memory_gb",
                        "max_gpu_util_percent",
                        "failure_reason",
                    ]
                ]
            ),
            "",
            "## Pipeline Trace",
            _markdown_table(
                case_results_df[
                    case_results_df["benchmark_mode"] == "pipeline_trace"
                ][
                    [
                        "run_mode",
                        "pdb_id",
                        "total_protein_length",
                        "wall_seconds",
                        "checkpoint_load_seconds",
                        "predict_seconds",
                        "peak_rss_gb",
                        "peak_gpu_memory_gb",
                        "max_gpu_util_percent",
                        "failure_reason",
                    ]
                ]
            ),
            "",
            "## Failures",
            _markdown_table(
                failures_df[
                    [
                        "benchmark_mode",
                        "run_mode",
                        "pdb_id",
                        "failure_reason",
                    ]
                ]
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
