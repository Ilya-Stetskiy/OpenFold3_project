from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from of_notebook_lib.runner import ensure_msa_cache_link

from .analysis import (
    derive_events_from_log_line,
    event_rows_to_dataframe,
    sample_rows_to_dataframe,
    summarize_case_metrics,
    summarize_run,
    write_summary_markdown,
)
from .interop import (
    RuntimeConfig,
    clone_runtime,
    collect_entry_compositions,
    compositions_to_dataframe,
    default_runs_root,
    parse_pdb_ids,
)
from .models import RuntimeBenchmarkRunResult
from .plots import (
    write_gpu_util_scatter_svg,
    write_metric_scatter_svg,
    write_timeline_svg,
)
from .telemetry import NvidiaSmiCollector, append_jsonl, sample_process_tree


CASE_COLUMNS = [
    "case_id",
    "benchmark_mode",
    "run_mode",
    "status",
    "failure_reason",
    "pdb_id",
    "chain_group",
    "total_protein_length",
    "wall_seconds",
    "checkpoint_load_seconds",
    "predict_seconds",
    "forward_seconds",
    "confidence_seconds",
    "peak_rss_gb",
    "peak_cpu_percent",
    "peak_gpu_memory_gb",
    "mean_gpu_util_percent",
    "max_gpu_util_percent",
    "process_count_peak",
    "gpu_metrics_available",
    "gpu_error",
    "local_cache_state",
    "reference_path",
    "query_path",
    "run_dir",
    "output_dir",
    "log_path",
    "timeline_svg_path",
    "case_summary_path",
]


def _slug_timestamp(prefix: str) -> str:
    safe_prefix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in prefix).strip("_")
    return f"{safe_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _sha1_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return hashlib.sha1(file_path.read_bytes()).hexdigest()


def _chain_group(chain_count: int) -> str:
    if chain_count == 1:
        return "single_chain"
    if chain_count == 2:
        return "double_chain"
    return "other_chain"


def _build_query_payload(composition) -> dict[str, Any]:
    return {
        "queries": {
            composition.pdb_id: {
                "chains": composition.molecules,
            }
        }
    }


def _build_predict_cmd(
    runtime: RuntimeConfig,
    *,
    query_path: Path,
    output_dir: Path,
    use_templates: bool,
    use_msa_server: bool,
    num_diffusion_samples: int,
    num_model_seeds: int,
    runner_yaml: str | Path | None,
) -> list[str]:
    cmd = [
        str(runtime.openfold_runner),
        "predict",
        f"--query_json={query_path}",
        f"--output_dir={output_dir}",
        f"--use_templates={str(use_templates).lower()}",
        f"--use_msa_server={str(use_msa_server).lower()}",
        f"--num_diffusion_samples={num_diffusion_samples}",
        f"--num_model_seeds={num_model_seeds}",
    ]
    if runner_yaml is not None:
        cmd.append(f"--runner_yaml={Path(runner_yaml)}")
    return cmd


def _run_command_text(
    cmd: list[str],
    *,
    cwd: Path | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        return {
            "ok": False,
            "error": str(exc),
        }
    return {
        "ok": completed.returncode == 0,
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _collect_python_environment(runtime: RuntimeConfig) -> dict[str, Any]:
    cmd = [
        str(runtime.openfold_python),
        "-c",
        (
            "import json, sys\n"
            "payload={'python_executable': sys.executable, 'python_version': sys.version}\n"
            "try:\n"
            " import torch\n"
            " payload['torch_version']=torch.__version__\n"
            " payload['torch_cuda_version']=getattr(torch.version,'cuda',None)\n"
            " payload['cuda_available']=torch.cuda.is_available()\n"
            "except Exception as exc:\n"
            " payload['torch_error']=str(exc)\n"
            "print(json.dumps(payload, ensure_ascii=False))\n"
        ),
    ]
    result = _run_command_text(cmd, cwd=runtime.project_dir)
    if not result.get("ok"):
        return result
    try:
        return json.loads(result["stdout"])
    except (TypeError, json.JSONDecodeError):
        return {
            "ok": False,
            "error": "Could not decode runtime Python environment payload",
            "stdout": result.get("stdout"),
        }


def _collect_git_info(path: Path) -> dict[str, Any]:
    head = _run_command_text(["git", "rev-parse", "HEAD"], cwd=path)
    branch = _run_command_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path)
    return {
        "path": str(path),
        "head": head.get("stdout") if head.get("ok") else None,
        "branch": branch.get("stdout") if branch.get("ok") else None,
    }


def _collect_cpu_info() -> dict[str, Any]:
    lscpu = _run_command_text(["lscpu"])
    return {
        "platform": platform.platform(),
        "node": platform.node(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "lscpu": lscpu.get("stdout") if lscpu.get("ok") else None,
    }


def _collect_manifest(
    *,
    run_name: str,
    run_root: Path,
    runtime: RuntimeConfig,
    modes: tuple[str, ...],
    sampling_interval_seconds: float,
    size_sweep_ids: list[str],
    pipeline_trace_ids: list[str],
    runner_yaml: str | Path | None,
    gpu_probe: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "run_root": str(run_root),
        "sampling_interval_seconds": sampling_interval_seconds,
        "modes": list(modes),
        "size_sweep_core_pdb_ids": size_sweep_ids,
        "pipeline_trace_pdb_ids": pipeline_trace_ids,
        "runner_yaml": None if runner_yaml is None else str(Path(runner_yaml)),
        "runner_yaml_sha1": _sha1_file(runner_yaml),
        "benchmark_defaults": {
            "size_sweep_core": {
                "use_msa_server": False,
                "use_templates": False,
                "num_diffusion_samples": 1,
                "num_model_seeds": 1,
                "cold_definition": "local_cold",
                "warm_definition": "warm_reuse",
            },
            "pipeline_trace": {
                "use_msa_server": True,
                "use_templates": False,
                "num_diffusion_samples": 1,
                "num_model_seeds": 1,
            },
        },
        "runtime": {
            "project_dir": str(runtime.project_dir),
            "openfold_repo_dir": str(runtime.openfold_repo_dir),
            "openfold_prefix": str(runtime.openfold_prefix),
            "openfold_runner": str(runtime.openfold_runner),
            "openfold_python": str(runtime.openfold_python),
        },
        "environment": {
            "cpu": _collect_cpu_info(),
            "openfold_repo_git": _collect_git_info(runtime.openfold_repo_dir),
            "runtime_python": _collect_python_environment(runtime),
            "gpu_probe": gpu_probe,
        },
    }


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _case_key(benchmark_mode: str, pdb_id: str) -> str:
    return f"{benchmark_mode}:{pdb_id}"


def _case_dir_name(case_key: str) -> str:
    return case_key.replace(":", "__")


def _write_case_summary(path: Path, row: dict[str, Any]) -> Path:
    lines = [
        f"# {row['case_id']}",
        "",
        f"- status: {row['status']}",
        f"- benchmark_mode: {row['benchmark_mode']}",
        f"- run_mode: {row['run_mode']}",
        f"- pdb_id: {row['pdb_id']}",
        f"- wall_seconds: {row['wall_seconds']}",
        f"- checkpoint_load_seconds: {row['checkpoint_load_seconds']}",
        f"- predict_seconds: {row['predict_seconds']}",
        f"- forward_seconds: {row['forward_seconds']}",
        f"- confidence_seconds: {row['confidence_seconds']}",
        f"- peak_rss_gb: {row['peak_rss_gb']}",
        f"- peak_cpu_percent: {row['peak_cpu_percent']}",
        f"- peak_gpu_memory_gb: {row['peak_gpu_memory_gb']}",
        f"- mean_gpu_util_percent: {row['mean_gpu_util_percent']}",
        f"- max_gpu_util_percent: {row['max_gpu_util_percent']}",
        f"- process_count_peak: {row['process_count_peak']}",
        f"- gpu_metrics_available: {row['gpu_metrics_available']}",
        f"- failure_reason: {row['failure_reason']}",
        f"- query_path: {row['query_path']}",
        f"- output_dir: {row['output_dir']}",
        f"- log_path: {row['log_path']}",
        f"- timeline_svg_path: {row['timeline_svg_path']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _normalize_modes(modes: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(modes, str):
        normalized = (modes.strip().lower(),)
    else:
        normalized = tuple(str(mode).strip().lower() for mode in modes if str(mode).strip())
    if not normalized:
        raise ValueError("At least one run mode must be provided")

    allowed_modes = {"cold", "warm"}
    invalid_modes = [mode for mode in normalized if mode not in allowed_modes]
    if invalid_modes:
        raise ValueError(
            f"Unsupported run mode(s): {invalid_modes}. Allowed: {sorted(allowed_modes)}"
        )
    return normalized


def _write_partial_case_results(path: Path, rows: list[dict[str, Any]]) -> Path:
    partial_df = _results_dataframe(rows)
    partial_df.to_csv(path, index=False)
    return path


def _detect_output_artifacts(output_dir: Path) -> dict[str, str | None]:
    timing_files = sorted(output_dir.rglob("timing.json"))
    summary_files = sorted(output_dir.rglob("summary.jsonl"))
    aggregated_files = sorted(output_dir.rglob("*_confidences_aggregated.json"))
    return {
        "timing_path": str(timing_files[0]) if timing_files else None,
        "summary_path": str(summary_files[0]) if summary_files else None,
        "aggregated_confidence_path": str(aggregated_files[0]) if aggregated_files else None,
    }


def _profile_predict_case(
    runtime: RuntimeConfig,
    *,
    query_path: Path,
    output_dir: Path,
    log_path: Path,
    run_id: str,
    case_id: str,
    benchmark_mode: str,
    run_mode: str,
    sampling_interval_seconds: float,
    use_templates: bool,
    use_msa_server: bool,
    runner_yaml: str | Path | None,
) -> tuple[int, float, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ensure_msa_cache_link(runtime)
    env = runtime.build_env()
    cmd = _build_predict_cmd(
        runtime,
        query_path=query_path,
        output_dir=output_dir,
        use_templates=use_templates,
        use_msa_server=use_msa_server,
        num_diffusion_samples=1,
        num_model_seeds=1,
        runner_yaml=runner_yaml,
    )

    process = subprocess.Popen(
        cmd,
        cwd=runtime.project_dir.resolve(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    started_wall = time.perf_counter()
    started_ts = time.time()
    gpu_collector = NvidiaSmiCollector()
    event_rows: list[dict[str, Any]] = [
        {
            "run_id": run_id,
            "case_id": case_id,
            "benchmark_mode": benchmark_mode,
            "run_mode": run_mode,
            "stage": "process",
            "event": "spawn",
            "timestamp": started_ts,
            "relative_seconds": 0.0,
            "pid": process.pid,
            "source": "wrapper",
            "command": cmd,
        }
    ]
    sample_rows: list[dict[str, Any]] = []
    previous_cpu_ticks: dict[int, tuple[float, float]] = {}
    sampler_lock = threading.Lock()

    def _sampler() -> None:
        nonlocal previous_cpu_ticks
        sample_seq = 0
        while True:
            sample_seq += 1
            now_ts = time.time()
            relative = time.perf_counter() - started_wall
            rows, next_ticks = sample_process_tree(
                process.pid,
                previous_cpu_ticks=previous_cpu_ticks,
                timestamp=now_ts,
                relative_seconds=relative,
                sample_seq=sample_seq,
                gpu_collector=gpu_collector,
                run_id=run_id,
                case_id=case_id,
                benchmark_mode=benchmark_mode,
                run_mode=run_mode,
            )
            with sampler_lock:
                sample_rows.extend(rows)
                previous_cpu_ticks = next_ticks
            if process.poll() is not None:
                break
            time.sleep(sampling_interval_seconds)

    sampler_thread = threading.Thread(target=_sampler, daemon=True)
    sampler_thread.start()

    with log_path.open("w", encoding="utf-8") as log_handle:
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            timestamp = time.time()
            relative_seconds = time.perf_counter() - started_wall
            event_rows.extend(
                derive_events_from_log_line(
                    line,
                    run_id=run_id,
                    case_id=case_id,
                    benchmark_mode=benchmark_mode,
                    run_mode=run_mode,
                    pid=process.pid,
                    timestamp=timestamp,
                    relative_seconds=relative_seconds,
                )
            )

    return_code = process.wait()
    sampler_thread.join()
    wall_seconds = time.perf_counter() - started_wall
    event_rows.append(
        {
            "run_id": run_id,
            "case_id": case_id,
            "benchmark_mode": benchmark_mode,
            "run_mode": run_mode,
            "stage": "process",
            "event": "exit",
            "timestamp": time.time(),
            "relative_seconds": wall_seconds,
            "pid": process.pid,
            "source": "wrapper",
            "return_code": return_code,
        }
    )

    artifacts = _detect_output_artifacts(output_dir)
    event_rows.append(
        {
            "run_id": run_id,
            "case_id": case_id,
            "benchmark_mode": benchmark_mode,
            "run_mode": run_mode,
            "stage": "writer_summary",
            "event": "end",
            "timestamp": time.time(),
            "relative_seconds": wall_seconds,
            "pid": process.pid,
            "source": "wrapper",
            **artifacts,
        }
    )
    return return_code, wall_seconds, sample_rows, event_rows, gpu_collector.probe()


def _build_case_row(
    *,
    case_id: str,
    benchmark_mode: str,
    run_mode: str,
    composition,
    query_path: Path | None,
    run_dir: Path | None,
    output_dir: Path | None,
    log_path: Path | None,
    timeline_svg_path: Path | None,
    case_summary_path: Path | None,
    metrics: dict[str, Any],
    failure_reason: str | None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "benchmark_mode": benchmark_mode,
        "run_mode": run_mode,
        "status": "failed" if failure_reason else "ok",
        "failure_reason": failure_reason,
        "pdb_id": composition.pdb_id,
        "chain_group": _chain_group(composition.chain_count),
        "total_protein_length": composition.total_protein_length,
        "wall_seconds": metrics.get("wall_seconds"),
        "checkpoint_load_seconds": metrics.get("checkpoint_load_seconds"),
        "predict_seconds": metrics.get("predict_seconds"),
        "forward_seconds": metrics.get("forward_seconds"),
        "confidence_seconds": metrics.get("confidence_seconds"),
        "peak_rss_gb": metrics.get("peak_rss_gb"),
        "peak_cpu_percent": metrics.get("peak_cpu_percent"),
        "peak_gpu_memory_gb": metrics.get("peak_gpu_memory_gb"),
        "mean_gpu_util_percent": metrics.get("mean_gpu_util_percent"),
        "max_gpu_util_percent": metrics.get("max_gpu_util_percent"),
        "process_count_peak": metrics.get("process_count_peak"),
        "gpu_metrics_available": metrics.get("gpu_metrics_available", False),
        "gpu_error": metrics.get("gpu_error"),
        "local_cache_state": "local_cold" if run_mode == "cold" else "warm_reuse",
        "reference_path": None if composition.source_path is None else str(composition.source_path),
        "query_path": None if query_path is None else str(query_path),
        "run_dir": None if run_dir is None else str(run_dir),
        "output_dir": None if output_dir is None else str(output_dir),
        "log_path": None if log_path is None else str(log_path),
        "timeline_svg_path": None if timeline_svg_path is None else str(timeline_svg_path),
        "case_summary_path": None if case_summary_path is None else str(case_summary_path),
    }


def _run_case(
    *,
    run_id: str,
    run_root: Path,
    runtime: RuntimeConfig,
    composition,
    benchmark_mode: str,
    run_mode: str,
    sampling_interval_seconds: float,
    runner_yaml: str | Path | None,
    use_templates: bool,
    use_msa_server: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    case_id = _case_key(benchmark_mode, composition.pdb_id)
    case_root = run_root / "cases" / _case_dir_name(case_id)
    shared_state_root = case_root / "shared_state"
    mode_root = case_root / "runs" / run_mode
    output_dir = mode_root / "output"
    query_path = mode_root / "query.json"
    log_path = mode_root / "run_openfold.log"
    timeline_svg_path = mode_root / "timeline.svg"
    case_summary_path = mode_root / "case_summary.md"

    if run_mode == "cold":
        _ensure_clean_dir(shared_state_root)
    mode_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_case = clone_runtime(
        runtime,
        results_dir=mode_root,
        msa_cache_dir=shared_state_root / "msa_cache" / "colabfold_msas",
        triton_cache_dir=shared_state_root / "triton_cache",
        fixed_msa_tmp_dir=shared_state_root / "fixed_msa_tmp",
    )

    query_path.write_text(
        json.dumps(_build_query_payload(composition), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return_code, wall_seconds, sample_rows, event_rows, gpu_probe = _profile_predict_case(
        runtime_case,
        query_path=query_path,
        output_dir=output_dir,
        log_path=log_path,
        run_id=run_id,
        case_id=case_id,
        benchmark_mode=benchmark_mode,
        run_mode=run_mode,
        sampling_interval_seconds=sampling_interval_seconds,
        use_templates=use_templates,
        use_msa_server=use_msa_server,
        runner_yaml=runner_yaml,
    )

    samples_df = sample_rows_to_dataframe(sample_rows)
    events_df = event_rows_to_dataframe(event_rows)
    metrics = summarize_case_metrics(
        samples_df,
        events_df,
        wall_seconds=wall_seconds,
    )
    failure_reason = None
    if return_code != 0:
        failure_reason = f"run_openfold exited with code {return_code}"

    if not samples_df.empty:
        write_timeline_svg(
            samples_df,
            events_df,
            output_path=timeline_svg_path,
            title=f"{composition.pdb_id} / {benchmark_mode} / {run_mode}",
        )

    row = _build_case_row(
        case_id=case_id,
        benchmark_mode=benchmark_mode,
        run_mode=run_mode,
        composition=composition,
        query_path=query_path,
        run_dir=mode_root,
        output_dir=output_dir,
        log_path=log_path,
        timeline_svg_path=timeline_svg_path if timeline_svg_path.exists() else None,
        case_summary_path=case_summary_path,
        metrics=metrics,
        failure_reason=failure_reason,
    )
    _write_case_summary(case_summary_path, row)
    row["case_summary_path"] = str(case_summary_path)
    row["gpu_probe_available"] = gpu_probe.get("available")
    return row, event_rows, sample_rows


def _results_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=CASE_COLUMNS)
    return pd.DataFrame(rows, columns=CASE_COLUMNS + ["gpu_probe_available"]).sort_values(
        by=["benchmark_mode", "pdb_id", "run_mode"],
        ascending=[True, True, True],
    )


def run_runtime_benchmark(
    runtime: RuntimeConfig,
    pdb_ids: str | list[str],
    *,
    modes: str | tuple[str, ...] | list[str] = ("cold", "warm"),
    sampling_interval_seconds: float = 1.0,
    output_root: str | Path | None = None,
    runner_yaml: str | Path | None = None,
    max_entries: int | None = None,
    cache_dir: str | Path | None = None,
    pipeline_trace_pdb_ids: str | list[str] | None = None,
) -> RuntimeBenchmarkRunResult:
    normalized_modes = _normalize_modes(modes)

    size_sweep_ids = parse_pdb_ids(pdb_ids, max_entries=max_entries)
    pipeline_trace_ids = (
        parse_pdb_ids(pipeline_trace_pdb_ids)
        if pipeline_trace_pdb_ids is not None
        else []
    )

    union_ids: list[str] = []
    for pdb_id in [*size_sweep_ids, *pipeline_trace_ids]:
        if pdb_id not in union_ids:
            union_ids.append(pdb_id)

    compositions = collect_entry_compositions(
        union_ids,
        cache_dir=cache_dir,
    )
    composition_by_id = {composition.pdb_id: composition for composition in compositions}
    preview_df = compositions_to_dataframe(compositions)
    preview_df["in_size_sweep_core"] = preview_df["pdb_id"].isin(size_sweep_ids)
    preview_df["in_pipeline_trace"] = preview_df["pdb_id"].isin(pipeline_trace_ids)

    runs_root = Path(output_root or default_runs_root()).expanduser().resolve()
    run_name = _slug_timestamp("runtime_benchmark")
    run_root = runs_root / run_name
    plots_dir = run_root / "plots"
    run_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    events_path = run_root / "events.jsonl"
    samples_path = run_root / "samples.jsonl"
    preview_path = run_root / "preview.csv"
    partial_case_results_path = run_root / "case_results.partial.csv"
    run_error_path = run_root / "run_error.txt"
    preview_df.to_csv(preview_path, index=False)

    manifest = _collect_manifest(
        run_name=run_name,
        run_root=run_root,
        runtime=runtime,
        modes=normalized_modes,
        sampling_interval_seconds=sampling_interval_seconds,
        size_sweep_ids=size_sweep_ids,
        pipeline_trace_ids=pipeline_trace_ids,
        runner_yaml=runner_yaml,
        gpu_probe=NvidiaSmiCollector().probe(),
    )
    manifest_path = _write_json(run_root / "manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    requested_case_specs: list[tuple[str, str, Any]] = []
    for pdb_id in size_sweep_ids:
        requested_case_specs.append(("size_sweep_core", pdb_id, composition_by_id[pdb_id]))
    for pdb_id in pipeline_trace_ids:
        requested_case_specs.append(("pipeline_trace", pdb_id, composition_by_id[pdb_id]))

    for benchmark_mode, _pdb_id, composition in requested_case_specs:
        if composition.status != "ok":
            for run_mode in normalized_modes:
                rows.append(
                    _build_case_row(
                        case_id=_case_key(benchmark_mode, composition.pdb_id),
                        benchmark_mode=benchmark_mode,
                        run_mode=run_mode,
                        composition=composition,
                        query_path=None,
                        run_dir=None,
                        output_dir=None,
                        log_path=None,
                        timeline_svg_path=None,
                        case_summary_path=None,
                        metrics={},
                        failure_reason=composition.issue or "Composition parsing failed",
                    )
                )
                _write_partial_case_results(partial_case_results_path, rows)
            continue

        for run_mode in normalized_modes:
            use_msa_server = benchmark_mode == "pipeline_trace"
            use_templates = False
            case_run_root = (
                run_root
                / "cases"
                / _case_dir_name(_case_key(benchmark_mode, composition.pdb_id))
                / "runs"
                / run_mode
            )
            try:
                row, case_event_rows, case_sample_rows = _run_case(
                    run_id=run_name,
                    run_root=run_root,
                    runtime=runtime,
                    composition=composition,
                    benchmark_mode=benchmark_mode,
                    run_mode=run_mode,
                    sampling_interval_seconds=sampling_interval_seconds,
                    runner_yaml=runner_yaml,
                    use_templates=use_templates,
                    use_msa_server=use_msa_server,
                )
                rows.append(row)
                append_jsonl(events_path, case_event_rows)
                append_jsonl(samples_path, case_sample_rows)
            except Exception as exc:
                error_text = traceback.format_exc()
                failure_path = _write_text(case_run_root / "run_error.txt", error_text)
                append_jsonl(
                    events_path,
                    [
                        {
                            "run_id": run_name,
                            "case_id": _case_key(benchmark_mode, composition.pdb_id),
                            "benchmark_mode": benchmark_mode,
                            "run_mode": run_mode,
                            "stage": "orchestration",
                            "event": "error",
                            "timestamp": time.time(),
                            "relative_seconds": 0.0,
                            "pid": None,
                            "source": "wrapper",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    ],
                )
                rows.append(
                    _build_case_row(
                        case_id=_case_key(benchmark_mode, composition.pdb_id),
                        benchmark_mode=benchmark_mode,
                        run_mode=run_mode,
                        composition=composition,
                        query_path=case_run_root / "query.json",
                        run_dir=case_run_root,
                        output_dir=case_run_root / "output",
                        log_path=failure_path,
                        timeline_svg_path=None,
                        case_summary_path=None,
                        metrics={},
                        failure_reason=f"{type(exc).__name__}: {exc}",
                    )
                )
            finally:
                _write_partial_case_results(partial_case_results_path, rows)

    try:
        case_results_df = _results_dataframe(rows)
        failures_df = case_results_df[case_results_df["status"] == "failed"].copy()
        case_results_csv_path = run_root / "case_results.csv"
        case_results_df.to_csv(case_results_csv_path, index=False)

        plot_paths = {
            "length_vs_wall_seconds_svg": write_metric_scatter_svg(
                case_results_df,
                output_path=plots_dir / "length_vs_wall_seconds.svg",
                y_column="wall_seconds",
                title="Protein length vs wall time",
                y_label="Wall time (s)",
            ),
            "length_vs_checkpoint_load_seconds_svg": write_metric_scatter_svg(
                case_results_df,
                output_path=plots_dir / "length_vs_checkpoint_load_seconds.svg",
                y_column="checkpoint_load_seconds",
                title="Protein length vs checkpoint load time",
                y_label="Checkpoint load time (s)",
            ),
            "length_vs_peak_gpu_memory_gb_svg": write_metric_scatter_svg(
                case_results_df,
                output_path=plots_dir / "length_vs_peak_gpu_memory_gb.svg",
                y_column="peak_gpu_memory_gb",
                title="Protein length vs peak GPU memory",
                y_label="Peak GPU memory (GiB)",
            ),
            "length_vs_peak_rss_gb_svg": write_metric_scatter_svg(
                case_results_df,
                output_path=plots_dir / "length_vs_peak_rss_gb.svg",
                y_column="peak_rss_gb",
                title="Protein length vs peak RSS",
                y_label="Peak RSS (GiB)",
            ),
            "length_vs_gpu_util_svg": write_gpu_util_scatter_svg(
                case_results_df,
                output_path=plots_dir / "length_vs_gpu_util.svg",
            ),
        }

        summary = summarize_run(preview_df, case_results_df)
        summary_path = run_root / "summary.md"
        write_summary_markdown(
            summary_path,
            summary=summary,
            case_results_df=case_results_df,
            failures_df=failures_df,
            plot_paths=plot_paths,
        )

        case_results_json_path = _write_json(
            run_root / "case_results.json",
            {
                "run_name": run_name,
                "run_root": str(run_root),
                "summary": summary,
                "preview": preview_df.to_dict(orient="records"),
                "records": case_results_df.to_dict(orient="records"),
            },
        )
    except Exception:
        _write_partial_case_results(partial_case_results_path, rows)
        _write_text(run_error_path, traceback.format_exc())
        raise

    return RuntimeBenchmarkRunResult(
        run_name=run_name,
        run_root=run_root,
        preview_df=preview_df,
        case_results_df=case_results_df,
        failures_df=failures_df,
        summary=summary,
        summary_path=summary_path,
        manifest_path=manifest_path,
        case_results_csv_path=case_results_csv_path,
        case_results_json_path=case_results_json_path,
        events_path=events_path,
        samples_path=samples_path,
        plot_paths=plot_paths,
    )
