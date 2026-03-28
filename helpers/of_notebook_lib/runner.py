from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .analysis import (
    best_samples_by_metric,
    collect_samples,
    copy_best_artifacts,
    samples_to_dataframe,
    write_best_samples_report,
)
from .config import RuntimeConfig
from .monitoring import RunMonitor


@dataclass(slots=True)
class RunResult:
    experiment_name: str
    run_dir: Path
    query_path: Path
    output_dir: Path
    summary_dir: Path
    log_path: Path
    samples_df: object
    return_code: int
    resource_csv_path: Path | None = None
    stage_marks_path: Path | None = None
    monitor_plot_path: Path | None = None


def _slug_timestamp(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")
    return f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_msa_cache_link(runtime: RuntimeConfig) -> None:
    target = runtime.fixed_msa_tmp_dir
    source = runtime.msa_cache_dir
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() or target.is_symlink():
        if target.is_symlink() and Path(os.readlink(target)) == source:
            return
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    os.symlink(source, target, target_is_directory=True)


def run_cmd(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line.rstrip(), flush=True)
            handle.write(line)
        return process.wait()


def run_prediction(
    runtime: RuntimeConfig,
    payload: dict,
    experiment_name: str,
    *,
    use_templates: bool = True,
    use_msa_server: bool = True,
    num_diffusion_samples: int = 1,
    num_model_seeds: int = 1,
    runner_yaml: str | Path | None = None,
    inference_ckpt_path: str | Path | None = None,
    inference_ckpt_name: str | None = None,
    enable_monitoring: bool = False,
) -> RunResult:
    run_dir = runtime.results_dir / _slug_timestamp(experiment_name)
    output_dir = run_dir / "output"
    summary_dir = run_dir / "summary"
    query_path = run_dir / "query.json"
    log_path = run_dir / "run_openfold.log"
    report_path = summary_dir / "best_samples_report.txt"
    run_params_path = summary_dir / "run_params.json"

    summary_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    monitor = RunMonitor(summary_dir) if enable_monitoring else None
    if monitor is not None:
        monitor.start()

    monitoring_artifacts = None
    try:
        query_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if monitor is not None:
            monitor.record_stage("query_written", f"queries={len(payload.get('queries', {}))}")

        ensure_msa_cache_link(runtime)
        env = runtime.build_env()
        if monitor is not None:
            monitor.record_stage("runtime_prepared")

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
        if runner_yaml:
            cmd.append(f"--runner_yaml={Path(runner_yaml)}")
        if inference_ckpt_path:
            cmd.append(f"--inference_ckpt_path={Path(inference_ckpt_path)}")
        if inference_ckpt_name:
            cmd.append(f"--inference_ckpt_name={inference_ckpt_name}")

        if monitor is not None:
            monitor.record_stage("openfold_started")
        return_code = run_cmd(cmd, env=env, log_path=log_path)
        if monitor is not None:
            monitor.record_stage("openfold_finished", f"return_code={return_code}")

        samples = collect_samples(output_dir)
        if monitor is not None:
            monitor.record_stage("samples_collected", f"count={len(samples)}")
        winners = best_samples_by_metric(samples)
        write_best_samples_report(report_path, samples, winners)
        copy_best_artifacts(summary_dir, winners)
        samples_df = samples_to_dataframe(samples)
        if monitor is not None:
            monitor.record_stage("summary_written", f"rows={len(samples_df)}")

        run_params = {
            "experiment_name": experiment_name,
            "run_dir": str(run_dir),
            "query_path": str(query_path),
            "output_dir": str(output_dir),
            "summary_dir": str(summary_dir),
            "log_path": str(log_path),
            "return_code": return_code,
            "use_templates": use_templates,
            "use_msa_server": use_msa_server,
            "num_diffusion_samples": num_diffusion_samples,
            "num_model_seeds": num_model_seeds,
            "runner_yaml": str(runner_yaml) if runner_yaml else None,
            "inference_ckpt_path": str(inference_ckpt_path) if inference_ckpt_path else None,
            "inference_ckpt_name": inference_ckpt_name,
            "enable_monitoring": enable_monitoring,
        }
        run_params_path.write_text(
            json.dumps(run_params, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    finally:
        if monitor is not None:
            monitoring_artifacts = monitor.stop()

    return RunResult(
        experiment_name=experiment_name,
        run_dir=run_dir,
        query_path=query_path,
        output_dir=output_dir,
        summary_dir=summary_dir,
        log_path=log_path,
        samples_df=samples_df,
        return_code=return_code,
        resource_csv_path=monitoring_artifacts.resource_csv_path if monitoring_artifacts else None,
        stage_marks_path=monitoring_artifacts.stage_marks_path if monitoring_artifacts else None,
        monitor_plot_path=monitoring_artifacts.monitor_plot_path if monitoring_artifacts else None,
    )
