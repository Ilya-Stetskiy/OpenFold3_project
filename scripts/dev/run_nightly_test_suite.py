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

"""One-command overnight test suite for mutation-screening validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from scripts.dev.runtime_monitor import RunMonitor


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(message: str, log_path: Path) -> None:
    line = f"{ts()} {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def stream_command(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    step_name: str,
) -> None:
    log_line(f"[{step_name}] START {' '.join(cmd)}", log_path)
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log_line(f"[{step_name}] {line.rstrip()}", log_path)
    return_code = process.wait()
    if return_code != 0:
        log_line(f"[{step_name}] FAIL exit_code={return_code}", log_path)
        raise subprocess.CalledProcessError(return_code, cmd)
    log_line(f"[{step_name}] OK", log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runtime_smoke/nightly_suite"),
        help="Root directory for all overnight artifacts and logs.",
    )
    parser.add_argument(
        "--comparison-query-json",
        type=Path,
        default=Path("examples/example_inference_inputs/query_spike_ace2_full.json"),
        help="Query JSON for original-vs-screening comparison stage.",
    )
    parser.add_argument(
        "--comparison-query-id",
        type=str,
        default="spike_ace2_full",
        help="Query id for the comparison stage.",
    )
    parser.add_argument(
        "--leucine-query-json",
        type=Path,
        default=Path(
            "examples/example_inference_inputs/query_test_leucine_zipper.json"
        ),
        help="Base query JSON for leucine zipper saturation stage.",
    )
    parser.add_argument(
        "--leucine-query-id",
        type=str,
        default="leucine_zipper",
        help="Query id for the leucine zipper stage.",
    )
    parser.add_argument(
        "--mutations-csv",
        type=Path,
        default=Path(
            "examples/example_screening_jobs/test_leucine_zipper_chainA_saturation.csv"
        ),
        help="Mutation CSV for the leucine zipper stage.",
    )
    parser.add_argument(
        "--runner-yaml",
        type=Path,
        default=Path(
            "examples/example_runner_yamls/nightly_metrics_only_low_mem.yml"
        ),
        help="Runner YAML used for both stages.",
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--num-diffusion-samples", type=int, default=1)
    parser.add_argument("--num-model-seeds", type=int, default=1)
    parser.add_argument("--comparison-tolerance", type=float, default=0.1)
    parser.add_argument("--num-cpu-workers", type=int, default=4)
    parser.add_argument("--max-inflight-queries", type=int, default=2)
    parser.add_argument("--min-free-disk-gb", type=float, default=2.0)
    parser.add_argument("--inference-ckpt-path", type=Path, default=None)
    parser.add_argument("--inference-ckpt-name", type=str, default=None)
    parser.add_argument("--include-wt", action="store_true")
    parser.add_argument("--use-msa-server", action="store_true")
    parser.add_argument("--use-templates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "nightly_suite.log"
    monitor = RunMonitor(logs_dir)
    monitor.start()

    comparison_out = output_root / "comparison"
    screening_out = output_root / "leucine_saturation"

    try:
        log_line(f"[suite] output_root={output_root}", log_path)
        log_line(f"[suite] log_file={log_path}", log_path)
        log_line(
            f"[suite] disk_guard_min_free_gb={args.min_free_disk_gb}",
            log_path,
        )
        log_line(
            f"[suite] monitoring_csv={monitor.artifacts.resource_csv_path}",
            log_path,
        )
        log_line(
            f"[suite] stage_marks_csv={monitor.artifacts.stage_marks_path}",
            log_path,
        )
        log_line(
            f"[suite] monitoring_png={monitor.artifacts.monitor_plot_path}",
            log_path,
        )
        monitor.record_stage("suite_initialized", f"output_root={output_root}")

        comparison_cmd = [
            args.python_bin,
            "scripts/dev/run_single_protein_comparison.py",
            "--query-json",
            str(args.comparison_query_json),
            "--query-id",
            args.comparison_query_id,
            "--runner-yaml",
            str(args.runner_yaml),
            "--output-root",
            str(comparison_out),
            "--num-diffusion-samples",
            str(args.num_diffusion_samples),
            "--num-model-seeds",
            str(args.num_model_seeds),
            "--tolerance",
            str(args.comparison_tolerance),
        ]
        if args.inference_ckpt_path is not None:
            comparison_cmd += ["--inference-ckpt-path", str(args.inference_ckpt_path)]
        if args.inference_ckpt_name is not None:
            comparison_cmd += ["--inference-ckpt-name", args.inference_ckpt_name]
        if args.use_msa_server:
            comparison_cmd.append("--use-msa-server")
        if args.use_templates:
            comparison_cmd.append("--use-templates")

        leucine_cmd = [
            args.python_bin,
            "scripts/dev/run_overnight_screening.py",
            "--base-query-json",
            str(args.leucine_query_json),
            "--query-id",
            args.leucine_query_id,
            "--mutations-csv",
            str(args.mutations_csv),
            "--output-root",
            str(screening_out),
            "--runner-yaml",
            str(args.runner_yaml),
            "--num-diffusion-samples",
            str(args.num_diffusion_samples),
            "--num-model-seeds",
            str(args.num_model_seeds),
            "--num-cpu-workers",
            str(args.num_cpu_workers),
            "--max-inflight-queries",
            str(args.max_inflight_queries),
            "--min-free-disk-gb",
            str(args.min_free_disk_gb),
        ]
        if args.inference_ckpt_path is not None:
            leucine_cmd += ["--inference-ckpt-path", str(args.inference_ckpt_path)]
        if args.inference_ckpt_name is not None:
            leucine_cmd += ["--inference-ckpt-name", args.inference_ckpt_name]
        if args.include_wt:
            leucine_cmd.append("--include-wt")
        if args.use_msa_server:
            leucine_cmd.append("--use-msa-server")
        if args.use_templates:
            leucine_cmd.append("--use-templates")

        monitor.record_stage("compare_started")
        stream_command(
            comparison_cmd,
            cwd=repo_root,
            log_path=log_path,
            step_name="compare",
        )
        monitor.record_stage("compare_finished")
        monitor.record_stage("leucine_started")
        stream_command(
            leucine_cmd,
            cwd=repo_root,
            log_path=log_path,
            step_name="leucine",
        )
        monitor.record_stage("leucine_finished")
        log_line("[suite] ALL_DONE", log_path)
        monitor.record_stage("suite_finished")
    except Exception as exc:
        monitor.record_stage("suite_failed", str(exc))
        raise
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
