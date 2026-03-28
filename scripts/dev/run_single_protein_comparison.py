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

"""Runtime comparison for single-protein predict vs mutation-screening."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_cmd(cmd: list[str], cwd: Path) -> float:
    print(f"[run] {' '.join(cmd)}")
    start = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, check=True)
    return time.perf_counter() - start


def find_first_aggregated_confidence_json(output_dir: Path) -> Path:
    matches = sorted(output_dir.rglob("*_confidences_aggregated.json"))
    if not matches:
        raise FileNotFoundError(
            f"No aggregated confidence json found under {output_dir}"
        )
    return matches[0]


def build_screening_job(
    query_id: str,
    query_payload: dict[str, Any],
    output_root: Path,
    runner_yaml: Path,
    num_diffusion_samples: int,
    num_model_seeds: int,
    use_msa_server: bool,
    use_templates: bool,
    inference_ckpt_path: str | None,
    inference_ckpt_name: str | None,
) -> dict[str, Any]:
    job = {
        "base_query": query_payload,
        "mutations": [],
        "output_dir": str(output_root / "screening"),
        "cache_dir": str(output_root / "cache"),
        "query_prefix": query_id,
        "include_wt": True,
        "run_baseline_first": True,
        "msa_policy": "reuse_precomputed",
        "template_policy": "reuse_precomputed",
        "output_policy": "metrics_only",
        "resume": True,
        "num_cpu_workers": 1,
        "max_inflight_queries": 1,
        "num_diffusion_samples": num_diffusion_samples,
        "num_model_seeds": num_model_seeds,
        "runner_yaml": str(runner_yaml),
        "use_msa_server": use_msa_server,
        "use_templates": use_templates,
    }
    if inference_ckpt_path is not None:
        job["inference_ckpt_path"] = inference_ckpt_path
    if inference_ckpt_name is not None:
        job["inference_ckpt_name"] = inference_ckpt_name
    return job


def compare_metric_dicts(
    baseline: dict[str, Any], screening: dict[str, Any], tolerance: float
) -> dict[str, Any]:
    numeric_keys = []
    for key in sorted(set(baseline) & set(screening)):
        if isinstance(baseline[key], (int, float)) and isinstance(
            screening[key], (int, float)
        ):
            numeric_keys.append(key)

    comparison = {
        "numeric_keys": numeric_keys,
        "differences": {},
        "within_tolerance": True,
    }
    for key in numeric_keys:
        diff = abs(float(baseline[key]) - float(screening[key]))
        comparison["differences"][key] = diff
        if diff > tolerance:
            comparison["within_tolerance"] = False
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-json",
        type=Path,
        default=Path("examples/example_inference_inputs/query_ubiquitin.json"),
    )
    parser.add_argument(
        "--runner-yaml",
        type=Path,
        default=Path("examples/example_runner_yamls/low_mem.yml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runtime_smoke/single_protein_compare"),
    )
    parser.add_argument("--query-id", type=str, default=None)
    parser.add_argument("--num-diffusion-samples", type=int, default=1)
    parser.add_argument("--num-model-seeds", type=int, default=1)
    parser.add_argument("--use-msa-server", action="store_true")
    parser.add_argument("--use-templates", action="store_true")
    parser.add_argument("--inference-ckpt-path", type=str, default=None)
    parser.add_argument("--inference-ckpt-name", type=str, default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--tolerance", type=float, default=0.1)
    args = parser.parse_args()

    repo_root = Path.cwd()
    query_data = load_json(args.query_json)
    queries = query_data["queries"]
    query_id = args.query_id or next(iter(queries))
    query_payload = queries[query_id]

    output_root = args.output_root.resolve()
    baseline_dir = output_root / "baseline_predict"
    screening_job_path = output_root / "screening_job.json"

    screening_job = build_screening_job(
        query_id=query_id,
        query_payload=query_payload,
        output_root=output_root,
        runner_yaml=args.runner_yaml.resolve(),
        num_diffusion_samples=args.num_diffusion_samples,
        num_model_seeds=args.num_model_seeds,
        use_msa_server=args.use_msa_server,
        use_templates=args.use_templates,
        inference_ckpt_path=args.inference_ckpt_path,
        inference_ckpt_name=args.inference_ckpt_name,
    )
    dump_json(screening_job_path, screening_job)

    baseline_cmd = [
        args.python_bin,
        "-m",
        "openfold3.run_openfold",
        "predict",
        "--query_json",
        str(args.query_json.resolve()),
        "--runner_yaml",
        str(args.runner_yaml.resolve()),
        "--output_dir",
        str(baseline_dir),
        "--use_msa_server",
        str(args.use_msa_server).lower(),
        "--use_templates",
        str(args.use_templates).lower(),
        "--num_diffusion_samples",
        str(args.num_diffusion_samples),
        "--num_model_seeds",
        str(args.num_model_seeds),
    ]
    if args.inference_ckpt_path is not None:
        baseline_cmd += ["--inference_ckpt_path", args.inference_ckpt_path]
    if args.inference_ckpt_name is not None:
        baseline_cmd += ["--inference_ckpt_name", args.inference_ckpt_name]

    screening_cmd = [
        args.python_bin,
        "-m",
        "openfold3.run_openfold",
        "screen-mutations",
        "--screening_job_json",
        str(screening_job_path),
    ]

    baseline_elapsed_seconds = run_cmd(baseline_cmd, cwd=repo_root)
    screening_elapsed_seconds = run_cmd(screening_cmd, cwd=repo_root)

    baseline_agg_path = find_first_aggregated_confidence_json(baseline_dir)
    baseline_metrics = load_json(baseline_agg_path)

    screening_results_path = output_root / "screening" / "results.jsonl"
    screening_rows = [
        json.loads(line)
        for line in screening_results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not screening_rows:
        raise RuntimeError("No rows written by screen-mutations")
    screening_best = screening_rows[0]

    comparison = compare_metric_dicts(
        baseline=baseline_metrics,
        screening=screening_best,
        tolerance=args.tolerance,
    )
    timing = {
        "baseline_elapsed_seconds": baseline_elapsed_seconds,
        "screening_elapsed_seconds": screening_elapsed_seconds,
        "delta_seconds": screening_elapsed_seconds - baseline_elapsed_seconds,
        "speedup_ratio": (
            baseline_elapsed_seconds / screening_elapsed_seconds
            if screening_elapsed_seconds > 0
            else None
        ),
        "screening_internal_total_seconds": screening_best.get("total_seconds"),
        "screening_internal_cpu_prep_seconds": screening_best.get(
            "cpu_prep_seconds"
        ),
        "screening_internal_gpu_inference_seconds": screening_best.get(
            "gpu_inference_seconds"
        ),
    }
    summary = {
        "query_id": query_id,
        "baseline_aggregated_confidence_path": str(baseline_agg_path),
        "screening_results_path": str(screening_results_path),
        "baseline_metrics": baseline_metrics,
        "screening_best_row": screening_best,
        "comparison": comparison,
        "timing": timing,
    }
    dump_json(output_root / "comparison_summary.json", summary)

    print(json.dumps(summary["comparison"], indent=2))
    print(json.dumps(summary["timing"], indent=2))
if __name__ == "__main__":
    main()
