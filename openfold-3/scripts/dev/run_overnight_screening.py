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

"""Prepare and run a long mutation-screening batch with disk-safe logging."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openfold3.mutation_runner import (
    MutationScreeningRunner,
    MutationSpec,
    ScreeningJob,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import Query


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def resolve_query_payload(
    query_data: dict[str, Any], query_id: str | None
) -> tuple[str, dict[str, Any]]:
    queries = query_data.get("queries")
    if not isinstance(queries, dict) or not queries:
        raise ValueError("Base query JSON must contain at least one query in 'queries'")

    resolved_query_id = query_id or next(iter(queries))
    if resolved_query_id not in queries:
        raise ValueError(
            f"Query id '{resolved_query_id}' was not found. "
            f"Available query ids: {sorted(queries)}"
        )
    return resolved_query_id, queries[resolved_query_id]


def load_mutations_from_csv(
    path: Path, max_mutations: int | None = None
) -> list[MutationSpec]:
    mutations: list[MutationSpec] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        required = {"chain_id", "position_1based", "from_residue", "to_residue"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Mutation CSV {path} is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            if not any(str(v).strip() for v in row.values()):
                continue
            mutations.append(
                MutationSpec(
                    chain_id=row["chain_id"].strip(),
                    position_1based=int(row["position_1based"]),
                    from_residue=row["from_residue"].strip(),
                    to_residue=row["to_residue"].strip(),
                )
            )
            if max_mutations is not None and len(mutations) >= max_mutations:
                break
    return mutations


def build_screening_job(args) -> ScreeningJob:
    query_data = load_json(args.base_query_json)
    query_id, query_payload = resolve_query_payload(query_data, args.query_id)
    base_query = Query.model_validate(query_payload)
    mutations = load_mutations_from_csv(
        args.mutations_csv, max_mutations=args.max_mutations
    )
    if not mutations:
        raise ValueError("Mutation CSV did not produce any mutations")

    output_root = args.output_root.resolve()
    log_file = output_root / "logs" / "overnight_screening.log"
    cache_dir = output_root / "cache"
    run_dir = output_root / "screening"

    job = ScreeningJob(
        base_query=base_query,
        mutations=mutations,
        output_dir=run_dir,
        cache_dir=cache_dir,
        query_prefix=args.query_prefix or query_id,
        include_wt=args.include_wt,
        run_baseline_first=True,
        msa_policy="reuse_precomputed",
        template_policy="reuse_precomputed",
        output_policy="metrics_only",
        resume=not args.no_resume,
        cache_query_results=not args.no_query_result_cache,
        num_cpu_workers=args.num_cpu_workers,
        max_inflight_queries=args.max_inflight_queries,
        subprocess_batch_size=args.subprocess_batch_size,
        num_diffusion_samples=args.num_diffusion_samples,
        num_model_seeds=args.num_model_seeds,
        runner_yaml=args.runner_yaml.resolve() if args.runner_yaml else None,
        inference_ckpt_path=(
            args.inference_ckpt_path.resolve()
            if args.inference_ckpt_path is not None
            else None
        ),
        inference_ckpt_name=args.inference_ckpt_name,
        use_msa_server=args.use_msa_server,
        use_templates=args.use_templates,
        min_free_disk_gb=args.min_free_disk_gb,
        cleanup_query_outputs=not args.keep_query_outputs,
        log_file=log_file,
    )

    job_json = {
        "base_query": base_query.model_dump(mode="json"),
        "mutations": [
            {
                "chain_id": m.chain_id,
                "position_1based": m.position_1based,
                "from_residue": m.from_residue,
                "to_residue": m.to_residue,
            }
            for m in mutations
        ],
        "output_dir": str(run_dir),
        "cache_dir": str(cache_dir),
        "query_prefix": job.query_prefix,
        "include_wt": job.include_wt,
        "run_baseline_first": job.run_baseline_first,
        "msa_policy": job.msa_policy,
        "template_policy": job.template_policy,
        "output_policy": job.output_policy,
        "resume": job.resume,
        "cache_query_results": job.cache_query_results,
        "num_cpu_workers": job.num_cpu_workers,
        "max_inflight_queries": job.max_inflight_queries,
        "subprocess_batch_size": job.subprocess_batch_size,
        "num_diffusion_samples": job.num_diffusion_samples,
        "num_model_seeds": job.num_model_seeds,
        "runner_yaml": str(job.runner_yaml) if job.runner_yaml else None,
        "inference_ckpt_path": (
            str(job.inference_ckpt_path) if job.inference_ckpt_path else None
        ),
        "inference_ckpt_name": job.inference_ckpt_name,
        "use_msa_server": job.use_msa_server,
        "use_templates": job.use_templates,
        "min_free_disk_gb": job.min_free_disk_gb,
        "cleanup_query_outputs": job.cleanup_query_outputs,
        "log_file": str(job.log_file) if job.log_file else None,
    }
    dump_json(output_root / "screening_job.generated.json", job_json)
    dump_json(
        output_root / "launch_manifest.json",
        {
            "base_query_json": str(args.base_query_json.resolve()),
            "mutations_csv": str(args.mutations_csv.resolve()),
            "mutation_count": len(mutations),
            "max_mutations": args.max_mutations,
            "output_root": str(output_root),
            "log_file": str(log_file),
            "query_prefix": job.query_prefix,
            "num_diffusion_samples": job.num_diffusion_samples,
            "num_model_seeds": job.num_model_seeds,
            "cache_query_results": job.cache_query_results,
            "subprocess_batch_size": job.subprocess_batch_size,
            "min_free_disk_gb": job.min_free_disk_gb,
            "cleanup_query_outputs": job.cleanup_query_outputs,
        },
    )
    return job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-query-json", type=Path, required=True)
    parser.add_argument("--mutations-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runner-yaml", type=Path, required=False)
    parser.add_argument("--query-id", type=str, default=None)
    parser.add_argument("--query-prefix", type=str, default=None)
    parser.add_argument("--num-diffusion-samples", type=int, default=1)
    parser.add_argument("--num-model-seeds", type=int, default=1)
    parser.add_argument("--max-mutations", type=int, default=None)
    parser.add_argument(
        "--num-cpu-workers", type=int, default=(__import__("os").cpu_count() or 1)
    )
    parser.add_argument("--max-inflight-queries", type=int, default=2)
    parser.add_argument("--subprocess-batch-size", type=int, default=1)
    parser.add_argument("--min-free-disk-gb", type=float, default=1.0)
    parser.add_argument("--inference-ckpt-path", type=Path, default=None)
    parser.add_argument("--inference-ckpt-name", type=str, default=None)
    parser.add_argument("--use-msa-server", action="store_true")
    parser.add_argument("--use-templates", action="store_true")
    parser.add_argument("--include-wt", action="store_true")
    parser.add_argument("--keep-query-outputs", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-query-result-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    job = build_screening_job(args)
    MutationScreeningRunner().run(job)


if __name__ == "__main__":
    main()

