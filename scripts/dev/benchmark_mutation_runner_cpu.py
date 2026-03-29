#!/usr/bin/env python
"""CPU-only benchmark harness for mutation_runner orchestration and cache behavior."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

from openfold3.mutation_runner import (
    MutationScreeningRunner,
    MutationSpec,
    ScreeningJob,
    ScreeningResultRow,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import Query


class BenchmarkBackend:
    def run(self, prepared_job):
        payload = json.loads(prepared_job.payload_path.read_text(encoding="utf-8"))
        query = payload["queries"][prepared_job.query_id]
        sequence_count = sum(1 for chain in query["chains"] if chain.get("sequence"))
        return ScreeningResultRow(
            mutation_id=prepared_job.mutation_id,
            query_id=prepared_job.query_id,
            query_hash=prepared_job.query_hash,
            sample_index=1,
            seed=1,
            sample_ranking_score=float(sequence_count),
            iptm=0.0,
            ptm=0.0,
            avg_plddt=0.0,
            gpde=0.0,
            has_clash=0.0,
            cache_hit=prepared_job.cache_hit,
            sequence_cache_hits=prepared_job.sequence_cache_hits,
            query_result_cache_hit=False,
            cpu_prep_seconds=prepared_job.cpu_prep_seconds,
            gpu_inference_seconds=0.0,
            total_seconds=prepared_job.cpu_prep_seconds,
            output_dir=str(prepared_job.output_dir),
            aggregated_confidence_path=None,
            mutation_spec=None
            if prepared_job.mutation_spec is None
            else {
                "chain_id": prepared_job.mutation_spec.chain_id,
                "position_1based": prepared_job.mutation_spec.position_1based,
                "from_residue": prepared_job.mutation_spec.from_residue,
                "to_residue": prepared_job.mutation_spec.to_residue,
            },
        )


def build_job(
    root: Path,
    num_mutations: int,
    include_wt: bool,
    *,
    cache_query_results: bool = True,
    subprocess_batch_size: int = 1,
) -> ScreeningJob:
    invariant_dir = root / "alignments_B"
    invariant_dir.mkdir(parents=True, exist_ok=True)
    (invariant_dir / "main.a3m").write_text(">query\nBBBB\n", encoding="utf-8")

    base_query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": "ACDEFGHIKLMNPQRSTVWY",
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": ["B"],
                    "sequence": "BBBB",
                    "main_msa_file_paths": [str(invariant_dir)],
                },
            ]
        }
    )
    targets = "ACDEFGHIKLMNPQRSTVWY"
    mutations = [
        MutationSpec(
            chain_id="A",
            position_1based=1 + (idx % 5),
            from_residue=targets[idx % 5],
            to_residue=targets[(idx + 7) % len(targets)],
        )
        for idx in range(num_mutations)
    ]
    return ScreeningJob(
        base_query=base_query,
        mutations=mutations,
        output_dir=root / "screening",
        cache_dir=root / "cache",
        include_wt=include_wt,
        cache_query_results=cache_query_results,
        run_baseline_first=True,
        output_policy="metrics_only",
        num_cpu_workers=1,
        max_inflight_queries=1,
        subprocess_batch_size=subprocess_batch_size,
    )


def summarize(name: str, values: list[float]) -> dict[str, float]:
    return {
        f"{name}_min": min(values),
        f"{name}_median": statistics.median(values),
        f"{name}_mean": statistics.mean(values),
        f"{name}_max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runtime_smoke/mutation_runner_cpu_bench"),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num-mutations", type=int, default=20)
    parser.add_argument("--include-wt", action="store_true")
    parser.add_argument("--subprocess-batch-size", type=int, default=1)
    parser.add_argument("--no-query-result-cache", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    backend = BenchmarkBackend()

    cold_wall = []
    cold_cpu = []
    for idx in range(args.repeats):
        run_root = output_root / f"cold_{idx}"
        shutil.rmtree(run_root, ignore_errors=True)
        job = build_job(
            run_root,
            args.num_mutations,
            args.include_wt,
            cache_query_results=not args.no_query_result_cache,
            subprocess_batch_size=args.subprocess_batch_size,
        )
        started = time.perf_counter()
        rows = MutationScreeningRunner(backend=backend).run(job)
        cold_wall.append(time.perf_counter() - started)
        cold_cpu.append(sum(row.cpu_prep_seconds for row in rows))

    warm_root = output_root / "warm"
    shutil.rmtree(warm_root, ignore_errors=True)
    warm_wall = []
    warm_cpu = []
    warm_cached_rows = []
    runner = MutationScreeningRunner(backend=backend)
    for _idx in range(args.repeats):
        job = build_job(
            warm_root,
            args.num_mutations,
            args.include_wt,
            cache_query_results=not args.no_query_result_cache,
            subprocess_batch_size=args.subprocess_batch_size,
        )
        started = time.perf_counter()
        rows = runner.run(job)
        warm_wall.append(time.perf_counter() - started)
        warm_cpu.append(sum(row.cpu_prep_seconds for row in rows))
        warm_cached_rows.append(sum(row.query_result_cache_hit for row in rows))

    summary = {
        "repeats": args.repeats,
        "num_mutations": args.num_mutations,
        "include_wt": args.include_wt,
        "cache_query_results": not args.no_query_result_cache,
        "subprocess_batch_size": args.subprocess_batch_size,
        **summarize("cold_wall_seconds", cold_wall),
        **summarize("cold_cpu_prep_seconds", cold_cpu),
        **summarize("warm_wall_seconds", warm_wall),
        **summarize("warm_cpu_prep_seconds", warm_cpu),
        "warm_cached_rows_per_run": warm_cached_rows,
    }
    summary_path = output_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
