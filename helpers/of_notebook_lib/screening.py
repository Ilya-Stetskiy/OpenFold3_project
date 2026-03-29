from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis import rank_mutations, summarize_mutation_batch
from .config import RuntimeConfig
from .query_builders import (
    CANONICAL_AA,
    build_mutation_scan_payload,
    normalize_molecules,
)
from .runner import RunResult, _slug_timestamp, run_prediction


@dataclass(slots=True)
class ScreeningBatchResult:
    experiment_name: str
    run_dir: Path
    job_json_path: Path
    output_dir: Path
    cache_dir: Path
    log_path: Path
    summary_path: Path
    rows_df: pd.DataFrame
    mutation_summary: pd.DataFrame
    mutation_ranking: pd.DataFrame
    elapsed_seconds: float
    return_code: int


@dataclass(slots=True)
class BatchApproachComparison:
    experiment_name: str
    run_dir: Path
    summary_path: Path
    predict_result: RunResult
    screening_result: ScreeningBatchResult
    comparison: dict[str, Any]


@dataclass(slots=True)
class ServerEndToEndResult:
    experiment_name: str
    run_dir: Path
    summary_path: Path
    single_result: RunResult
    screening_result: ScreeningBatchResult | None
    gpu_probe: dict[str, Any]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_timed_cmd(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path | None,
    log_path: Path,
) -> tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
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
        return process.wait(), time.perf_counter() - started


def _resolve_openfold_repo_dir(
    runtime: RuntimeConfig,
    repo_dir: str | Path | None = None,
) -> Path | None:
    resolved = Path(repo_dir) if repo_dir is not None else runtime.openfold_repo_dir
    resolved = resolved.expanduser().resolve()
    expected = resolved / "openfold3" / "run_openfold.py"
    if expected.exists():
        return resolved
    if repo_dir is None:
        runner = runtime.openfold_runner
        if runner.exists():
            return runtime.project_dir.resolve()
        return None
    if not expected.exists():
        raise FileNotFoundError(
            f"OPENFOLD_REPO_DIR does not look like an openfold-3 checkout: {resolved}"
        )
    return None


def _mutation_specs_for_position(
    molecules: list[dict],
    *,
    mutation_chain_id: str,
    position_1based: int,
    amino_acids: str | list[str],
    include_wt: bool,
) -> tuple[list[dict[str, Any]], str]:
    normalized = normalize_molecules(molecules)
    residues = list(amino_acids) if isinstance(amino_acids, str) else list(amino_acids)
    residues = [str(residue).upper() for residue in residues]
    invalid = sorted({residue for residue in residues if residue not in CANONICAL_AA})
    if invalid:
        raise ValueError(f"Unsupported residues for screening: {invalid}")

    wt_residue: str | None = None
    for molecule in normalized:
        if mutation_chain_id in molecule["chain_ids"]:
            sequence = molecule.get("sequence")
            if not sequence:
                raise ValueError(f"Chain '{mutation_chain_id}' does not have a sequence field")
            if position_1based < 1 or position_1based > len(sequence):
                raise ValueError(
                    f"Mutation position {position_1based} is outside sequence length {len(sequence)}"
                )
            wt_residue = sequence[position_1based - 1]
            break

    if wt_residue is None:
        raise ValueError(f"Chain '{mutation_chain_id}' was not found in molecules")

    mutation_specs = []
    for residue in residues:
        if include_wt or residue != wt_residue:
            mutation_specs.append(
                {
                    "chain_id": mutation_chain_id,
                    "position_1based": position_1based,
                    "from_residue": wt_residue,
                    "to_residue": residue,
                }
            )
    return mutation_specs, wt_residue


def _screening_rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).copy()
    if "query_id" in df.columns:
        df["query_name"] = df["query_id"]
    if "mutation_id" in df.columns:
        df["mutation_label"] = df["mutation_id"]
    if "query_name" not in df.columns:
        df["query_name"] = None
    if "mutation_label" not in df.columns:
        df["mutation_label"] = df["query_name"]
    if "sample_name" not in df.columns:
        df["sample_name"] = df["query_name"]
    return df


def run_screened_mutation_scan(
    runtime: RuntimeConfig,
    experiment_name: str,
    molecules: list[dict],
    *,
    mutation_chain_id: str,
    position_1based: int,
    amino_acids: str | list[str],
    include_wt: bool = True,
    use_templates: bool = True,
    use_msa_server: bool = True,
    num_diffusion_samples: int = 1,
    num_model_seeds: int = 1,
    runner_yaml: str | Path | None = None,
    inference_ckpt_path: str | Path | None = None,
    inference_ckpt_name: str | None = None,
    repo_dir: str | Path | None = None,
    keep_query_outputs: bool = False,
    num_cpu_workers: int = 1,
    max_inflight_queries: int = 1,
    cache_query_results: bool = True,
    subprocess_batch_size: int = 1,
) -> ScreeningBatchResult:
    repo_root = _resolve_openfold_repo_dir(runtime, repo_dir)
    run_dir = runtime.results_dir / _slug_timestamp(f"{experiment_name}_screening")
    output_dir = run_dir / "screening"
    cache_dir = run_dir / "cache"
    log_path = run_dir / "screen_mutations.log"
    summary_path = run_dir / "screening_summary.json"
    job_json_path = run_dir / "screening_job.json"

    mutation_specs, _ = _mutation_specs_for_position(
        molecules,
        mutation_chain_id=mutation_chain_id,
        position_1based=position_1based,
        amino_acids=amino_acids,
        include_wt=False,
    )
    normalized = normalize_molecules(molecules)
    job = {
        "base_query": {"chains": normalized},
        "mutations": mutation_specs,
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "query_prefix": experiment_name,
        "include_wt": include_wt,
        "run_baseline_first": True,
        "msa_policy": "reuse_precomputed",
        "template_policy": "reuse_precomputed",
        "output_policy": "metrics_only",
        "resume": True,
        "cache_query_results": cache_query_results,
        "num_cpu_workers": num_cpu_workers,
        "max_inflight_queries": max_inflight_queries,
        "subprocess_batch_size": max(1, subprocess_batch_size),
        "num_diffusion_samples": num_diffusion_samples,
        "num_model_seeds": num_model_seeds,
        "runner_yaml": str(Path(runner_yaml)) if runner_yaml else None,
        "inference_ckpt_path": str(Path(inference_ckpt_path))
        if inference_ckpt_path
        else None,
        "inference_ckpt_name": inference_ckpt_name,
        "use_msa_server": use_msa_server,
        "use_templates": use_templates,
        "min_free_disk_gb": 1.0,
        "cleanup_query_outputs": not keep_query_outputs,
        "log_file": str(run_dir / "screening_runtime.log"),
    }
    _write_json(job_json_path, job)

    cmd = [
        str(runtime.openfold_python),
        "-m",
        "openfold3.run_openfold",
        "screen-mutations",
        "--screening_job_json",
        str(job_json_path),
    ]
    env = runtime.build_env()
    return_code, elapsed_seconds = _run_timed_cmd(
        cmd,
        env=env,
        cwd=repo_root,
        log_path=log_path,
    )
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

    results_path = output_dir / "results.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows_df = _screening_rows_to_dataframe(rows)
    mutation_summary = summarize_mutation_batch(rows_df)
    mutation_ranking = rank_mutations(mutation_summary, top_n=len(mutation_summary))

    summary_payload = {
        "experiment_name": experiment_name,
        "run_dir": str(run_dir),
        "job_json_path": str(job_json_path),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "log_path": str(log_path),
        "return_code": return_code,
        "elapsed_seconds": elapsed_seconds,
        "cache_query_results": cache_query_results,
        "subprocess_batch_size": max(1, subprocess_batch_size),
        "row_count": int(len(rows_df)),
        "mutation_summary_count": int(len(mutation_summary)),
        "query_result_cache_hits": int(
            rows_df.get("query_result_cache_hit", pd.Series(dtype=bool)).sum()
        )
        if not rows_df.empty
        else 0,
    }
    _write_json(summary_path, summary_payload)

    return ScreeningBatchResult(
        experiment_name=experiment_name,
        run_dir=run_dir,
        job_json_path=job_json_path,
        output_dir=output_dir,
        cache_dir=cache_dir,
        log_path=log_path,
        summary_path=summary_path,
        rows_df=rows_df,
        mutation_summary=mutation_summary,
        mutation_ranking=mutation_ranking,
        elapsed_seconds=elapsed_seconds,
        return_code=return_code,
    )


def compare_mutation_batch_approaches(
    runtime: RuntimeConfig,
    experiment_name: str,
    molecules: list[dict],
    *,
    mutation_chain_id: str,
    position_1based: int,
    amino_acids: str | list[str],
    include_wt: bool = True,
    use_templates: bool = True,
    use_msa_server: bool = True,
    num_diffusion_samples: int = 1,
    num_model_seeds: int = 1,
    runner_yaml: str | Path | None = None,
    inference_ckpt_path: str | Path | None = None,
    inference_ckpt_name: str | None = None,
    repo_dir: str | Path | None = None,
    cache_query_results: bool = True,
    subprocess_batch_size: int = 1,
) -> BatchApproachComparison:
    payload = build_mutation_scan_payload(
        query_prefix=experiment_name,
        molecules=molecules,
        mutation_chain_id=mutation_chain_id,
        position_1based=position_1based,
        amino_acids=amino_acids,
        include_wt=include_wt,
    )
    predict_result = run_prediction(
        runtime,
        payload,
        experiment_name=f"{experiment_name}_predict_batch",
        use_templates=use_templates,
        use_msa_server=use_msa_server,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        runner_yaml=runner_yaml,
        inference_ckpt_path=inference_ckpt_path,
        inference_ckpt_name=inference_ckpt_name,
    )
    screening_result = run_screened_mutation_scan(
        runtime=runtime,
        experiment_name=f"{experiment_name}_screen",
        molecules=molecules,
        mutation_chain_id=mutation_chain_id,
        position_1based=position_1based,
        amino_acids=amino_acids,
        include_wt=include_wt,
        use_templates=use_templates,
        use_msa_server=use_msa_server,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        runner_yaml=runner_yaml,
        inference_ckpt_path=inference_ckpt_path,
        inference_ckpt_name=inference_ckpt_name,
        repo_dir=repo_dir,
        cache_query_results=cache_query_results,
        subprocess_batch_size=subprocess_batch_size,
    )

    run_dir = runtime.results_dir / _slug_timestamp(f"{experiment_name}_compare")
    summary_path = run_dir / "comparison_summary.json"
    screening_internal_total = float(screening_result.rows_df["total_seconds"].sum()) if not screening_result.rows_df.empty else 0.0
    comparison = {
        "experiment_name": experiment_name,
        "predict_batch_elapsed_seconds": predict_result.elapsed_seconds,
        "screen_mutations_elapsed_seconds": screening_result.elapsed_seconds,
        "time_saved_seconds": predict_result.elapsed_seconds - screening_result.elapsed_seconds,
        "speedup_ratio": (
            predict_result.elapsed_seconds / screening_result.elapsed_seconds
            if screening_result.elapsed_seconds > 0
            else None
        ),
        "screen_mutations_internal_total_seconds": screening_internal_total,
        "cache_query_results": cache_query_results,
        "subprocess_batch_size": max(1, subprocess_batch_size),
        "screen_mutations_query_result_cache_hits": int(
            screening_result.rows_df.get("query_result_cache_hit", pd.Series(dtype=bool)).sum()
        )
        if not screening_result.rows_df.empty
        else 0,
        "predict_batch_query_count": int(len(payload["queries"])),
        "screen_mutations_row_count": int(len(screening_result.rows_df)),
    }
    _write_json(summary_path, comparison)
    return BatchApproachComparison(
        experiment_name=experiment_name,
        run_dir=run_dir,
        summary_path=summary_path,
        predict_result=predict_result,
        screening_result=screening_result,
        comparison=comparison,
    )


def _probe_gpu() -> dict[str, Any]:
    cmd = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "command": " ".join(cmd),
        "stdout": completed.stdout.strip(),
    }


def run_server_end_to_end_smoke(
    runtime: RuntimeConfig,
    experiment_name: str,
    molecules: list[dict],
    *,
    mutation_chain_id: str,
    position_1based: int,
    amino_acids: str | list[str],
    include_wt: bool = True,
    use_templates: bool = False,
    use_msa_server: bool = True,
    num_diffusion_samples: int = 1,
    num_model_seeds: int = 1,
    runner_yaml: str | Path | None = None,
    inference_ckpt_path: str | Path | None = None,
    inference_ckpt_name: str | None = None,
    repo_dir: str | Path | None = None,
    run_screening: bool = True,
    cache_query_results: bool = True,
    subprocess_batch_size: int = 1,
) -> ServerEndToEndResult:
    gpu_probe = _probe_gpu()
    single_result = run_prediction(
        runtime=runtime,
        payload={"queries": {experiment_name: {"chains": normalize_molecules(molecules)}}},
        experiment_name=f"{experiment_name}_single",
        use_templates=use_templates,
        use_msa_server=use_msa_server,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        runner_yaml=runner_yaml,
        inference_ckpt_path=inference_ckpt_path,
        inference_ckpt_name=inference_ckpt_name,
    )

    screening_result = None
    if run_screening:
        screening_result = run_screened_mutation_scan(
            runtime=runtime,
            experiment_name=f"{experiment_name}_screening",
            molecules=molecules,
            mutation_chain_id=mutation_chain_id,
            position_1based=position_1based,
            amino_acids=amino_acids,
            include_wt=include_wt,
            use_templates=use_templates,
            use_msa_server=use_msa_server,
            num_diffusion_samples=num_diffusion_samples,
            num_model_seeds=num_model_seeds,
            runner_yaml=runner_yaml,
            inference_ckpt_path=inference_ckpt_path,
            inference_ckpt_name=inference_ckpt_name,
            repo_dir=repo_dir,
            cache_query_results=cache_query_results,
            subprocess_batch_size=subprocess_batch_size,
        )

    run_dir = runtime.results_dir / _slug_timestamp(f"{experiment_name}_server_e2e")
    summary_path = run_dir / "server_end_to_end_summary.json"
    summary = {
        "experiment_name": experiment_name,
        "gpu_probe": gpu_probe,
        "single_run_dir": str(single_result.run_dir),
        "single_elapsed_seconds": single_result.elapsed_seconds,
        "single_output_dir": str(single_result.output_dir),
        "screening_run_dir": str(screening_result.run_dir) if screening_result else None,
        "screening_elapsed_seconds": (
            screening_result.elapsed_seconds if screening_result else None
        ),
        "screening_output_dir": (
            str(screening_result.output_dir) if screening_result else None
        ),
        "cache_query_results": cache_query_results,
        "subprocess_batch_size": max(1, subprocess_batch_size),
    }
    _write_json(summary_path, summary)
    return ServerEndToEndResult(
        experiment_name=experiment_name,
        run_dir=run_dir,
        summary_path=summary_path,
        single_result=single_result,
        screening_result=screening_result,
        gpu_probe=gpu_probe,
    )
