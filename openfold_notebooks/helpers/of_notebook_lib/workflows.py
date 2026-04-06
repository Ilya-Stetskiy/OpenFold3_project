from __future__ import annotations

import pandas as pd

from .analysis import rank_mutations, summarize_mutation_batch
from .config import RuntimeConfig
from .query_builders import (
    apply_mutation_to_molecules,
    build_mutation_scan_payload,
    build_single_query_payload,
    normalize_molecules,
)
from .runner import RunResult, run_prediction
from .screening import (
    BatchApproachComparison,
    ScreeningBatchResult,
    ServerEndToEndResult,
    compare_mutation_batch_approaches,
    run_screened_mutation_scan,
    run_server_end_to_end_smoke,
)


def run_single_case(
    runtime: RuntimeConfig,
    experiment_name: str,
    molecules: list[dict],
    *,
    mutation: dict | None = None,
    use_templates: bool = True,
    use_msa_server: bool = True,
    num_diffusion_samples: int = 1,
    num_model_seeds: int = 1,
    runner_yaml: str | None = None,
    inference_ckpt_path: str | None = None,
    inference_ckpt_name: str | None = None,
    enable_monitoring: bool = False,
) -> RunResult:
    work_molecules = normalize_molecules(molecules)

    if mutation and mutation.get("enabled"):
        work_molecules = apply_mutation_to_molecules(
            work_molecules,
            chain_id=mutation["chain_id"],
            position_1based=int(mutation["position_1based"]),
            new_residue=mutation["new_residue"],
        )

    payload = build_single_query_payload(experiment_name, work_molecules)
    return run_prediction(
        runtime,
        payload,
        experiment_name=experiment_name,
        use_templates=use_templates,
        use_msa_server=use_msa_server,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        runner_yaml=runner_yaml,
        inference_ckpt_path=inference_ckpt_path,
        inference_ckpt_name=inference_ckpt_name,
        enable_monitoring=enable_monitoring,
    )


def run_mutation_scan(
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
    runner_yaml: str | None = None,
    inference_ckpt_path: str | None = None,
    inference_ckpt_name: str | None = None,
    enable_monitoring: bool = True,
) -> tuple[RunResult, pd.DataFrame, pd.DataFrame]:
    payload = build_mutation_scan_payload(
        query_prefix=experiment_name,
        molecules=molecules,
        mutation_chain_id=mutation_chain_id,
        position_1based=position_1based,
        amino_acids=amino_acids,
        include_wt=include_wt,
    )

    result = run_prediction(
        runtime,
        payload,
        experiment_name=experiment_name,
        use_templates=use_templates,
        use_msa_server=use_msa_server,
        num_diffusion_samples=num_diffusion_samples,
        num_model_seeds=num_model_seeds,
        runner_yaml=runner_yaml,
        inference_ckpt_path=inference_ckpt_path,
        inference_ckpt_name=inference_ckpt_name,
        enable_monitoring=enable_monitoring,
    )

    mutation_summary = summarize_mutation_batch(result.samples_df)
    mutation_ranking = rank_mutations(mutation_summary, top_n=len(mutation_summary))
    return result, mutation_summary, mutation_ranking


def run_screened_mutation_case(
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
    runner_yaml: str | None = None,
    inference_ckpt_path: str | None = None,
    inference_ckpt_name: str | None = None,
    repo_dir: str | None = None,
    cache_query_results: bool = True,
    subprocess_batch_size: int = 1,
    dispatch_partial_batches: bool = False,
    batch_gather_timeout_seconds: float | None = None,
    output_policy: str = "metrics_only",
    keep_query_outputs: bool | None = None,
) -> ScreeningBatchResult:
    return run_screened_mutation_scan(
        runtime=runtime,
        experiment_name=experiment_name,
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
        dispatch_partial_batches=dispatch_partial_batches,
        batch_gather_timeout_seconds=batch_gather_timeout_seconds,
        output_policy=output_policy,
        keep_query_outputs=keep_query_outputs,
    )


def compare_mutation_batch_case(
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
    runner_yaml: str | None = None,
    inference_ckpt_path: str | None = None,
    inference_ckpt_name: str | None = None,
    repo_dir: str | None = None,
    cache_query_results: bool = True,
    subprocess_batch_size: int = 1,
    dispatch_partial_batches: bool = False,
    batch_gather_timeout_seconds: float | None = None,
    screening_output_policy: str = "metrics_only",
    keep_screening_query_outputs: bool | None = None,
) -> BatchApproachComparison:
    return compare_mutation_batch_approaches(
        runtime=runtime,
        experiment_name=experiment_name,
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
        dispatch_partial_batches=dispatch_partial_batches,
        batch_gather_timeout_seconds=batch_gather_timeout_seconds,
        screening_output_policy=screening_output_policy,
        keep_screening_query_outputs=keep_screening_query_outputs,
    )


def run_server_end_to_end_case(
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
    runner_yaml: str | None = None,
    inference_ckpt_path: str | None = None,
    inference_ckpt_name: str | None = None,
    repo_dir: str | None = None,
    run_screening: bool = True,
    cache_query_results: bool = True,
    subprocess_batch_size: int = 1,
    dispatch_partial_batches: bool = False,
    batch_gather_timeout_seconds: float | None = None,
    screening_output_policy: str = "metrics_only",
    keep_screening_query_outputs: bool | None = None,
) -> ServerEndToEndResult:
    return run_server_end_to_end_smoke(
        runtime=runtime,
        experiment_name=experiment_name,
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
        run_screening=run_screening,
        cache_query_results=cache_query_results,
        subprocess_batch_size=subprocess_batch_size,
        dispatch_partial_batches=dispatch_partial_batches,
        batch_gather_timeout_seconds=batch_gather_timeout_seconds,
        screening_output_policy=screening_output_policy,
        keep_screening_query_outputs=keep_screening_query_outputs,
    )
