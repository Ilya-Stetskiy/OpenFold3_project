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
    )

    mutation_summary = summarize_mutation_batch(result.samples_df)
    mutation_ranking = rank_mutations(mutation_summary, top_n=len(mutation_summary))
    return result, mutation_summary, mutation_ranking
