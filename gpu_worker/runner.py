from __future__ import annotations

import json
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .artifacts import write_json
from .schemas import (
    LeaseJob,
    PointMutation,
    PredictBatchPayload,
    VariantBatchPayload,
    VariantSpec,
    WorkerConfig,
)

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class PreparedCommand:
    cmd: list[str]
    cwd: Path
    env: dict[str, str]
    generated_inputs: list[Path]
    output_dir: Path
    log_path: Path
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunOutcome:
    return_code: int
    timed_out: bool
    elapsed_seconds: float


def normalize_molecules(molecules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, molecule in enumerate(molecules, start=1):
        mol = deepcopy(molecule)
        molecule_type = mol.get("molecule_type") or mol.get("type")
        if not molecule_type:
            raise ValueError(f"Molecule #{index} is missing molecule_type")
        chain_ids = mol.get("chain_ids")
        if chain_ids is None and "id" in mol:
            chain_ids = [mol["id"]]
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]
        if not chain_ids:
            raise ValueError(f"Molecule #{index} is missing chain identifiers")
        out: dict[str, Any] = {
            "molecule_type": str(molecule_type).lower(),
            "chain_ids": [str(chain_id) for chain_id in chain_ids],
        }
        if mol.get("sequence") is not None:
            out["sequence"] = str(mol["sequence"]).strip().upper()
        for field_name in (
            "smiles",
            "ccd_codes",
            "modifications",
            "main_msa_file_paths",
            "paired_msa_file_paths",
            "template_alignment_file_path",
            "template_entry_chain_ids",
        ):
            if mol.get(field_name) is not None:
                out[field_name] = deepcopy(mol[field_name])
        normalized.append(out)
    return normalized


def apply_point_mutation_to_molecules(
    molecules: list[dict[str, Any]],
    mutation: PointMutation,
) -> list[dict[str, Any]]:
    work = normalize_molecules(molecules)
    changed = False
    for molecule in work:
        if mutation.chain_id not in molecule["chain_ids"]:
            continue
        sequence = molecule.get("sequence")
        if not sequence:
            raise ValueError(f"Chain {mutation.chain_id} does not have a sequence")
        current = sequence[mutation.position_1based - 1]
        if mutation.from_residue is not None and current != mutation.from_residue:
            raise ValueError(
                f"Expected {mutation.from_residue} at {mutation.chain_id}:"
                f"{mutation.position_1based}, found {current}"
            )
        if mutation.to_residue not in CANONICAL_AA:
            raise ValueError(f"Unsupported residue {mutation.to_residue}")
        position = mutation.position_1based - 1
        molecule["sequence"] = (
            sequence[:position] + mutation.to_residue + sequence[position + 1 :]
        )
        changed = True
    if not changed:
        raise ValueError(f"Chain {mutation.chain_id} was not found")
    return work


class OpenFoldWorkerRunner:
    def __init__(self, config: WorkerConfig) -> None:
        self.config = config

    def prepare(self, lease: LeaseJob, work_dir: Path) -> PreparedCommand:
        if lease.job_type == "predict_batch":
            payload = PredictBatchPayload.model_validate(lease.payload)
            return self._prepare_predict(lease, payload, work_dir)
        payload = VariantBatchPayload.model_validate(lease.payload)
        return self._prepare_variant(lease, payload, work_dir)

    def _prepare_predict(
        self,
        lease: LeaseJob,
        payload: PredictBatchPayload,
        work_dir: Path,
    ) -> PreparedCommand:
        if (
            lease.limits.max_queries is not None
            and len(payload.queries) > lease.limits.max_queries
        ):
            raise ValueError(
                f"Job has {len(payload.queries)} queries, "
                f"max is {lease.limits.max_queries}"
            )
        query_json = {"queries": {}}
        if payload.seeds:
            query_json["seeds"] = payload.seeds
        for index, query in enumerate(payload.queries, start=1):
            query_id = query.query_id or f"{lease.job_id}_query_{index:04d}"
            query_json["queries"][query_id] = {
                "chains": normalize_molecules(query.molecules)
            }
        query_path = work_dir / "query.json"
        output_dir = work_dir / "output"
        log_path = work_dir / "run_openfold.log"
        write_json(query_path, query_json)
        cmd = self._predict_cmd(
            query_path=query_path,
            output_dir=output_dir,
            use_msa_server=payload.use_msa_server,
            use_templates=payload.use_templates,
            num_diffusion_samples=payload.num_diffusion_samples,
            num_model_seeds=payload.num_model_seeds,
            runner_yaml=payload.runner_yaml,
            inference_ckpt_path=payload.inference_ckpt_path,
            inference_ckpt_name=payload.inference_ckpt_name,
        )
        return PreparedCommand(
            cmd=cmd,
            cwd=self.config.effective_openfold_repo_dir,
            env=self._build_env(),
            generated_inputs=[query_path],
            output_dir=output_dir,
            log_path=log_path,
            summary={"query_count": len(payload.queries), "mode": "predict_batch"},
        )

    def _prepare_variant(
        self,
        lease: LeaseJob,
        payload: VariantBatchPayload,
        work_dir: Path,
    ) -> PreparedCommand:
        if (
            lease.limits.max_variants is not None
            and len(payload.variants) > lease.limits.max_variants
        ):
            raise ValueError(
                f"Job has {len(payload.variants)} variants, "
                f"max is {lease.limits.max_variants}"
            )
        if self._can_use_screen_mutations(payload.variants):
            return self._prepare_screen_mutations(lease, payload, work_dir)
        return self._prepare_variant_predict(lease, payload, work_dir)

    @staticmethod
    def _can_use_screen_mutations(variants: list[VariantSpec]) -> bool:
        return all(
            variant.mutations
            and len(variant.mutations) == 1
            and variant.molecules is None
            for variant in variants
        )

    def _prepare_screen_mutations(
        self,
        lease: LeaseJob,
        payload: VariantBatchPayload,
        work_dir: Path,
    ) -> PreparedCommand:
        mutations = []
        base_molecules = normalize_molecules(payload.base_query.molecules)
        for variant in payload.variants:
            assert variant.mutations is not None
            mutation = variant.mutations[0]
            from_residue = mutation.from_residue or self._infer_from_residue(
                base_molecules,
                mutation,
            )
            mutations.append(
                {
                    "chain_id": mutation.chain_id,
                    "position_1based": mutation.position_1based,
                    "from_residue": from_residue,
                    "to_residue": mutation.to_residue,
                }
            )
        output_dir = work_dir / "screening"
        cache_dir = work_dir / "cache"
        log_path = work_dir / "screen_mutations.log"
        job_path = work_dir / "screening_job.json"
        job = {
            "base_query": {"chains": base_molecules},
            "mutations": mutations,
            "output_dir": str(output_dir),
            "cache_dir": str(cache_dir),
            "query_prefix": payload.query_prefix,
            "include_wt": payload.include_wt,
            "run_baseline_first": True,
            "msa_policy": "reuse_precomputed",
            "template_policy": "reuse_precomputed",
            "output_policy": payload.output_policy,
            "resume": True,
            "cache_query_results": payload.cache_query_results,
            "num_cpu_workers": payload.num_cpu_workers,
            "max_inflight_queries": payload.max_inflight_queries,
            "subprocess_batch_size": payload.subprocess_batch_size,
            "dispatch_partial_batches": payload.dispatch_partial_batches,
            "num_diffusion_samples": payload.num_diffusion_samples,
            "num_model_seeds": payload.num_model_seeds,
            "runner_yaml": payload.runner_yaml,
            "inference_ckpt_path": payload.inference_ckpt_path,
            "inference_ckpt_name": payload.inference_ckpt_name,
            "use_msa_server": payload.use_msa_server,
            "use_templates": payload.use_templates,
            "min_free_disk_gb": (
                lease.limits.min_free_disk_gb or self.config.min_free_disk_gb
            ),
            "cleanup_query_outputs": False,
            "log_file": str(work_dir / "screening_runtime.log"),
        }
        if payload.batch_gather_timeout_seconds is not None:
            job["batch_gather_timeout_seconds"] = payload.batch_gather_timeout_seconds
        write_json(job_path, job)
        cmd = [
            str(self.config.openfold_python),
            "-m",
            "openfold3.run_openfold",
            "screen-mutations",
            "--screening_job_json",
            str(job_path),
        ]
        return PreparedCommand(
            cmd=cmd,
            cwd=self.config.effective_openfold_repo_dir,
            env=self._build_env(),
            generated_inputs=[job_path],
            output_dir=output_dir,
            log_path=log_path,
            summary={
                "variant_count": len(payload.variants),
                "mode": "screen-mutations",
            },
        )

    def _prepare_variant_predict(
        self,
        lease: LeaseJob,
        payload: VariantBatchPayload,
        work_dir: Path,
    ) -> PreparedCommand:
        queries = {}
        if payload.include_wt:
            queries[f"{payload.query_prefix}__WT"] = {
                "chains": normalize_molecules(payload.base_query.molecules)
            }
        for variant in payload.variants:
            if variant.molecules is not None:
                molecules = normalize_molecules(variant.molecules)
            else:
                molecules = normalize_molecules(payload.base_query.molecules)
                for mutation in variant.mutations or []:
                    molecules = apply_point_mutation_to_molecules(molecules, mutation)
            queries[f"{payload.query_prefix}__{variant.variant_id}"] = {
                "chains": molecules
            }
        query_path = work_dir / "query.json"
        output_dir = work_dir / "output"
        log_path = work_dir / "run_openfold.log"
        write_json(query_path, {"queries": queries})
        cmd = self._predict_cmd(
            query_path=query_path,
            output_dir=output_dir,
            use_msa_server=payload.use_msa_server,
            use_templates=payload.use_templates,
            num_diffusion_samples=payload.num_diffusion_samples,
            num_model_seeds=payload.num_model_seeds,
            runner_yaml=payload.runner_yaml,
            inference_ckpt_path=payload.inference_ckpt_path,
            inference_ckpt_name=payload.inference_ckpt_name,
        )
        return PreparedCommand(
            cmd=cmd,
            cwd=self.config.effective_openfold_repo_dir,
            env=self._build_env(),
            generated_inputs=[query_path],
            output_dir=output_dir,
            log_path=log_path,
            summary={"variant_count": len(payload.variants), "mode": "variant_predict"},
        )

    @staticmethod
    def _infer_from_residue(
        molecules: list[dict[str, Any]],
        mutation: PointMutation,
    ) -> str:
        for molecule in molecules:
            if mutation.chain_id in molecule["chain_ids"]:
                sequence = molecule.get("sequence")
                if not sequence:
                    raise ValueError(
                        f"Chain {mutation.chain_id} does not have a sequence"
                    )
                return sequence[mutation.position_1based - 1]
        raise ValueError(f"Chain {mutation.chain_id} was not found")

    def _predict_cmd(
        self,
        *,
        query_path: Path,
        output_dir: Path,
        use_msa_server: bool,
        use_templates: bool,
        num_diffusion_samples: int | None,
        num_model_seeds: int | None,
        runner_yaml: str | None,
        inference_ckpt_path: str | None,
        inference_ckpt_name: str | None,
    ) -> list[str]:
        cmd = [
            str(self.config.openfold_python),
            "-m",
            "openfold3.run_openfold",
            "predict",
            "--query_json",
            str(query_path),
            "--output_dir",
            str(output_dir),
            "--use_msa_server",
            str(use_msa_server).lower(),
            "--use_templates",
            str(use_templates).lower(),
        ]
        if num_diffusion_samples is not None:
            cmd += ["--num_diffusion_samples", str(num_diffusion_samples)]
        if num_model_seeds is not None:
            cmd += ["--num_model_seeds", str(num_model_seeds)]
        if runner_yaml is not None:
            cmd += ["--runner_yaml", runner_yaml]
        if inference_ckpt_path is not None:
            cmd += ["--inference_ckpt_path", inference_ckpt_path]
        if inference_ckpt_name is not None:
            cmd += ["--inference_ckpt_name", inference_ckpt_name]
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        project_dir = str(self.config.openfold_project_dir)
        repo_dir = str(self.config.effective_openfold_repo_dir)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{repo_dir}:{project_dir}" + (f":{existing}" if existing else "")
        )
        if self.config.openfold_prefix is not None:
            prefix_bin = self.config.openfold_prefix / "bin"
            if prefix_bin.exists():
                env["PATH"] = f"{prefix_bin}:{env.get('PATH', '')}"
        return env


def run_command(
    command: PreparedCommand,
    timeout_seconds: int,
    progress_callback: Callable[[float], None] | None = None,
) -> RunOutcome:
    command.output_dir.mkdir(parents=True, exist_ok=True)
    command.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timed_out = False
    process = subprocess.Popen(
        command.cmd,
        cwd=command.cwd,
        env=command.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    try:
        with command.log_path.open("w", encoding="utf-8") as log_file:
            while True:
                elapsed = time.perf_counter() - started
                if timeout_seconds and elapsed > timeout_seconds:
                    timed_out = True
                    _terminate_process_tree(process)
                    break
                readable, _, _ = select.select([process.stdout], [], [], 0.2)
                if readable:
                    line = process.stdout.readline()
                    if line:
                        log_file.write(line)
                        log_file.flush()
                if process.poll() is not None:
                    break
                if progress_callback is not None:
                    progress_callback(elapsed)
        remainder = process.stdout.read()
        if remainder:
            with command.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(remainder)
        return_code = process.wait()
    finally:
        process.stdout.close()
    return RunOutcome(
        return_code=return_code,
        timed_out=timed_out,
        elapsed_seconds=time.perf_counter() - started,
    )


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if sys.platform != "win32":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            process.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
