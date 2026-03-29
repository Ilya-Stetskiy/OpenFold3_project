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

"""Mutation-screening orchestration layer for OpenFold3."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from openfold3.projects.of3_all_atom.config.inference_query_format import (
    Chain,
    InferenceQuerySet,
    Query,
)

logger = logging.getLogger(__name__)

CANONICAL_AA = tuple("ACDEFGHIKLMNPQRSTVWY")
BATCH_GATHER_TIMEOUT_SECONDS = 0.05


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _metric_or_default(value: Any, default: float) -> float:
    return default if value is None else value


def _chunked(items: list[Any], chunk_size: int) -> list[list[Any]]:
    size = max(1, chunk_size)
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def apply_point_mutation(
    sequence: str,
    position_1based: int,
    new_residue: str,
    expected_residue: str | None = None,
) -> str:
    if not sequence:
        raise ValueError("sequence must be non-empty")
    if position_1based < 1 or position_1based > len(sequence):
        raise ValueError(
            f"position_1based={position_1based} is out of bounds for sequence length"
            f" {len(sequence)}"
        )
    new_residue = new_residue.upper()
    if new_residue not in CANONICAL_AA:
        raise ValueError(f"new_residue must be canonical amino acid, got {new_residue}")

    position = position_1based - 1
    current = sequence[position]
    if expected_residue is not None and current != expected_residue.upper():
        raise ValueError(
            f"Expected residue {expected_residue.upper()} at position"
            f" {position_1based}, found {current}"
        )
    if current == new_residue:
        return sequence
    return sequence[:position] + new_residue + sequence[position + 1 :]


@dataclass(frozen=True)
class MutationSpec:
    chain_id: str
    position_1based: int
    from_residue: str
    to_residue: str

    @property
    def mutation_id(self) -> str:
        return (
            f"{self.chain_id}_{self.from_residue.upper()}"
            f"{self.position_1based}{self.to_residue.upper()}"
        )


@dataclass
class ScreeningResultRow:
    mutation_id: str
    query_id: str
    query_hash: str
    sample_index: int | None
    seed: int | None
    sample_ranking_score: float | None
    iptm: float | None
    ptm: float | None
    avg_plddt: float | None
    gpde: float | None
    has_clash: float | None
    cache_hit: bool
    sequence_cache_hits: int
    query_result_cache_hit: bool
    cpu_prep_seconds: float
    gpu_inference_seconds: float
    total_seconds: float
    output_dir: str
    aggregated_confidence_path: str | None = None
    mutation_spec: dict[str, Any] | None = None
    derived_interface_metrics: dict[str, Any] = field(default_factory=dict)
    query_output_cleaned: bool = False


@dataclass
class ScreeningJob:
    base_query: Query
    mutations: list[MutationSpec]
    output_dir: Path
    cache_dir: Path
    query_prefix: str = "screen"
    include_wt: bool = True
    run_baseline_first: bool = True
    msa_policy: str = "reuse_precomputed"
    template_policy: str = "reuse_precomputed"
    output_policy: str = "metrics_only"
    resume: bool = True
    cache_query_results: bool = True
    num_cpu_workers: int = 4
    max_inflight_queries: int = 4
    subprocess_batch_size: int = 1
    dispatch_partial_batches: bool = False
    batch_gather_timeout_seconds: float = BATCH_GATHER_TIMEOUT_SECONDS
    num_diffusion_samples: int | None = None
    num_model_seeds: int | None = None
    runner_yaml: Path | None = None
    inference_ckpt_path: Path | None = None
    inference_ckpt_name: str | None = None
    use_msa_server: bool = False
    use_templates: bool = False
    min_free_disk_gb: float = 1.0
    cleanup_query_outputs: bool = True
    log_file: Path | None = None

    @classmethod
    def from_json_file(cls, path: Path) -> ScreeningJob:
        data = json.loads(path.read_text(encoding="utf-8"))
        base_query = Query.model_validate(data["base_query"])
        mutations = [MutationSpec(**item) for item in data["mutations"]]
        data["base_query"] = base_query
        data["mutations"] = mutations
        data["output_dir"] = Path(data["output_dir"])
        data["cache_dir"] = Path(data["cache_dir"])
        if data.get("runner_yaml") is not None:
            data["runner_yaml"] = Path(data["runner_yaml"])
        if data.get("inference_ckpt_path") is not None:
            data["inference_ckpt_path"] = Path(data["inference_ckpt_path"])
        if data.get("log_file") is not None:
            data["log_file"] = Path(data["log_file"])
        return cls(**data)


@dataclass
class PreparedMutationJob:
    mutation_id: str
    mutation_spec: MutationSpec | None
    query_id: str
    query_hash: str
    payload_path: Path
    output_dir: Path
    cache_hit: bool
    sequence_cache_hits: int
    cpu_prep_seconds: float


class PredictBackend(Protocol):
    def run(self, prepared_job: PreparedMutationJob) -> ScreeningResultRow: ...


class SequenceArtifactCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "sequence_index.json"
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        else:
            self.index = {}
        self._lock = threading.Lock()

    def _persist(self) -> None:
        self.index_path.write_text(
            json.dumps(self.index, indent=2, sort_keys=True), encoding="utf-8"
        )

    def register_chain(self, chain: Chain) -> None:
        if chain.sequence is None:
            return
        payload = {}
        if chain.main_msa_file_paths:
            payload["main_msa_file_paths"] = [str(p) for p in chain.main_msa_file_paths]
        if chain.paired_msa_file_paths:
            payload["paired_msa_file_paths"] = [
                str(p) for p in chain.paired_msa_file_paths
            ]
        if chain.template_alignment_file_path:
            payload["template_alignment_file_path"] = str(
                chain.template_alignment_file_path
            )
        if chain.template_entry_chain_ids:
            payload["template_entry_chain_ids"] = list(chain.template_entry_chain_ids)
        if not payload:
            return

        seq_key = sequence_hash(chain.sequence)
        with self._lock:
            self.index[seq_key] = payload
            self._persist()

    def resolve_chain(self, chain: Chain) -> bool:
        if chain.sequence is None:
            return False
        seq_key = sequence_hash(chain.sequence)
        entry = self.index.get(seq_key)
        if entry is None:
            return False

        hit = False

        entry_main = entry.get("main_msa_file_paths")
        if entry_main and (
            not chain.main_msa_file_paths
            or [str(p) for p in chain.main_msa_file_paths] == entry_main
        ):
            hit = True
        if not chain.main_msa_file_paths and entry_main:
            chain.main_msa_file_paths = [Path(p) for p in entry["main_msa_file_paths"]]

        entry_paired = entry.get("paired_msa_file_paths")
        if entry_paired and (
            not chain.paired_msa_file_paths
            or [str(p) for p in chain.paired_msa_file_paths] == entry_paired
        ):
            hit = True
        if not chain.paired_msa_file_paths and entry_paired:
            chain.paired_msa_file_paths = [
                Path(p) for p in entry["paired_msa_file_paths"]
            ]

        entry_template_alignment = entry.get("template_alignment_file_path")
        if (
            entry_template_alignment is not None
            and (
                chain.template_alignment_file_path is None
                or str(chain.template_alignment_file_path)
                == entry_template_alignment
            )
        ):
            hit = True
        if (
            chain.template_alignment_file_path is None
            and entry_template_alignment is not None
        ):
            chain.template_alignment_file_path = Path(
                entry["template_alignment_file_path"]
            )

        entry_template_ids = entry.get("template_entry_chain_ids")
        if (
            entry_template_ids is not None
            and (
                not chain.template_entry_chain_ids
                or list(chain.template_entry_chain_ids) == entry_template_ids
            )
        ):
            hit = True
        if not chain.template_entry_chain_ids and entry_template_ids is not None:
            chain.template_entry_chain_ids = list(entry["template_entry_chain_ids"])
        return hit


class QueryResultCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, query_hash: str) -> Path:
        return self.root / f"{query_hash}.json"

    def load(self, query_hash: str) -> ScreeningResultRow | None:
        path = self._path_for(query_hash)
        if not path.exists():
            return None
        return ScreeningResultRow(**json.loads(path.read_text(encoding="utf-8")))

    def store(self, row: ScreeningResultRow) -> None:
        path = self._path_for(row.query_hash)
        path.write_text(
            json.dumps(asdict(row), indent=2, default=_json_default),
            encoding="utf-8",
        )


class SubprocessOpenFoldBackend:
    def __init__(self, job: ScreeningJob):
        self.job = job
        self._batch_counter = 0

    def _cleanup_query_output_dir(self, output_dir: Path) -> None:
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)

    @staticmethod
    def _summarize_query_ids(
        prepared_jobs: list[PreparedMutationJob], max_items: int = 4
    ) -> str:
        query_ids = [prepared_job.query_id for prepared_job in prepared_jobs]
        if len(query_ids) <= max_items:
            return ", ".join(query_ids)
        head = ", ".join(query_ids[:max_items])
        return f"{head}, ... ({len(query_ids)} total)"

    @staticmethod
    def _log_label_for_batch(
        prepared_jobs: list[PreparedMutationJob], output_dir: Path
    ) -> str:
        if len(prepared_jobs) == 1:
            return prepared_jobs[0].query_id
        return f"{output_dir.name}|{len(prepared_jobs)}q"

    def _runner_yaml_for_output_dir(self, output_dir: Path) -> Path:
        config = {}
        if self.job.runner_yaml is not None:
            config = (
                yaml.safe_load(self.job.runner_yaml.read_text(encoding="utf-8")) or {}
            )

        output_settings = config.setdefault("output_writer_settings", {})
        output_settings["metrics_only"] = self.job.output_policy == "metrics_only"
        output_settings["cif_only"] = self.job.output_policy == "cif_only"
        output_settings.setdefault("summary_filename", "summary.jsonl")

        data_module_args = config.setdefault("data_module_args", {})
        data_module_args.setdefault("predict_num_workers", self.job.num_cpu_workers)
        data_module_args.setdefault("persistent_workers", True)
        data_module_args.setdefault("predict_persistent_workers", True)
        data_module_args.setdefault("pin_memory", True)

        experiment_settings = config.setdefault("experiment_settings", {})
        experiment_settings.setdefault("skip_existing", self.job.resume)

        yaml_path = output_dir / "runner.override.yml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return yaml_path

    def _build_predict_cmd(
        self,
        query_json: Path,
        output_dir: Path,
        runner_yaml: Path,
    ) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "openfold3.run_openfold",
            "predict",
            "--query_json",
            str(query_json),
            "--output_dir",
            str(output_dir),
            "--use_msa_server",
            str(self.job.use_msa_server).lower(),
            "--use_templates",
            str(self.job.use_templates).lower(),
            "--runner_yaml",
            str(runner_yaml),
        ]

        if self.job.num_diffusion_samples is not None:
            cmd += ["--num_diffusion_samples", str(self.job.num_diffusion_samples)]
        if self.job.num_model_seeds is not None:
            cmd += ["--num_model_seeds", str(self.job.num_model_seeds)]
        if self.job.inference_ckpt_path is not None:
            cmd += ["--inference_ckpt_path", str(self.job.inference_ckpt_path)]
        if self.job.inference_ckpt_name is not None:
            cmd += ["--inference_ckpt_name", self.job.inference_ckpt_name]
        return cmd

    @staticmethod
    def _predict_subprocess_env() -> dict[str, str]:
        env = os.environ.copy()
        warning_filters = [
            "ignore::DeprecationWarning",
            "ignore::FutureWarning",
        ]
        existing = env.get("PYTHONWARNINGS")
        env["PYTHONWARNINGS"] = ",".join(
            [*([existing] if existing else []), *warning_filters]
        )
        return env

    @staticmethod
    def _should_skip_subprocess_log_line(stripped: str) -> bool:
        if stripped == "return data.pin_memory(device)":
            return True

        noisy_substrings = (
            "DeprecationWarning:",
            "FutureWarning:",
            "The 'predict_dataloader' does not have many workers",
            "`isinstance(treespec, LeafSpec)` is deprecated",
        )
        return any(token in stripped for token in noisy_substrings)

    def _run_predict(self, cmd: list[str], log_label: str) -> float:
        start = time.perf_counter()
        process = subprocess.Popen(
            cmd,
            env=self._predict_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        previous_line = None
        for line in process.stdout:
            stripped = line.strip()
            if (
                not stripped
                or stripped == previous_line
                or self._should_skip_subprocess_log_line(stripped)
            ):
                continue
            previous_line = stripped
            logger.info("[predict:%s] %s", log_label, stripped)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        return time.perf_counter() - start

    @staticmethod
    def _merge_payloads(prepared_jobs: list[PreparedMutationJob]) -> dict[str, Any]:
        merged_payload: dict[str, Any] = {"queries": {}}
        for prepared_job in prepared_jobs:
            payload = json.loads(prepared_job.payload_path.read_text(encoding="utf-8"))
            if "seeds" in payload and "seeds" not in merged_payload:
                merged_payload["seeds"] = payload["seeds"]
            merged_payload["queries"].update(payload.get("queries", {}))
        return merged_payload

    def _write_batch_payload(
        self,
        prepared_jobs: list[PreparedMutationJob],
        batch_output_dir: Path,
    ) -> Path:
        payload_path = batch_output_dir / "batched_queries.json"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(
            json.dumps(self._merge_payloads(prepared_jobs), indent=2),
            encoding="utf-8",
        )
        return payload_path

    def _parse_best_summary_row(
        self, summary_path: Path, query_id: str
    ) -> dict[str, Any]:
        rows = []
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("query_id") == query_id:
                rows.append(row)
        if not rows:
            raise RuntimeError(f"No summary rows found for query_id={query_id}")

        def sort_key(item: dict[str, Any]):
            return (
                item.get("sample_ranking_score") is not None,
                _metric_or_default(item.get("sample_ranking_score"), float("-inf")),
                _metric_or_default(item.get("iptm"), float("-inf")),
                _metric_or_default(item.get("ptm"), float("-inf")),
                _metric_or_default(item.get("avg_plddt"), float("-inf")),
                -_metric_or_default(item.get("gpde"), float("inf")),
            )

        return max(rows, key=sort_key)

    def _row_from_best(
        self,
        prepared_job: PreparedMutationJob,
        best: dict[str, Any],
        *,
        gpu_seconds: float,
        output_dir: Path,
        query_output_cleaned: bool,
        aggregated_confidence_path: str | None,
    ) -> ScreeningResultRow:
        return ScreeningResultRow(
            mutation_id=prepared_job.mutation_id,
            query_id=prepared_job.query_id,
            query_hash=prepared_job.query_hash,
            sample_index=best.get("sample_index"),
            seed=best.get("seed"),
            sample_ranking_score=best.get("sample_ranking_score"),
            iptm=best.get("iptm"),
            ptm=best.get("ptm"),
            avg_plddt=best.get("avg_plddt"),
            gpde=best.get("gpde"),
            has_clash=best.get("has_clash"),
            cache_hit=prepared_job.cache_hit,
            sequence_cache_hits=prepared_job.sequence_cache_hits,
            query_result_cache_hit=False,
            cpu_prep_seconds=prepared_job.cpu_prep_seconds,
            gpu_inference_seconds=gpu_seconds,
            total_seconds=prepared_job.cpu_prep_seconds + gpu_seconds,
            output_dir=str(output_dir),
            aggregated_confidence_path=aggregated_confidence_path,
            mutation_spec=(
                asdict(prepared_job.mutation_spec)
                if prepared_job.mutation_spec is not None
                else None
            ),
            derived_interface_metrics=best.get("derived_interface_metrics", {}),
            query_output_cleaned=query_output_cleaned,
        )

    def run(self, prepared_job: PreparedMutationJob) -> ScreeningResultRow:
        return self.run_batch([prepared_job])[0]

    def run_batch(
        self, prepared_jobs: list[PreparedMutationJob]
    ) -> list[ScreeningResultRow]:
        if not prepared_jobs:
            return []

        if len(prepared_jobs) == 1:
            output_dir = prepared_jobs[0].output_dir
            payload_path = prepared_jobs[0].payload_path
            log_label = self._log_label_for_batch(prepared_jobs, output_dir)
            query_output_dirs = {
                prepared_jobs[0].query_id: prepared_jobs[0].output_dir,
            }
        else:
            self._batch_counter += 1
            output_dir = (
                self.job.output_dir
                / "batched_runs"
                / f"batch_{self._batch_counter:04d}"
            )
            payload_path = self._write_batch_payload(prepared_jobs, output_dir)
            log_label = self._log_label_for_batch(prepared_jobs, output_dir)
            query_output_dirs = {
                prepared_job.query_id: output_dir / prepared_job.query_id
                for prepared_job in prepared_jobs
            }

        runner_yaml = self._runner_yaml_for_output_dir(output_dir)
        cmd = self._build_predict_cmd(payload_path, output_dir, runner_yaml)
        logger.info(
            "Launching %s query(s) via %s: %s",
            len(prepared_jobs),
            log_label,
            self._summarize_query_ids(prepared_jobs),
        )
        batch_gpu_seconds = self._run_predict(cmd, log_label)
        gpu_seconds_per_query = batch_gpu_seconds / len(prepared_jobs)
        summary_path = output_dir / "summary.jsonl"

        rows = []
        for prepared_job in prepared_jobs:
            best = self._parse_best_summary_row(summary_path, prepared_job.query_id)
            aggregated_confidence_path = best.get("aggregated_confidence_path")
            query_output_cleaned = False
            if self.job.cleanup_query_outputs:
                aggregated_confidence_path = None
                query_output_cleaned = True
            rows.append(
                self._row_from_best(
                    prepared_job,
                    best,
                    gpu_seconds=gpu_seconds_per_query,
                    output_dir=query_output_dirs[prepared_job.query_id],
                    query_output_cleaned=query_output_cleaned,
                    aggregated_confidence_path=aggregated_confidence_path,
                )
            )

        if self.job.cleanup_query_outputs:
            self._cleanup_query_output_dir(output_dir)

        return rows


class MutationScreeningRunner:
    def __init__(self, backend: PredictBackend | None = None):
        self.backend = backend

    @staticmethod
    def _configure_logging(job: ScreeningJob) -> None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        if not any(
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
            for handler in logger.handlers
        ):
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        if job.log_file is not None:
            resolved_log = job.log_file.resolve()
            resolved_log.parent.mkdir(parents=True, exist_ok=True)
            if not any(
                isinstance(handler, logging.FileHandler)
                and Path(handler.baseFilename) == resolved_log
                for handler in logger.handlers
            ):
                file_handler = logging.FileHandler(resolved_log, encoding="utf-8")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

    @staticmethod
    def _check_free_disk_or_raise(target_dir: Path, min_free_disk_gb: float) -> None:
        usage = shutil.disk_usage(target_dir.resolve())
        free_gb = usage.free / (1024**3)
        if free_gb < min_free_disk_gb:
            raise RuntimeError(
                f"Low disk space for {target_dir}: {free_gb:.2f} GB free,"
                f" below required reserve of {min_free_disk_gb:.2f} GB"
            )

    @staticmethod
    def _clone_query(query: Query) -> Query:
        return Query.model_validate(query.model_dump())

    def _register_base_chain_cache(
        self,
        base_query: Query,
        sequence_cache: SequenceArtifactCache,
        mutable_chain_ids: set[str],
    ) -> None:
        for chain in base_query.chains:
            if self._chain_overlaps_mutable_ids(chain, mutable_chain_ids):
                continue
            sequence_cache.register_chain(chain)

    def _apply_mutation(self, query: Query, mutation: MutationSpec) -> None:
        for chain in query.chains:
            if mutation.chain_id in chain.chain_ids and chain.sequence is not None:
                chain.sequence = apply_point_mutation(
                    chain.sequence,
                    mutation.position_1based,
                    mutation.to_residue,
                    expected_residue=mutation.from_residue,
                )
                return
        raise ValueError(f"Could not find mutable protein chain {mutation.chain_id}")

    @staticmethod
    def _chain_overlaps_mutable_ids(chain: Chain, mutable_chain_ids: set[str]) -> bool:
        return any(chain_id in mutable_chain_ids for chain_id in chain.chain_ids)

    @staticmethod
    def _mutable_chain_ids_from_query(
        query: Query, mutations: list[MutationSpec]
    ) -> set[str]:
        requested_chain_ids = {mutation.chain_id for mutation in mutations}
        mutable_chain_ids: set[str] = set()
        for chain in query.chains:
            if any(chain_id in requested_chain_ids for chain_id in chain.chain_ids):
                mutable_chain_ids.update(chain.chain_ids)
        return mutable_chain_ids

    def _resolve_sequence_cache(
        self,
        query: Query,
        sequence_cache: SequenceArtifactCache,
        mutable_chain_ids: set[str],
    ) -> int:
        hits = 0
        for chain in query.chains:
            if self._chain_overlaps_mutable_ids(chain, mutable_chain_ids):
                continue
            if sequence_cache.resolve_chain(chain):
                hits += 1
            sequence_cache.register_chain(chain)
        return hits

    def _query_hash(self, query: Query, job: ScreeningJob) -> str:
        payload = {
            "query": query.model_dump(mode="json"),
            "job": {
                "msa_policy": job.msa_policy,
                "template_policy": job.template_policy,
                "output_policy": job.output_policy,
                "num_diffusion_samples": job.num_diffusion_samples,
                "num_model_seeds": job.num_model_seeds,
                "runner_yaml": str(job.runner_yaml) if job.runner_yaml else None,
                "inference_ckpt_path": (
                    str(job.inference_ckpt_path) if job.inference_ckpt_path else None
                ),
                "inference_ckpt_name": job.inference_ckpt_name,
                "use_msa_server": job.use_msa_server,
                "use_templates": job.use_templates,
            },
        }
        return _hash_payload(payload)

    def _write_payload(
        self, query_id: str, query: Query, query_hash: str, cache_dir: Path
    ) -> Path:
        payload_dir = cache_dir / "payloads"
        payload_dir.mkdir(parents=True, exist_ok=True)
        payload_path = payload_dir / f"{query_id}_{query_hash}.json"
        if not payload_path.exists():
            query_set = InferenceQuerySet(queries={query_id: query})
            payload_path.write_text(
                query_set.model_dump_json(indent=2), encoding="utf-8"
            )
        return payload_path

    def _prepare_single_job(
        self,
        job: ScreeningJob,
        mutation_spec: MutationSpec | None,
        sequence_cache: SequenceArtifactCache,
        result_cache: QueryResultCache | None,
        mutable_chain_ids: set[str],
    ) -> PreparedMutationJob | ScreeningResultRow:
        started = time.perf_counter()
        query = self._clone_query(job.base_query)
        mutation_id = "WT" if mutation_spec is None else mutation_spec.mutation_id
        query_id = f"{job.query_prefix}_{mutation_id}"

        if mutation_spec is not None:
            self._apply_mutation(query, mutation_spec)

        sequence_cache_hits = self._resolve_sequence_cache(
            query, sequence_cache, mutable_chain_ids
        )
        query_hash = self._query_hash(query, job)

        if job.resume and result_cache is not None:
            cached = result_cache.load(query_hash)
            if cached is not None:
                cached.cache_hit = True
                cached.query_result_cache_hit = True
                return cached

        payload_path = self._write_payload(query_id, query, query_hash, job.cache_dir)
        cpu_seconds = time.perf_counter() - started
        return PreparedMutationJob(
            mutation_id=mutation_id,
            mutation_spec=mutation_spec,
            query_id=query_id,
            query_hash=query_hash,
            payload_path=payload_path,
            output_dir=job.output_dir / "runs" / query_id,
            cache_hit=sequence_cache_hits > 0,
            sequence_cache_hits=sequence_cache_hits,
            cpu_prep_seconds=cpu_seconds,
        )

    def _entries(self, job: ScreeningJob) -> list[MutationSpec | None]:
        entries: list[MutationSpec | None] = list(job.mutations)
        if job.include_wt:
            if job.run_baseline_first:
                entries = [None] + entries
            else:
                entries = entries + [None]
        return entries

    def _run_prepared_jobs(
        self,
        job: ScreeningJob,
        backend: PredictBackend,
        prepared_jobs: list[PreparedMutationJob],
        result_cache: QueryResultCache | None,
        total_entries: int,
        rows: list[ScreeningResultRow],
    ) -> None:
        if not prepared_jobs:
            return

        self._check_free_disk_or_raise(job.output_dir, job.min_free_disk_gb)
        batch_query_ids = ", ".join(
            prepared_job.query_id for prepared_job in prepared_jobs
        )
        logger.info(
            "Launching batch of %s query(s): %s",
            len(prepared_jobs),
            batch_query_ids,
        )

        batch_runner = getattr(backend, "run_batch", None)
        if callable(batch_runner):
            batch_rows = batch_runner(prepared_jobs)
        else:
            batch_rows = [backend.run(prepared_job) for prepared_job in prepared_jobs]

        for row in batch_rows:
            if result_cache is not None:
                result_cache.store(row)
            self._check_free_disk_or_raise(job.output_dir, job.min_free_disk_gb)
            logger.info(
                "Completed %s/%s: %s in %.2fs "
                "(cache_hit=%s, sequence_cache_hits=%s, cleaned=%s)",
                len(rows) + 1,
                total_entries,
                row.query_id,
                row.total_seconds,
                row.cache_hit,
                row.sequence_cache_hits,
                row.query_output_cleaned,
            )
            rows.append(row)

    def run(self, job: ScreeningJob) -> list[ScreeningResultRow]:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        job.cache_dir.mkdir(parents=True, exist_ok=True)
        self._configure_logging(job)
        logger.info("Starting screening job in %s", job.output_dir)
        logger.info(
            "Disk reserve guard: %.2f GB, cleanup_query_outputs=%s",
            job.min_free_disk_gb,
            job.cleanup_query_outputs,
        )
        self._check_free_disk_or_raise(job.output_dir, job.min_free_disk_gb)

        sequence_cache = SequenceArtifactCache(job.cache_dir / "sequence")
        result_cache = (
            QueryResultCache(job.cache_dir / "results")
            if job.cache_query_results
            else None
        )
        mutable_chain_ids = self._mutable_chain_ids_from_query(
            job.base_query, job.mutations
        )
        self._register_base_chain_cache(
            job.base_query, sequence_cache, mutable_chain_ids
        )

        backend = self.backend or SubprocessOpenFoldBackend(job)
        prepared_queue: queue.Queue[
            PreparedMutationJob | ScreeningResultRow | BaseException | None
        ] = (
            queue.Queue(maxsize=max(1, job.max_inflight_queries))
        )
        entries = self._entries(job)
        total_entries = len(entries)
        rows: list[ScreeningResultRow] = []
        pending_prepared_jobs: list[PreparedMutationJob] = []
        batch_size = max(1, job.subprocess_batch_size)
        gather_timeout_seconds = max(0.0, job.batch_gather_timeout_seconds)
        producer_finished = False

        def producer() -> None:
            try:
                with ThreadPoolExecutor(
                    max_workers=max(1, job.num_cpu_workers)
                ) as executor:
                    futures = [
                        executor.submit(
                            self._prepare_single_job,
                            job,
                            mutation_spec,
                            sequence_cache,
                            result_cache,
                            mutable_chain_ids,
                        )
                        for mutation_spec in entries
                    ]
                    for future in as_completed(futures):
                        prepared_queue.put(future.result())
            except BaseException as exc:
                prepared_queue.put(exc)
            finally:
                prepared_queue.put(None)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        def handle_item(
            item: PreparedMutationJob | ScreeningResultRow | BaseException | None,
        ) -> None:
            nonlocal producer_finished
            if item is None:
                producer_finished = True
                return
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, ScreeningResultRow):
                logger.info(
                    "Using cached result for %s (query_result_cache_hit=%s)",
                    item.query_id,
                    item.query_result_cache_hit,
                )
                rows.append(item)
                return
            pending_prepared_jobs.append(item)

        while not producer_finished:
            handle_item(prepared_queue.get())
            timed_out_waiting_for_more = False
            while not producer_finished and len(pending_prepared_jobs) < batch_size:
                try:
                    handle_item(
                        prepared_queue.get(timeout=gather_timeout_seconds)
                    )
                except queue.Empty:
                    timed_out_waiting_for_more = True
                    break
            should_dispatch_partial_batch = (
                job.dispatch_partial_batches
                and timed_out_waiting_for_more
                and len(pending_prepared_jobs) > 0
            )
            if pending_prepared_jobs and (
                producer_finished
                or len(pending_prepared_jobs) >= batch_size
                or should_dispatch_partial_batch
            ):
                if should_dispatch_partial_batch:
                    logger.info(
                        "Dispatching partial batch of %s query(s) after %.2fs gather timeout",
                        len(pending_prepared_jobs),
                        gather_timeout_seconds,
                    )
                self._run_prepared_jobs(
                    job,
                    backend,
                    list(pending_prepared_jobs),
                    result_cache,
                    total_entries,
                    rows,
                )
                pending_prepared_jobs.clear()

        producer_thread.join()
        self._write_results(job.output_dir, rows)
        self._write_manifest(job, rows)
        logger.info("Screening job completed with %s rows", len(rows))
        return rows

    def _write_results(self, output_dir: Path, rows: list[ScreeningResultRow]) -> None:
        jsonl_path = output_dir / "results.jsonl"
        csv_path = output_dir / "results.csv"

        with jsonl_path.open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(asdict(row), default=_json_default) + "\n")

        if rows:
            fieldnames = list(asdict(rows[0]).keys())
            with csv_path.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            key: json.dumps(value, default=_json_default)
                            if isinstance(value, (dict, list))
                            else value
                            for key, value in asdict(row).items()
                        }
                    )

    def _write_manifest(
        self, job: ScreeningJob, rows: list[ScreeningResultRow]
    ) -> None:
        manifest = {
            "query_prefix": job.query_prefix,
            "output_policy": job.output_policy,
            "msa_policy": job.msa_policy,
            "template_policy": job.template_policy,
            "num_mutations": len(job.mutations),
            "include_wt": job.include_wt,
            "run_baseline_first": job.run_baseline_first,
            "cache_query_results": job.cache_query_results,
            "subprocess_batch_size": max(1, job.subprocess_batch_size),
            "dispatch_partial_batches": job.dispatch_partial_batches,
            "batch_gather_timeout_seconds": job.batch_gather_timeout_seconds,
            "rows_written": len(rows),
            "results_jsonl": str(job.output_dir / "results.jsonl"),
            "results_csv": str(job.output_dir / "results.csv"),
        }
        (job.output_dir / "screening_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )


def run_screening_job_from_json(
    screening_job_json: Path, output_dir: Path | None = None
) -> list[ScreeningResultRow]:
    job = ScreeningJob.from_json_file(screening_job_json)
    if output_dir is not None:
        job.output_dir = output_dir
    return MutationScreeningRunner().run(job)
