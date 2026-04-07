from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from openfold3.benchmark.harness import DdgBenchmarkHarness
from openfold3.benchmark.methods import multiscale_methods
from openfold3.benchmark.models import BenchmarkCase, MutationInput
from openfold3.mutation_runner import CANONICAL_AA, apply_point_mutation
from openfold3.panel_profiling import PanelExperimentProfiler, PanelProfilingArtifacts
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
    Query,
)

logger = logging.getLogger(__name__)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _sort_key(row: dict[str, Any]) -> tuple[bool, float, float, float, float, float]:
    def _metric(value: Any, default: float) -> float:
        return default if value is None else float(value)

    return (
        row.get("sample_ranking_score") is not None,
        _metric(row.get("sample_ranking_score"), float("-inf")),
        _metric(row.get("iptm"), float("-inf")),
        _metric(row.get("ptm"), float("-inf")),
        _metric(row.get("avg_plddt"), float("-inf")),
        -_metric(row.get("gpde"), float("inf")),
    )


def _load_single_query(query_json: Path) -> tuple[str, Query]:
    query_set = InferenceQuerySet.from_json(query_json)
    if len(query_set.queries) != 1:
        raise ValueError(
            f"Expected exactly one WT query in {query_json}, got {len(query_set.queries)}"
        )
    return next(iter(query_set.queries.items()))


def _load_single_query_set(query_json: Path) -> InferenceQuerySet:
    query_set = InferenceQuerySet.from_json(query_json)
    if len(query_set.queries) != 1:
        raise ValueError(
            f"Expected exactly one query in {query_json}, got {len(query_set.queries)}"
        )
    return query_set


def _clone_query(query: Query) -> Query:
    return Query.model_validate(query.model_dump())


def _find_chain_sequence(query: Query, mutable_chain_id: str) -> str:
    for chain in query.chains:
        if mutable_chain_id in chain.chain_ids and chain.sequence is not None:
            return str(chain.sequence)
    raise ValueError(f"Could not find mutable protein chain {mutable_chain_id}")


def _panel_id(
    target_id: str, chain_id: str, from_residue: str, position_1based: int
) -> str:
    return f"{target_id}_{chain_id}_{from_residue.upper()}{position_1based}"


def _job_id(panel_id: str, to_residue: str) -> str:
    return f"{panel_id}{to_residue.upper()}"


def _best_summary_rows(summary_path: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = row.get("query_id")
        if not query_id:
            continue
        grouped.setdefault(str(query_id), []).append(row)
    return {query_id: max(rows, key=_sort_key) for query_id, rows in grouped.items()}


def _merge_inference_query_sets(query_json_paths: Iterable[Path]) -> InferenceQuerySet:
    merged_queries: dict[str, Query] = {}
    seeds: list[int] | None = None
    for query_json_path in query_json_paths:
        query_set = InferenceQuerySet.from_json(query_json_path)
        if seeds is None:
            seeds = list(query_set.seeds)
        for query_id, query in query_set.queries.items():
            if query_id in merged_queries:
                raise ValueError(f"Duplicate query_id {query_id} in merged predict batch")
            merged_queries[query_id] = query
    if not merged_queries:
        raise ValueError("No queries were provided for merged predict batch")
    return InferenceQuerySet(
        seeds=[42] if seeds is None else seeds,
        queries=merged_queries,
    )


def _infer_structure_path(aggregated_confidence_path: str | None) -> Path | None:
    if aggregated_confidence_path is None:
        return None
    path = Path(aggregated_confidence_path)
    if not path.exists():
        return None
    name = path.name
    if name.endswith("_confidences_aggregated.json"):
        prefix = name.removesuffix("_confidences_aggregated.json")
    elif name.endswith("_confidences.json"):
        prefix = name.removesuffix("_confidences.json")
    else:
        prefix = path.stem
    candidates = [
        path.with_name(f"{prefix}_model.cif"),
        path.with_name(f"{prefix}_model.pdb"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@dataclass(frozen=True)
class MutationPanel:
    panel_id: str
    chain_id: str
    position_1based: int
    from_residue: str
    job_ids: tuple[str, ...]
    mutations: tuple[MutationInput, ...]


@dataclass(frozen=True)
class PanelStandConfig:
    target_id: str
    wt_query_json: Path
    output_root: Path
    mutable_chain_id: str
    positions: tuple[int, ...]
    runner_yaml: Path | None = None
    inference_ckpt_path: Path | None = None
    inference_ckpt_name: str | None = None
    msa_computation_settings_yaml: Path | None = None
    num_diffusion_samples: int | None = None
    num_model_seeds: int | None = None
    msa_panel_workers: int = 1
    analysis_workers: int = 4
    reuse_wt_msa_for_mutants: bool = True
    predict_strategy: str = "adaptive"
    predict_panel_chunk_size: int | None = 8
    cleanup_intermediates: bool = True
    enable_profiling: bool = True
    profiling_sample_interval_seconds: float = 1.0


class PanelStandState:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS wt_baseline (
                    target_id TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    msa_status TEXT NOT NULL DEFAULT 'pending',
                    predict_status TEXT NOT NULL DEFAULT 'pending',
                    analysis_status TEXT NOT NULL DEFAULT 'pending',
                    msa_dir TEXT,
                    msa_query_json TEXT,
                    predict_dir TEXT,
                    structure_path TEXT,
                    confidence_path TEXT,
                    rosetta_score REAL,
                    wt_report_path TEXT,
                    last_error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS panels (
                    panel_id TEXT PRIMARY KEY,
                    target_id TEXT NOT NULL,
                    chain_id TEXT NOT NULL,
                    position_1based INTEGER NOT NULL,
                    from_residue TEXT NOT NULL,
                    msa_status TEXT NOT NULL DEFAULT 'pending',
                    predict_status TEXT NOT NULL DEFAULT 'pending',
                    cleanup_status TEXT NOT NULL DEFAULT 'pending',
                    query_json_path TEXT NOT NULL,
                    msa_dir TEXT NOT NULL,
                    msa_query_json TEXT,
                    predict_dir TEXT NOT NULL,
                    last_error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    panel_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    chain_id TEXT NOT NULL,
                    position_1based INTEGER NOT NULL,
                    from_residue TEXT NOT NULL,
                    to_residue TEXT NOT NULL,
                    predict_status TEXT NOT NULL DEFAULT 'pending',
                    analysis_status TEXT NOT NULL DEFAULT 'pending',
                    structure_path TEXT,
                    confidence_path TEXT,
                    report_path TEXT,
                    rosetta_delta_vs_wt REAL,
                    last_error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(panel_id) REFERENCES panels(panel_id)
                );

                CREATE TABLE IF NOT EXISTS method_results (
                    job_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score REAL,
                    units TEXT,
                    details_json TEXT NOT NULL,
                    PRIMARY KEY(job_id, method),
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );
                """
            )
            self.conn.commit()

    def upsert_wt(self, target_id: str, query_id: str) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO wt_baseline (target_id, query_id)
                VALUES (?, ?)
                ON CONFLICT(target_id) DO UPDATE SET
                    query_id=excluded.query_id,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (target_id, query_id),
            )
            self.conn.commit()

    def upsert_panel(self, target_id: str, panel: MutationPanel, panel_dir: Path) -> None:
        query_json_path = panel_dir / "queries.json"
        msa_dir = panel_dir / "msa"
        predict_dir = panel_dir / "predict"
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO panels (
                    panel_id, target_id, chain_id, position_1based, from_residue,
                    query_json_path, msa_dir, predict_dir
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(panel_id) DO UPDATE SET
                    target_id=excluded.target_id,
                    chain_id=excluded.chain_id,
                    position_1based=excluded.position_1based,
                    from_residue=excluded.from_residue,
                    query_json_path=excluded.query_json_path,
                    msa_dir=excluded.msa_dir,
                    predict_dir=excluded.predict_dir,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    panel.panel_id,
                    target_id,
                    panel.chain_id,
                    panel.position_1based,
                    panel.from_residue,
                    str(query_json_path),
                    str(msa_dir),
                    str(predict_dir),
                ),
            )
            for job_id, mutation in zip(panel.job_ids, panel.mutations, strict=True):
                self.conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, panel_id, target_id, query_id, chain_id,
                        position_1based, from_residue, to_residue
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        panel_id=excluded.panel_id,
                        target_id=excluded.target_id,
                        query_id=excluded.query_id,
                        chain_id=excluded.chain_id,
                        position_1based=excluded.position_1based,
                        from_residue=excluded.from_residue,
                        to_residue=excluded.to_residue,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        job_id,
                        panel.panel_id,
                        target_id,
                        job_id,
                        mutation.chain_id,
                        mutation.position_1based,
                        mutation.from_residue,
                        mutation.to_residue,
                    ),
                )
            self.conn.commit()

    def fetch_wt(self, target_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM wt_baseline WHERE target_id = ?",
                (target_id,),
            ).fetchone()

    def fetch_panel(self, panel_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM panels WHERE panel_id = ?",
                (panel_id,),
            ).fetchone()

    def fetch_job(self, job_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()

    def list_jobs_for_panel(self, panel_id: str) -> list[sqlite3.Row]:
        with self._lock:
            return list(
                self.conn.execute(
                    "SELECT * FROM jobs WHERE panel_id = ? ORDER BY job_id",
                    (panel_id,),
                )
            )

    def list_panels(self) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute("SELECT * FROM panels ORDER BY panel_id"))

    def set_wt_stage(self, target_id: str, stage: str, status: str, **extra: Any) -> None:
        columns = [f"{stage}_status = ?", "updated_at = CURRENT_TIMESTAMP"]
        values: list[Any] = [status]
        for key, value in extra.items():
            columns.append(f"{key} = ?")
            values.append(None if value is None else str(value) if isinstance(value, Path) else value)
        values.append(target_id)
        with self._lock:
            self.conn.execute(
                f"UPDATE wt_baseline SET {', '.join(columns)} WHERE target_id = ?",
                values,
            )
            self.conn.commit()

    def set_panel_stage(self, panel_id: str, stage: str, status: str, **extra: Any) -> None:
        columns = [f"{stage}_status = ?", "updated_at = CURRENT_TIMESTAMP"]
        values: list[Any] = [status]
        for key, value in extra.items():
            columns.append(f"{key} = ?")
            values.append(None if value is None else str(value) if isinstance(value, Path) else value)
        values.append(panel_id)
        with self._lock:
            self.conn.execute(
                f"UPDATE panels SET {', '.join(columns)} WHERE panel_id = ?",
                values,
            )
            self.conn.commit()

    def update_job_predict(
        self,
        job_id: str,
        *,
        status: str,
        structure_path: Path | None,
        confidence_path: Path | None,
        last_error: str | None = None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE jobs
                SET predict_status = ?, structure_path = ?, confidence_path = ?,
                    last_error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                """,
                (
                    status,
                    None if structure_path is None else str(structure_path),
                    None if confidence_path is None else str(confidence_path),
                    last_error,
                    job_id,
                ),
            )
            self.conn.commit()

    def update_job_analysis(
        self,
        job_id: str,
        *,
        status: str,
        report_path: Path | None,
        rosetta_delta_vs_wt: float | None,
        last_error: str | None = None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE jobs
                SET analysis_status = ?, report_path = ?, rosetta_delta_vs_wt = ?,
                    last_error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                """,
                (
                    status,
                    None if report_path is None else str(report_path),
                    rosetta_delta_vs_wt,
                    last_error,
                    job_id,
                ),
            )
            self.conn.commit()

    def replace_method_results(self, job_id: str, results: Iterable[dict[str, Any]]) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM method_results WHERE job_id = ?", (job_id,))
            for result in results:
                self.conn.execute(
                    """
                    INSERT INTO method_results (job_id, method, status, score, units, details_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        result["method"],
                        result["status"],
                        result["score"],
                        result["units"],
                        json.dumps(result["details"], sort_keys=True, default=_json_default),
                    ),
                )
            self.conn.commit()

    def fetch_method_score(self, job_id: str, method: str) -> float | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT score FROM method_results WHERE job_id = ? AND method = ?",
                (job_id, method),
            ).fetchone()
        if row is None:
            return None
        value = row["score"]
        return None if value is None else float(value)

    def panel_is_ready_for_cleanup(self, panel_id: str) -> bool:
        with self._lock:
            rows = list(
                self.conn.execute(
                    "SELECT analysis_status FROM jobs WHERE panel_id = ?",
                    (panel_id,),
                )
            )
        return bool(rows) and all(row["analysis_status"] == "done" for row in rows)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            panel_rows = list(
                self.conn.execute(
                    """
                    SELECT msa_status, predict_status, cleanup_status, COUNT(*) AS n
                    FROM panels
                    GROUP BY msa_status, predict_status, cleanup_status
                    """
                )
            )
            job_rows = list(
                self.conn.execute(
                    """
                    SELECT predict_status, analysis_status, COUNT(*) AS n
                    FROM jobs
                    GROUP BY predict_status, analysis_status
                    """
                )
            )
        return {
            "panels": [dict(row) for row in panel_rows],
            "jobs": [dict(row) for row in job_rows],
        }


class PanelDdgStandRunner:
    def __init__(self, config: PanelStandConfig):
        self.config = config
        self.output_root = config.output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.db = PanelStandState(self.output_root / "state.sqlite")
        self.harness = DdgBenchmarkHarness(methods=multiscale_methods())
        self.analysis_allowed = threading.Event()
        self.analysis_queue: queue.Queue[str | None] = queue.Queue()
        self._deferred_analysis_job_ids: list[str] = []
        self._deferred_analysis_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._predict_batch_lock = threading.Lock()
        self._predict_batch_index = 0
        self._panel_msa_seconds: dict[str, float] = {}
        self._job_analysis_seconds: dict[str, float] = {}
        self.profiler = (
            PanelExperimentProfiler(
                output_root=self.output_root / "profiling",
                run_id=self.config.target_id,
                sample_interval_seconds=self.config.profiling_sample_interval_seconds,
            )
            if self.config.enable_profiling
            else None
        )

    def close(self) -> None:
        self.db.close()

    def _log(self, message: str) -> None:
        with self._log_lock:
            print(message, flush=True)
        logger.info(message)

    def _python_cmd_prefix(self) -> list[str]:
        return [sys.executable, "-m", "openfold3.run_openfold"]

    def _run_subprocess(self, command: list[str], prefix: str, cwd: Path | None = None) -> None:
        env = dict(os.environ)
        process = subprocess.Popen(
            command,
            cwd=None if cwd is None else str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        if self.profiler is not None:
            self.profiler.record_stage(
                "subprocess",
                "start",
                source=prefix,
                details={"pid": process.pid, "command": command},
            )
        assert process.stdout is not None
        for line in process.stdout:
            self._log(f"[{prefix}] {line.rstrip()}")
            if self.profiler is not None:
                self.profiler.record_log_line(line, source=prefix)
        return_code = process.wait()
        if self.profiler is not None:
            self.profiler.record_stage(
                "subprocess",
                "end",
                source=prefix,
                details={"pid": process.pid, "return_code": return_code},
            )
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

    def _panel_dir(self, panel_id: str) -> Path:
        return self.output_root / "panels" / panel_id

    def _report_path(self, job_id: str) -> Path:
        return self.output_root / "reports" / f"{job_id}.json"

    def _wt_dir(self) -> Path:
        return self.output_root / "wt"

    def _predict_batch_root_dir(self) -> Path:
        return self.output_root / "predict_batch"

    def _next_batched_predict_dir(self) -> Path:
        with self._predict_batch_lock:
            batch_root = self._predict_batch_root_dir()
            batch_root.mkdir(parents=True, exist_ok=True)
            while True:
                self._predict_batch_index += 1
                candidate = batch_root / f"batch_{self._predict_batch_index:04d}"
                if candidate.exists():
                    continue
                return candidate

    def _build_panels(self) -> tuple[str, Query, list[MutationPanel]]:
        wt_query_id, wt_query = _load_single_query(self.config.wt_query_json)
        sequence = _find_chain_sequence(wt_query, self.config.mutable_chain_id)
        panels: list[MutationPanel] = []
        for position in self.config.positions:
            from_residue = sequence[position - 1]
            panel_name = _panel_id(
                self.config.target_id,
                self.config.mutable_chain_id,
                from_residue,
                position,
            )
            mutations: list[MutationInput] = []
            job_ids: list[str] = []
            for to_residue in CANONICAL_AA:
                if to_residue == from_residue:
                    continue
                mutation = MutationInput(
                    chain_id=self.config.mutable_chain_id,
                    from_residue=from_residue,
                    position_1based=position,
                    to_residue=to_residue,
                )
                mutations.append(mutation)
                job_ids.append(_job_id(panel_name, to_residue))
            panels.append(
                MutationPanel(
                    panel_id=panel_name,
                    chain_id=self.config.mutable_chain_id,
                    position_1based=position,
                    from_residue=from_residue,
                    job_ids=tuple(job_ids),
                    mutations=tuple(mutations),
                )
            )
        return wt_query_id, wt_query, panels

    def _write_wt_query_json(self, wt_query_id: str, wt_query: Query) -> Path:
        wt_dir = self._wt_dir()
        wt_dir.mkdir(parents=True, exist_ok=True)
        path = wt_dir / "query.json"
        if not path.exists():
            query_set = InferenceQuerySet(queries={wt_query_id: wt_query})
            path.write_text(query_set.model_dump_json(indent=2), encoding="utf-8")
        return path

    def _write_panel_query_json(self, base_query: Query, panel: MutationPanel) -> Path:
        panel_dir = self._panel_dir(panel.panel_id)
        panel_dir.mkdir(parents=True, exist_ok=True)
        query_json = panel_dir / "queries.json"
        if query_json.exists():
            return query_json
        queries: dict[str, Query] = {}
        for job_id, mutation in zip(panel.job_ids, panel.mutations, strict=True):
            query = _clone_query(base_query)
            for chain in query.chains:
                if mutation.chain_id in chain.chain_ids and chain.sequence is not None:
                    chain.sequence = apply_point_mutation(
                        chain.sequence,
                        mutation.position_1based,
                        mutation.to_residue,
                        expected_residue=mutation.from_residue,
                    )
                    break
            else:
                raise ValueError(
                    f"Could not find mutable chain {mutation.chain_id} for panel {panel.panel_id}"
                )
            queries[job_id] = query
        query_json.write_text(
            InferenceQuerySet(queries=queries).model_dump_json(indent=2),
            encoding="utf-8",
        )
        return query_json

    def _write_panel_query_msa_json(self, wt_query_msa_json: Path, panel: MutationPanel) -> Path:
        panel_dir = self._panel_dir(panel.panel_id)
        msa_dir = panel_dir / "msa"
        msa_dir.mkdir(parents=True, exist_ok=True)
        query_msa_json = msa_dir / "query_msa.json"
        if query_msa_json.exists():
            return query_msa_json
        wt_query_set = _load_single_query_set(wt_query_msa_json)
        wt_query = next(iter(wt_query_set.queries.values()))
        queries: dict[str, Query] = {}
        for job_id, mutation in zip(panel.job_ids, panel.mutations, strict=True):
            query = _clone_query(wt_query)
            for chain in query.chains:
                if mutation.chain_id in chain.chain_ids and chain.sequence is not None:
                    chain.sequence = apply_point_mutation(
                        chain.sequence,
                        mutation.position_1based,
                        mutation.to_residue,
                        expected_residue=mutation.from_residue,
                    )
                    break
            else:
                raise ValueError(
                    f"Could not find mutable chain {mutation.chain_id} for panel {panel.panel_id}"
                )
            queries[job_id] = query
        query_msa_json.write_text(
            InferenceQuerySet(seeds=list(wt_query_set.seeds), queries=queries).model_dump_json(
                indent=2
            ),
            encoding="utf-8",
        )
        return query_msa_json

    def _align_msa(self, query_json: Path, output_dir: Path, label: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        query_msa_json = output_dir / "query_msa.json"
        if query_msa_json.exists():
            self._log(f"[resume] MSA already present for {label}: {query_msa_json}")
            return query_msa_json
        command = self._python_cmd_prefix() + [
            "align-msa-server",
            "--query_json",
            str(query_json),
            "--output_dir",
            str(output_dir),
        ]
        if self.config.msa_computation_settings_yaml is not None:
            command += [
                "--msa_computation_settings_yaml",
                str(self.config.msa_computation_settings_yaml),
            ]
        self._run_subprocess(command, f"msa:{label}")
        if not query_msa_json.exists():
            raise RuntimeError(f"MSA stage did not produce {query_msa_json}")
        return query_msa_json

    def _predict(self, query_json: Path, output_dir: Path, label: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.jsonl"
        if summary_path.exists():
            self._log(f"[resume] predict summary already present for {label}: {summary_path}")
            return summary_path
        command = self._python_cmd_prefix() + [
            "predict",
            "--query_json",
            str(query_json),
            "--output_dir",
            str(output_dir),
            "--use_msa_server",
            "false",
            "--use_templates",
            "false",
        ]
        if self.config.runner_yaml is not None:
            command += ["--runner_yaml", str(self.config.runner_yaml)]
        if self.config.inference_ckpt_path is not None:
            command += ["--inference_ckpt_path", str(self.config.inference_ckpt_path)]
        if self.config.inference_ckpt_name is not None:
            command += ["--inference_ckpt_name", self.config.inference_ckpt_name]
        if self.config.num_diffusion_samples is not None:
            command += ["--num_diffusion_samples", str(self.config.num_diffusion_samples)]
        if self.config.num_model_seeds is not None:
            command += ["--num_model_seeds", str(self.config.num_model_seeds)]
        self._run_subprocess(command, f"predict:{label}")
        if not summary_path.exists():
            raise RuntimeError(f"Predict stage did not produce {summary_path}")
        return summary_path

    def _enqueue_analysis_jobs(self, job_ids: Iterable[str]) -> None:
        if self.analysis_allowed.is_set():
            for job_id in job_ids:
                self.analysis_queue.put(job_id)
            return
        with self._deferred_analysis_lock:
            self._deferred_analysis_job_ids.extend(job_ids)

    def _drain_deferred_analysis_jobs(self) -> None:
        with self._deferred_analysis_lock:
            pending = list(self._deferred_analysis_job_ids)
            self._deferred_analysis_job_ids.clear()
        for job_id in pending:
            self.analysis_queue.put(job_id)

    def _effective_predict_panel_chunk_size(self, panel_count: int) -> int:
        configured = self.config.predict_panel_chunk_size
        if configured is None or configured <= 0:
            return max(1, panel_count)
        return max(1, min(configured, panel_count))

    def _analysis_finished_for_job(self, job_id: str) -> bool:
        row = self.db.fetch_job(job_id)
        if row is None:
            return True
        return str(row["analysis_status"]) in {"done", "failed"}

    def _wait_for_analysis_completion(
        self,
        job_ids: Iterable[str],
        *,
        poll_seconds: float = 0.2,
    ) -> None:
        pending = {str(job_id) for job_id in job_ids}
        while pending:
            finished = {job_id for job_id in pending if self._analysis_finished_for_job(job_id)}
            pending.difference_update(finished)
            if pending:
                time.sleep(poll_seconds)

    def _job_ids_for_panels(self, panel_ids: Iterable[str]) -> list[str]:
        job_ids: list[str] = []
        for panel_id in panel_ids:
            job_ids.extend(str(row["job_id"]) for row in self.db.list_jobs_for_panel(str(panel_id)))
        return job_ids

    def _select_predict_mode(
        self,
        *,
        first_chunk_cpu_seconds: float,
        first_chunk_gpu_seconds: float,
    ) -> str:
        requested = str(self.config.predict_strategy).strip().lower()
        if requested in {"single", "single_batch"}:
            return "single_batch"
        if requested in {"chunked", "parallel"}:
            return "chunked"
        if requested != "adaptive":
            raise ValueError(f"Unsupported predict_strategy: {self.config.predict_strategy}")
        if first_chunk_gpu_seconds >= first_chunk_cpu_seconds:
            return "chunked"
        return "single_batch"

    def _ensure_wt(self, wt_query_id: str, wt_query: Query) -> None:
        self.db.upsert_wt(self.config.target_id, wt_query_id)
        wt_row = self.db.fetch_wt(self.config.target_id)
        assert wt_row is not None

        wt_query_json = self._write_wt_query_json(wt_query_id, wt_query)
        wt_dir = self._wt_dir()
        msa_dir = wt_dir / "msa"
        predict_dir = wt_dir / "predict"
        wt_report_path = self.output_root / "wt_report.json"

        try:
            if wt_row["msa_status"] != "done" or not (msa_dir / "query_msa.json").exists():
                if self.profiler is not None:
                    self.profiler.record_stage("wt_msa", "start", source="panel_stand")
                self.db.set_wt_stage(
                    self.config.target_id,
                    "msa",
                    "running",
                    msa_dir=msa_dir,
                )
                query_msa_json = self._align_msa(wt_query_json, msa_dir, "wt")
                self.db.set_wt_stage(
                    self.config.target_id,
                    "msa",
                    "done",
                    msa_dir=msa_dir,
                    msa_query_json=query_msa_json,
                    last_error=None,
                )
                if self.profiler is not None:
                    self.profiler.record_stage("wt_msa", "end", source="panel_stand")
            else:
                query_msa_json = msa_dir / "query_msa.json"

            wt_row = self.db.fetch_wt(self.config.target_id)
            assert wt_row is not None
            if wt_row["predict_status"] != "done" or not (predict_dir / "summary.jsonl").exists():
                if self.profiler is not None:
                    self.profiler.record_stage("wt_predict", "start", source="panel_stand")
                self.db.set_wt_stage(
                    self.config.target_id,
                    "predict",
                    "running",
                    predict_dir=predict_dir,
                )
                summary_path = self._predict(query_msa_json, predict_dir, "wt")
                best_rows = _best_summary_rows(summary_path)
                best = best_rows[wt_query_id]
                confidence_path = Path(best["aggregated_confidence_path"])
                structure_path = _infer_structure_path(best["aggregated_confidence_path"])
                self.db.set_wt_stage(
                    self.config.target_id,
                    "predict",
                    "done",
                    predict_dir=predict_dir,
                    structure_path=structure_path,
                    confidence_path=confidence_path,
                    last_error=None,
                )
                if self.profiler is not None:
                    self.profiler.record_stage("wt_predict", "end", source="panel_stand")

            wt_row = self.db.fetch_wt(self.config.target_id)
            assert wt_row is not None
            if wt_row["analysis_status"] != "done" or not wt_report_path.exists():
                if self.profiler is not None:
                    self.profiler.record_stage("wt_analysis", "start", source="panel_stand")
                if wt_row["structure_path"] is None:
                    raise RuntimeError("WT predict stage is complete but structure_path is missing")
                case = BenchmarkCase(
                    case_id=f"{self.config.target_id}_WT",
                    structure_path=Path(wt_row["structure_path"]),
                    confidence_path=(
                        None
                        if wt_row["confidence_path"] is None
                        else Path(wt_row["confidence_path"])
                    ),
                    notes="WT baseline for panel ddG stand",
                )
                self.db.set_wt_stage(self.config.target_id, "analysis", "running")
                report = self.harness.run_case(case)
                DdgBenchmarkHarness.write_report(report, wt_report_path)
                rosetta_score = None
                for result in report.results:
                    if result.method == "rosetta_score" and result.score is not None:
                        rosetta_score = float(result.score)
                        break
                self.db.set_wt_stage(
                    self.config.target_id,
                    "analysis",
                    "done",
                    wt_report_path=wt_report_path,
                    rosetta_score=rosetta_score,
                    last_error=None,
                )
                if self.profiler is not None:
                    self.profiler.record_stage("wt_analysis", "end", source="panel_stand")
        except Exception as exc:
            self.db.set_wt_stage(self.config.target_id, "analysis", "failed", last_error=str(exc))
            raise

    def _ensure_wt_msa_only(self, wt_query_id: str, wt_query: Query) -> Path:
        self.db.upsert_wt(self.config.target_id, wt_query_id)
        wt_query_json = self._write_wt_query_json(wt_query_id, wt_query)
        wt_dir = self._wt_dir()
        msa_dir = wt_dir / "msa"
        try:
            if not (msa_dir / "query_msa.json").exists():
                if self.profiler is not None:
                    self.profiler.record_stage("wt_msa", "start", source="panel_stand")
                self.db.set_wt_stage(
                    self.config.target_id,
                    "msa",
                    "running",
                    msa_dir=msa_dir,
                )
                query_msa_json = self._align_msa(wt_query_json, msa_dir, "wt")
                self.db.set_wt_stage(
                    self.config.target_id,
                    "msa",
                    "done",
                    msa_dir=msa_dir,
                    msa_query_json=query_msa_json,
                    last_error=None,
                )
                if self.profiler is not None:
                    self.profiler.record_stage("wt_msa", "end", source="panel_stand")
                return query_msa_json
            query_msa_json = msa_dir / "query_msa.json"
            self.db.set_wt_stage(
                self.config.target_id,
                "msa",
                "done",
                msa_dir=msa_dir,
                msa_query_json=query_msa_json,
                last_error=None,
            )
            return query_msa_json
        except Exception as exc:
            self.db.set_wt_stage(self.config.target_id, "msa", "failed", last_error=str(exc))
            raise

    def _ensure_panel_msa(
        self,
        panel: MutationPanel,
        base_query: Query,
        *,
        wt_query_msa_json: Path | None = None,
    ) -> bool:
        panel_dir = self._panel_dir(panel.panel_id)
        query_json = self._write_panel_query_json(base_query, panel)
        self.db.upsert_panel(self.config.target_id, panel, panel_dir)
        row = self.db.fetch_panel(panel.panel_id)
        assert row is not None
        msa_dir = panel_dir / "msa"
        started = time.perf_counter()
        try:
            if self.profiler is not None:
                self.profiler.record_stage("panel_msa", "start", source=panel.panel_id)
            if row["msa_status"] == "done" and (msa_dir / "query_msa.json").exists():
                self._log(f"[resume] panel MSA ready {panel.panel_id}")
                return True
            self.db.set_panel_stage(panel.panel_id, "msa", "running", last_error=None)
            if self.config.reuse_wt_msa_for_mutants:
                if wt_query_msa_json is None:
                    raise RuntimeError("WT MSA reuse is enabled but WT query_msa.json is missing")
                query_msa_json = self._write_panel_query_msa_json(wt_query_msa_json, panel)
                self._log(f"[msa reuse] panel {panel.panel_id} <- {wt_query_msa_json}")
            else:
                query_msa_json = self._align_msa(query_json, msa_dir, panel.panel_id)
            self.db.set_panel_stage(
                panel.panel_id,
                "msa",
                "done",
                msa_query_json=query_msa_json,
                last_error=None,
            )
            if self.profiler is not None:
                self.profiler.record_stage("panel_msa", "end", source=panel.panel_id)
            self._panel_msa_seconds[panel.panel_id] = time.perf_counter() - started
            return True
        except Exception as exc:
            self.db.set_panel_stage(panel.panel_id, "msa", "failed", last_error=str(exc))
            self._log(f"[msa-failed] {panel.panel_id}: {exc}")
            return False

    def _predict_panel_batch(self, panel_ids: Iterable[str]) -> dict[str, Any]:
        panel_ids = [str(panel_id) for panel_id in panel_ids]
        if not panel_ids:
            return {
                "panel_ids": [],
                "job_ids": [],
                "predict_total_seconds": 0.0,
                "checkpoint_load_seconds": None,
                "gpu_inference_seconds": 0.0,
            }
        predict_batch_dir = self._next_batched_predict_dir()
        predict_batch_dir.mkdir(parents=True, exist_ok=True)
        merged_query_json = predict_batch_dir / "query_msa_merged.json"
        panel_rows: dict[str, sqlite3.Row] = {}
        jobs_by_panel: dict[str, list[sqlite3.Row]] = {}
        msa_query_paths: list[Path] = []
        for panel_id in panel_ids:
            panel_row = self.db.fetch_panel(panel_id)
            if panel_row is None:
                continue
            panel_rows[panel_id] = panel_row
            jobs_by_panel[panel_id] = self.db.list_jobs_for_panel(panel_id)
            msa_query_paths.append(Path(panel_row["msa_query_json"]))
            self.db.set_panel_stage(panel_id, "predict", "running", last_error=None)

        if not msa_query_paths:
            return {
                "panel_ids": [],
                "job_ids": [],
                "predict_total_seconds": 0.0,
                "checkpoint_load_seconds": None,
                "gpu_inference_seconds": 0.0,
            }

        try:
            batch_started = time.perf_counter()
            profiler_window_start = (
                None if self.profiler is None else self.profiler.relative_seconds()
            )
            if self.profiler is not None:
                self.profiler.record_stage(
                    "panel_batch_predict",
                    "start",
                    source="panel_stand",
                    details={"panel_count": len(panel_ids)},
                )
            merged_query_set = _merge_inference_query_sets(msa_query_paths)
            merged_query_json.write_text(
                merged_query_set.model_dump_json(indent=2),
                encoding="utf-8",
            )
            summary_path = self._predict(merged_query_json, predict_batch_dir, "panel_batch")
            predict_total_seconds = time.perf_counter() - batch_started
            best_rows = _best_summary_rows(summary_path)
            profiler_window_end = (
                None if self.profiler is None else self.profiler.relative_seconds()
            )
            checkpoint_load_seconds = (
                None
                if self.profiler is None or profiler_window_start is None or profiler_window_end is None
                else self.profiler.stage_duration_seconds(
                    "checkpoint_load",
                    source="predict:panel_batch",
                    since_relative_seconds=profiler_window_start,
                    until_relative_seconds=profiler_window_end,
                )
            )
            if self.profiler is not None:
                self.profiler.record_stage(
                    "panel_batch_predict",
                    "end",
                    source="panel_stand",
                    details={"panel_count": len(panel_ids), "summary_path": str(summary_path)},
                )
        except Exception as exc:
            for panel_id in panel_ids:
                self.db.set_panel_stage(panel_id, "predict", "failed", last_error=str(exc))
            self._log(f"[predict-batch-failed] {exc}")
            return {
                "panel_ids": panel_ids,
                "job_ids": self._job_ids_for_panels(panel_ids),
                "predict_total_seconds": 0.0,
                "checkpoint_load_seconds": None,
                "gpu_inference_seconds": 0.0,
            }

        for panel_id in panel_ids:
            jobs = jobs_by_panel.get(panel_id, [])
            queued_job_ids: list[str] = []
            failures: list[str] = []
            for job in jobs:
                best = best_rows.get(job["query_id"])
                if best is None:
                    self.db.update_job_predict(
                        job["job_id"],
                        status="failed",
                        structure_path=None,
                        confidence_path=None,
                        last_error=f"missing_summary_row:{job['query_id']}",
                    )
                    failures.append(str(job["job_id"]))
                    continue
                confidence_path = Path(best["aggregated_confidence_path"])
                structure_path = _infer_structure_path(best["aggregated_confidence_path"])
                if structure_path is None:
                    self.db.update_job_predict(
                        job["job_id"],
                        status="failed",
                        structure_path=None,
                        confidence_path=confidence_path,
                        last_error="structure_output_missing",
                    )
                    failures.append(str(job["job_id"]))
                    continue
                self.db.update_job_predict(
                    job["job_id"],
                    status="done",
                    structure_path=structure_path,
                    confidence_path=confidence_path,
                    last_error=None,
                )
                if job["analysis_status"] != "done":
                    queued_job_ids.append(str(job["job_id"]))

            if failures:
                self.db.set_panel_stage(
                    panel_id,
                    "predict",
                    "failed",
                    last_error=",".join(failures),
                )
            else:
                self.db.set_panel_stage(panel_id, "predict", "done", last_error=None)
            self._enqueue_analysis_jobs(queued_job_ids)
        checkpoint_load_value = (
            0.0 if checkpoint_load_seconds is None else float(checkpoint_load_seconds)
        )
        return {
            "panel_ids": panel_ids,
            "job_ids": self._job_ids_for_panels(panel_ids),
            "predict_total_seconds": float(predict_total_seconds),
            "checkpoint_load_seconds": checkpoint_load_seconds,
            "gpu_inference_seconds": max(0.0, float(predict_total_seconds) - checkpoint_load_value),
        }

    def _panel_ready_for_predict(self, panel_id: str) -> bool:
        panel_row = self.db.fetch_panel(panel_id)
        if panel_row is None:
            return False
        panel_dir = self._panel_dir(panel_id)
        msa_query_json = panel_dir / "msa" / "query_msa.json"
        jobs = self.db.list_jobs_for_panel(panel_id)
        if (
            panel_row["predict_status"] == "done"
            and all(
                job["analysis_status"] == "done"
                or (
                    job["structure_path"] is not None
                    and Path(job["structure_path"]).exists()
                )
                for job in jobs
            )
        ):
            self._log(f"[resume] panel predict ready {panel_id}")
            self._enqueue_analysis_jobs(
                job["job_id"] for job in jobs if job["analysis_status"] != "done"
            )
            return False
        return panel_row["msa_status"] == "done" and msa_query_json.exists()

    def _analysis_worker(self) -> None:
        self.analysis_allowed.wait()
        while True:
            job_id = self.analysis_queue.get()
            if job_id is None:
                self.analysis_queue.task_done()
                return
            try:
                self._analyze_job(str(job_id))
            finally:
                self.analysis_queue.task_done()

    def _analyze_job(self, job_id: str) -> None:
        job_row = self.db.fetch_job(job_id)
        wt_row = self.db.fetch_wt(self.config.target_id)
        if job_row is None or wt_row is None:
            return
        if job_row["analysis_status"] == "done":
            return
        if job_row["structure_path"] is None:
            self.db.update_job_analysis(
                job_id,
                status="failed",
                report_path=None,
                rosetta_delta_vs_wt=None,
                last_error="missing_structure_path",
            )
            return
        mutation = MutationInput(
            chain_id=str(job_row["chain_id"]),
            from_residue=str(job_row["from_residue"]),
            position_1based=int(job_row["position_1based"]),
            to_residue=str(job_row["to_residue"]),
        )
        case = BenchmarkCase(
            case_id=job_id,
            structure_path=Path(job_row["structure_path"]),
            confidence_path=(
                None
                if job_row["confidence_path"] is None
                else Path(job_row["confidence_path"])
            ),
            mutations=(mutation,),
            notes=f"Panel stand case for {job_id}",
        )
        report_path = self._report_path(job_id)
        try:
            if self.profiler is not None:
                self.profiler.record_stage("job_analysis", "start", source=job_id)
            started = time.perf_counter()
            report = self.harness.run_case(case)
            DdgBenchmarkHarness.write_report(report, report_path)
            self.db.replace_method_results(job_id, [asdict(result) for result in report.results])
            rosetta_score = None
            for result in report.results:
                if result.method == "rosetta_score" and result.score is not None:
                    rosetta_score = float(result.score)
                    break
            wt_rosetta = wt_row["rosetta_score"]
            rosetta_delta = None
            if rosetta_score is not None and wt_rosetta is not None:
                rosetta_delta = rosetta_score - float(wt_rosetta)
            self.db.update_job_analysis(
                job_id,
                status="done",
                report_path=report_path,
                rosetta_delta_vs_wt=rosetta_delta,
                last_error=None,
            )
            if self.profiler is not None:
                self.profiler.record_stage("job_analysis", "end", source=job_id)
            self._job_analysis_seconds[job_id] = time.perf_counter() - started
            self._maybe_cleanup_panel(str(job_row["panel_id"]))
        except Exception as exc:
            self.db.update_job_analysis(
                job_id,
                status="failed",
                report_path=None,
                rosetta_delta_vs_wt=None,
                last_error=str(exc),
            )
            self._log(f"[analysis-failed] {job_id}: {exc}")

    def _maybe_cleanup_panel(self, panel_id: str) -> None:
        if not self.config.cleanup_intermediates:
            return
        panel_row = self.db.fetch_panel(panel_id)
        if panel_row is None:
            return
        if panel_row["cleanup_status"] == "done":
            return
        if not self.db.panel_is_ready_for_cleanup(panel_id):
            return
        panel_dir = self._panel_dir(panel_id)
        for relative in ("msa", "predict"):
            path = panel_dir / relative
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        self.db.set_panel_stage(panel_id, "cleanup", "done", last_error=None)

    def _write_summary(
        self, profiling_artifacts: PanelProfilingArtifacts | None = None
    ) -> Path:
        payload = {
            "target_id": self.config.target_id,
            "mutable_chain_id": self.config.mutable_chain_id,
            "positions": list(self.config.positions),
            "summary": self.db.summary(),
            "profiling": (
                None
                if profiling_artifacts is None
                else {
                    "events_path": str(profiling_artifacts.events_path),
                    "samples_path": str(profiling_artifacts.samples_path),
                    "summary_path": str(profiling_artifacts.summary_path),
                    "timeline_svg_path": str(profiling_artifacts.timeline_svg_path),
                }
            ),
        }
        path = self.output_root / "summary.json"
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return path

    def _build_result_payload(
        self,
        *,
        panels: list[MutationPanel],
        profiling_artifacts: PanelProfilingArtifacts | None,
        run_mode: str,
        chosen_predict_mode: str | None = None,
        warmup_cpu_seconds: float | None = None,
        warmup_gpu_seconds: float | None = None,
        warmup_checkpoint_load_seconds: float | None = None,
    ) -> dict[str, Any]:
        summary_path = self._write_summary(profiling_artifacts=profiling_artifacts)
        return {
            "run_mode": run_mode,
            "output_root": str(self.output_root),
            "state_db": str(self.output_root / "state.sqlite"),
            "summary_path": str(summary_path),
            "panel_count": len(panels),
            "mutant_job_count": sum(len(panel.job_ids) for panel in panels),
            "predict_strategy_requested": self.config.predict_strategy,
            "predict_strategy_selected": chosen_predict_mode,
            "predict_panel_chunk_size": self.config.predict_panel_chunk_size,
            "warmup_cpu_seconds": warmup_cpu_seconds,
            "warmup_gpu_seconds": warmup_gpu_seconds,
            "warmup_checkpoint_load_seconds": warmup_checkpoint_load_seconds,
            "profiling_summary_path": (
                None if profiling_artifacts is None else str(profiling_artifacts.summary_path)
            ),
            "profiling_events_path": (
                None if profiling_artifacts is None else str(profiling_artifacts.events_path)
            ),
            "profiling_samples_path": (
                None if profiling_artifacts is None else str(profiling_artifacts.samples_path)
            ),
            "profiling_timeline_svg_path": (
                None
                if profiling_artifacts is None
                else str(profiling_artifacts.timeline_svg_path)
            ),
        }

    def prepare_inputs(self) -> dict[str, Any]:
        wt_query_id, wt_query, panels = self._build_panels()
        wt_query_msa_json = self._ensure_wt_msa_only(wt_query_id, wt_query)
        for panel in panels:
            self.db.upsert_panel(
                self.config.target_id,
                panel,
                self._panel_dir(panel.panel_id),
            )
        if self.config.msa_panel_workers <= 1:
            for panel in panels:
                self._log(f"[prepare {panels.index(panel) + 1}/{len(panels)}] {panel.panel_id}")
                self._ensure_panel_msa(
                    panel,
                    wt_query,
                    wt_query_msa_json=wt_query_msa_json,
                )
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(
                max_workers=max(1, self.config.msa_panel_workers)
            ) as executor:
                futures = {
                    executor.submit(
                        self._ensure_panel_msa,
                        panel,
                        wt_query,
                        wt_query_msa_json=wt_query_msa_json,
                    ): panel
                    for panel in panels
                }
                for future in as_completed(futures):
                    future.result()
        return self._build_result_payload(
            panels=panels,
            profiling_artifacts=None,
            run_mode="prepare",
        )

    def _run_predict_and_analysis(self) -> tuple[
        list[MutationPanel], str | None, float | None, float | None, float | None
    ]:
        wt_query_id, wt_query, panels = self._build_panels()
        self._ensure_wt(wt_query_id, wt_query)
        wt_row = self.db.fetch_wt(self.config.target_id)
        if wt_row is None or wt_row["msa_query_json"] is None:
            raise RuntimeError("WT MSA stage did not produce msa_query_json")
        wt_query_msa_json = Path(wt_row["msa_query_json"])
        for panel in panels:
            self.db.upsert_panel(
                self.config.target_id,
                panel,
                self._panel_dir(panel.panel_id),
            )

        chosen_predict_mode: str | None = None
        warmup_cpu_seconds: float | None = None
        warmup_gpu_seconds: float | None = None
        warmup_checkpoint_load_seconds: float | None = None
        analysis_threads = [
            threading.Thread(target=self._analysis_worker, daemon=True)
            for _ in range(max(1, self.config.analysis_workers))
        ]
        for thread in analysis_threads:
            thread.start()
        self.analysis_allowed.set()

        predict_panel_chunk_size = self._effective_predict_panel_chunk_size(len(panels))
        pending_after_warmup: list[str] = []
        warmup_chunk_dispatched = False

        def _dispatch_or_buffer_ready_panel(panel_id: str) -> None:
            nonlocal warmup_chunk_dispatched
            nonlocal chosen_predict_mode
            nonlocal warmup_cpu_seconds
            nonlocal warmup_gpu_seconds
            nonlocal warmup_checkpoint_load_seconds
            if not self._panel_ready_for_predict(panel_id):
                return
            if chosen_predict_mode == "single_batch":
                pending_after_warmup.append(panel_id)
                return
            ready_for_predict.append(panel_id)
            if warmup_chunk_dispatched and chosen_predict_mode == "chunked":
                if len(ready_for_predict) >= predict_panel_chunk_size:
                    chunk = list(ready_for_predict[:predict_panel_chunk_size])
                    del ready_for_predict[:predict_panel_chunk_size]
                    self._log(
                        f"[predict batch] launching chunk for {len(chunk)} panels"
                    )
                    self._predict_panel_batch(chunk)
                return
            if not warmup_chunk_dispatched and len(ready_for_predict) >= predict_panel_chunk_size:
                first_chunk = list(ready_for_predict[:predict_panel_chunk_size])
                del ready_for_predict[:predict_panel_chunk_size]
                first_chunk_job_ids = self._job_ids_for_panels(first_chunk)
                warmup_msa_seconds = sum(
                    float(self._panel_msa_seconds.get(chunk_panel_id, 0.0))
                    for chunk_panel_id in first_chunk
                )
                self._log(
                    f"[predict warmup] launching first chunk for {len(first_chunk)} panels"
                )
                predict_metrics = self._predict_panel_batch(first_chunk)
                analysis_started = time.perf_counter()
                self._wait_for_analysis_completion(first_chunk_job_ids)
                warmup_analysis_seconds = time.perf_counter() - analysis_started
                warmup_checkpoint_load_seconds = predict_metrics["checkpoint_load_seconds"]
                checkpoint_load_value = (
                    0.0
                    if warmup_checkpoint_load_seconds is None
                    else float(warmup_checkpoint_load_seconds)
                )
                warmup_gpu_seconds = float(predict_metrics["gpu_inference_seconds"])
                warmup_cpu_seconds = (
                    warmup_msa_seconds + checkpoint_load_value + warmup_analysis_seconds
                )
                chosen_predict_mode = self._select_predict_mode(
                    first_chunk_cpu_seconds=warmup_cpu_seconds,
                    first_chunk_gpu_seconds=warmup_gpu_seconds,
                )
                warmup_chunk_dispatched = True
                self._log(
                    "[predict strategy] "
                    f"{chosen_predict_mode} "
                    f"(warmup cpu={warmup_cpu_seconds:.2f}s, "
                    f"gpu={warmup_gpu_seconds:.2f}s, "
                    f"checkpoint_load={checkpoint_load_value:.2f}s)"
                )
                if chosen_predict_mode == "single_batch":
                    pending_after_warmup.extend(ready_for_predict)
                    ready_for_predict.clear()

        ready_for_predict: list[str] = []
        if self.config.msa_panel_workers <= 1:
            for index, panel in enumerate(panels, start=1):
                self._log(f"[msa {index}/{len(panels)}] {panel.panel_id}")
                if self._ensure_panel_msa(
                    panel,
                    wt_query,
                    wt_query_msa_json=wt_query_msa_json,
                ):
                    _dispatch_or_buffer_ready_panel(panel.panel_id)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(
                max_workers=max(1, self.config.msa_panel_workers)
            ) as executor:
                futures = {
                    executor.submit(
                        self._ensure_panel_msa,
                        panel,
                        wt_query,
                        wt_query_msa_json=wt_query_msa_json,
                    ): panel
                    for panel in panels
                }
                for future in as_completed(futures):
                    panel = futures[future]
                    if future.result():
                        _dispatch_or_buffer_ready_panel(panel.panel_id)

        if chosen_predict_mode is None:
            chosen_predict_mode = "single_batch"

        if chosen_predict_mode == "single_batch":
            remaining_panel_ids = list(pending_after_warmup) + list(ready_for_predict)
            if remaining_panel_ids:
                self._log(
                    f"[predict batch] launching single remaining batch for {len(remaining_panel_ids)} panels"
                )
                self._predict_panel_batch(remaining_panel_ids)
        elif ready_for_predict:
            self._log(
                f"[predict batch] launching final chunk for {len(ready_for_predict)} panels"
            )
            self._predict_panel_batch(ready_for_predict)
        self._drain_deferred_analysis_jobs()
        for _ in analysis_threads:
            self.analysis_queue.put(None)
        for thread in analysis_threads:
            thread.join()
        return (
            panels,
            chosen_predict_mode,
            warmup_cpu_seconds,
            warmup_gpu_seconds,
            warmup_checkpoint_load_seconds,
        )

    def run_predict_and_analysis(self) -> dict[str, Any]:
        profiling_artifacts: PanelProfilingArtifacts | None = None
        if self.profiler is not None:
            self.profiler.start()
        try:
            (
                panels,
                chosen_predict_mode,
                warmup_cpu_seconds,
                warmup_gpu_seconds,
                warmup_checkpoint_load_seconds,
            ) = self._run_predict_and_analysis()
        except Exception:
            if self.profiler is not None:
                profiling_artifacts = self.profiler.stop()
            raise

        if self.profiler is not None:
            profiling_artifacts = self.profiler.stop()
        return self._build_result_payload(
            panels=panels,
            profiling_artifacts=profiling_artifacts,
            run_mode="resume",
            chosen_predict_mode=chosen_predict_mode,
            warmup_cpu_seconds=warmup_cpu_seconds,
            warmup_gpu_seconds=warmup_gpu_seconds,
            warmup_checkpoint_load_seconds=warmup_checkpoint_load_seconds,
        )

    def run(self) -> dict[str, Any]:
        self.prepare_inputs()
        payload = self.run_predict_and_analysis()
        payload["run_mode"] = "full"
        return payload
