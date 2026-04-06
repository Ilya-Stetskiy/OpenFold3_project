from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from openfold3.benchmark.harness import HarnessReport

from .models import DatasetKind, TestbenchConfig


class SQLiteRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                output_root TEXT NOT NULL,
                dataset_kind TEXT NOT NULL,
                gpu_concurrency INTEGER NOT NULL,
                cpu_prep_workers INTEGER NOT NULL,
                cpu_ddg_workers INTEGER NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS cases (
                case_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                case_id TEXT NOT NULL,
                dataset_kind TEXT NOT NULL,
                structure_path TEXT NOT NULL,
                confidence_path TEXT,
                experimental_ddg REAL,
                notes TEXT,
                report_path TEXT NOT NULL,
                structure_summary_json TEXT NOT NULL,
                UNIQUE(run_id, case_id),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS method_runs (
                method_run_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                case_pk INTEGER NOT NULL,
                method TEXT NOT NULL,
                status TEXT NOT NULL,
                score REAL,
                units TEXT,
                details_json TEXT NOT NULL,
                UNIQUE(case_pk, method),
                FOREIGN KEY(case_pk) REFERENCES cases(case_pk)
            );

            CREATE TABLE IF NOT EXISTS stage_runs (
                stage_run_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                case_pk INTEGER NOT NULL,
                stage_name TEXT NOT NULL,
                status TEXT NOT NULL,
                runtime_seconds REAL,
                details_json TEXT NOT NULL,
                FOREIGN KEY(case_pk) REFERENCES cases(case_pk)
            );
            """
        )
        self.conn.commit()

    def create_run(self, config: TestbenchConfig) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO runs (
                output_root, dataset_kind, gpu_concurrency,
                cpu_prep_workers, cpu_ddg_workers, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(config.output_root),
                config.dataset_kind,
                config.gpu_concurrency,
                config.cpu_prep_workers,
                config.cpu_ddg_workers,
                config.notes,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def insert_case_report(
        self,
        run_id: int,
        dataset_kind: DatasetKind,
        report: HarnessReport,
        report_path: Path,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT OR REPLACE INTO cases (
                run_id, case_id, dataset_kind, structure_path, confidence_path,
                experimental_ddg, notes, report_path, structure_summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                report.case_id,
                dataset_kind,
                report.structure_path,
                report.confidence_path,
                report.experimental_ddg,
                report.notes,
                str(report_path),
                json.dumps(report.structure_summary, sort_keys=True),
            ),
        )
        self.conn.commit()
        case_pk = int(cur.lastrowid)
        if case_pk == 0:
            row = self.conn.execute(
                "SELECT case_pk FROM cases WHERE run_id = ? AND case_id = ?",
                (run_id, report.case_id),
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to resolve case_pk after insert")
            case_pk = int(row["case_pk"])
        for result in report.results:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO method_runs (
                    case_pk, method, status, score, units, details_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    case_pk,
                    result.method,
                    result.status,
                    result.score,
                    result.units,
                    json.dumps(result.details, sort_keys=True),
                ),
            )
            if isinstance(result.details, dict):
                if "runtime_seconds" in result.details:
                    self.conn.execute(
                        """
                        INSERT INTO stage_runs (
                            case_pk, stage_name, status, runtime_seconds, details_json
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            case_pk,
                            f"method:{result.method}",
                            result.status,
                            result.details.get("runtime_seconds"),
                            json.dumps(result.details, sort_keys=True),
                        ),
                    )
                for stage_key in (
                    "pdb_prepare_runtime_seconds",
                    "buildmodel_runtime_seconds",
                    "analyse_mutant_runtime_seconds",
                    "analyse_wt_runtime_seconds",
                ):
                    if stage_key in result.details:
                        self.conn.execute(
                            """
                            INSERT INTO stage_runs (
                                case_pk, stage_name, status, runtime_seconds, details_json
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                case_pk,
                                stage_key.removesuffix("_runtime_seconds"),
                                result.status,
                                result.details.get(stage_key),
                                json.dumps({"method": result.method}, sort_keys=True),
                            ),
                        )
        self.conn.commit()
        return case_pk

    def fetch_method_rows(self, run_id: int) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT
                    r.run_id,
                    c.case_id,
                    c.dataset_kind,
                    c.experimental_ddg,
                    m.method,
                    m.status,
                    m.score,
                    m.units,
                    m.details_json
                FROM method_runs m
                JOIN cases c ON c.case_pk = m.case_pk
                JOIN runs r ON r.run_id = c.run_id
                WHERE r.run_id = ?
                ORDER BY c.case_id, m.method
                """,
                (run_id,),
            )
        )

    def fetch_run(self, run_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

    def list_case_ids(self, run_id: int) -> Iterable[str]:
        rows = self.conn.execute(
            "SELECT case_id FROM cases WHERE run_id = ? ORDER BY case_id",
            (run_id,),
        )
        for row in rows:
            yield str(row["case_id"])

    def fetch_stage_rows(self, run_id: int) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT
                    c.case_id,
                    s.stage_name,
                    s.status,
                    s.runtime_seconds,
                    s.details_json
                FROM stage_runs s
                JOIN cases c ON c.case_pk = s.case_pk
                WHERE c.run_id = ?
                ORDER BY c.case_id, s.stage_name
                """,
                (run_id,),
            )
        )
