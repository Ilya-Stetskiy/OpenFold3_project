from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..data.canonical import CanonicalMutationRecord


@dataclass(frozen=True, slots=True)
class BackendRunResult:
    status: str
    raw_output_path: Path
    error_message: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AdapterResult:
    method: str
    representation: str
    raw_value: float | None
    ddg: float | None
    ddg_std: float | None
    n_runs_requested: int | None
    n_runs_valid: int | None
    per_run_ddg: tuple[float, ...]
    success: bool
    status: str
    error_message: str | None
    units: str | None
    structure_source: str
    input_path: Path
    raw_output_path: Path
    normalized_output_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        payload["raw_output_path"] = str(self.raw_output_path)
        payload["normalized_output_path"] = str(self.normalized_output_path)
        return payload


class DDGAdapter(ABC):
    method_name: str
    sequence_based: bool = False

    def validate_environment(self) -> None:
        return None

    @abstractmethod
    def prepare_input(self, record: CanonicalMutationRecord, structure_source: str, work_dir: Path) -> Path: ...

    @abstractmethod
    def run(
        self,
        prepared_input_path: Path,
        record: CanonicalMutationRecord,
        structure_source: str,
        work_dir: Path,
    ) -> BackendRunResult: ...

    @abstractmethod
    def parse_output(self, raw_output_path: Path) -> float | None: ...

    @abstractmethod
    def normalize(self, raw_value: float | None) -> float | None: ...

    def extract_summary(self, raw_output_path: Path) -> dict[str, Any]:
        return {}

    def representation(self) -> str:
        return "sequence" if self.sequence_based else "structure"

    def result_unit(self, ddg: float | None) -> str | None:
        return "kcal/mol" if ddg is not None else None

    def normalization_details(self, raw_output_path: Path) -> dict[str, Any]:
        return {}

    def predict(self, record: CanonicalMutationRecord, structure_source: str, work_dir: Path) -> AdapterResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = self.prepare_input(record, structure_source, work_dir)
        backend_result = self.run(input_path, record, structure_source, work_dir)
        raw_value = self.parse_output(backend_result.raw_output_path)
        summary = self.extract_summary(backend_result.raw_output_path)
        ddg = self.normalize(raw_value)
        result_unit = self.result_unit(ddg)
        success = backend_result.status == "ok" and ddg is not None
        error_message = backend_result.error_message
        if backend_result.status == "ok" and ddg is None:
            success = False
            error_message = "Backend returned ok without numeric ddG"
        normalized_output_path = work_dir / "normalized_result.json"
        normalized_output_path.write_text(
            json.dumps(
                {
                    "method": self.method_name,
                    "representation": self.representation(),
                    "raw_value": raw_value,
                    "ddg": ddg,
                    "units": result_unit,
                    "status": backend_result.status,
                    "error_message": error_message,
                    **self.normalization_details(backend_result.raw_output_path),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return AdapterResult(
            method=self.method_name,
            representation=self.representation(),
            raw_value=raw_value,
            ddg=ddg,
            ddg_std=None if summary.get("ddg_std") is None else float(summary["ddg_std"]),
            n_runs_requested=None if summary.get("n_runs_requested") is None else int(summary["n_runs_requested"]),
            n_runs_valid=None if summary.get("n_runs_valid") is None else int(summary["n_runs_valid"]),
            per_run_ddg=tuple(float(value) for value in summary.get("per_run_ddg", ())),
            success=success,
            status=backend_result.status,
            error_message=error_message,
            units=result_unit,
            structure_source=structure_source,
            input_path=input_path,
            raw_output_path=backend_result.raw_output_path,
            normalized_output_path=normalized_output_path,
        )
