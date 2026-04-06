from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from openfold3.benchmark.models import BenchmarkCase

DatasetKind = Literal["benchmark", "exploratory"]


@dataclass(frozen=True)
class TestbenchConfig:
    output_root: Path
    dataset_kind: DatasetKind = "exploratory"
    gpu_concurrency: int = 1
    cpu_prep_workers: int = 1
    cpu_ddg_workers: int = 1
    registry_path: Path | None = None
    notes: str | None = None

    def resolved(self) -> "TestbenchConfig":
        output_root = self.output_root.resolve()
        registry_path = (
            self.registry_path.resolve()
            if self.registry_path is not None
            else (output_root / "registry.sqlite")
        )
        return TestbenchConfig(
            output_root=output_root,
            dataset_kind=self.dataset_kind,
            gpu_concurrency=self.gpu_concurrency,
            cpu_prep_workers=self.cpu_prep_workers,
            cpu_ddg_workers=self.cpu_ddg_workers,
            registry_path=registry_path,
            notes=self.notes,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CaseExecutionRecord:
    case: BenchmarkCase
    report_path: Path
    dataset_kind: DatasetKind
    method_statuses: dict[str, str] = field(default_factory=dict)

