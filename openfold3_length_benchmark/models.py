from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(slots=True, frozen=True)
class ProteinChainRecord:
    entity_id: str
    chain_id: str
    sequence: str
    length: int
    description: str | None = None
    label_chain_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class ExcludedComponent:
    entity_id: str
    category: str
    description: str
    chain_ids: tuple[str, ...] = ()

    def summary_label(self) -> str:
        chain_part = ""
        if self.chain_ids:
            chain_part = f" chains={','.join(self.chain_ids)}"
        return f"{self.category}:{self.entity_id}:{self.description}{chain_part}"


@dataclass(slots=True)
class EntryComposition:
    pdb_id: str
    source_path: Path | None
    molecules: list[dict]
    protein_chains: list[ProteinChainRecord]
    chain_lengths: dict[str, int]
    total_protein_length: int
    excluded_components: list[ExcludedComponent] = field(default_factory=list)
    status: str = "ok"
    issue: str | None = None

    @property
    def chain_count(self) -> int:
        return len(self.chain_lengths)

    @property
    def chain_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.chain_lengths))

    @property
    def molecule_count(self) -> int:
        return len(self.molecules)

    def to_preview_row(self) -> dict[str, object]:
        return {
            "pdb_id": self.pdb_id,
            "status": self.status,
            "total_protein_length": self.total_protein_length,
            "chain_count": self.chain_count,
            "chain_ids": ",".join(self.chain_ids),
            "chain_lengths": ",".join(
                f"{chain_id}:{self.chain_lengths[chain_id]}" for chain_id in self.chain_ids
            ),
            "molecule_count": self.molecule_count,
            "excluded_component_count": len(self.excluded_components),
            "excluded_components": "; ".join(
                component.summary_label() for component in self.excluded_components
            ),
            "reference_path": None if self.source_path is None else str(self.source_path),
            "failure_reason": self.issue,
        }

    @classmethod
    def failed(cls, pdb_id: str, issue: str) -> "EntryComposition":
        return cls(
            pdb_id=pdb_id,
            source_path=None,
            molecules=[],
            protein_chains=[],
            chain_lengths={},
            total_protein_length=0,
            excluded_components=[],
            status="failed",
            issue=issue,
        )


@dataclass(slots=True)
class BenchmarkRunResult:
    run_name: str
    run_root: Path
    preview_df: pd.DataFrame
    results_df: pd.DataFrame
    failures_df: pd.DataFrame
    summary: dict[str, object]
    summary_path: Path
    results_csv_path: Path
    results_json_path: Path
    failures_csv_path: Path
    plot_paths: dict[str, Path]
