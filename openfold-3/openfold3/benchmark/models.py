from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MutationInput:
    chain_id: str
    from_residue: str
    position_1based: int
    to_residue: str

    @property
    def mutation_id(self) -> str:
        return (
            f"{self.chain_id}_{self.from_residue.upper()}"
            f"{self.position_1based}{self.to_residue.upper()}"
        )


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    structure_path: Path
    confidence_path: Path | None = None
    mutations: tuple[MutationInput, ...] = ()
    chain_groups: tuple[tuple[str, ...], ...] = ()
    experimental_ddg: float | None = None
    notes: str | None = None
    pdb_id: str | None = None

    def with_resolved_paths(self) -> "BenchmarkCase":
        return BenchmarkCase(
            case_id=self.case_id,
            structure_path=self.structure_path.resolve(),
            confidence_path=(
                None if self.confidence_path is None else self.confidence_path.resolve()
            ),
            mutations=self.mutations,
            chain_groups=self.chain_groups,
            experimental_ddg=self.experimental_ddg,
            notes=self.notes,
            pdb_id=self.pdb_id,
        )


@dataclass(frozen=True)
class AtomRecord:
    chain_id: str
    residue_name: str
    residue_id: str
    atom_name: str
    x: float
    y: float
    z: float
    b_factor: float | None = None
    group_pdb: str | None = None


@dataclass(frozen=True)
class ResidueRecord:
    chain_id: str
    residue_name: str
    residue_id: str
    atom_names: tuple[str, ...]
    ca_coord: tuple[float, float, float] | None = None
    b_factor_mean: float | None = None


@dataclass(frozen=True)
class StructureSummary:
    atom_count: int
    residue_count: int
    chain_ids: tuple[str, ...]
    residues_by_chain: dict[str, tuple[ResidueRecord, ...]]
    inferred_chain_groups: tuple[tuple[str, ...], ...]
    min_inter_chain_atom_distance: float | None
    interface_atom_contacts_5a: int
    interface_ca_contacts_8a: int
    chain_pair_min_distances: dict[str, float] = field(default_factory=dict)
