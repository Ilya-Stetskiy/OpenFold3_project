from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


@dataclass(frozen=True, slots=True)
class StructurePaths:
    experimental: str | None
    openfold: str | None
    foldx: str | None


@dataclass(frozen=True, slots=True)
class CanonicalMutationRecord:
    protein_id: str
    sequence: str
    mutation: str
    position: int
    wt: str
    mut: str
    structure_paths: StructurePaths
    experimental_ddg: float | None

    @property
    def mutated_sequence(self) -> str:
        return self.sequence[: self.position - 1] + self.mut + self.sequence[self.position :]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["structure_paths"] = asdict(self.structure_paths)
        return payload


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"Missing required field: {key}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Empty required field: {key}")
    return text


def _normalize_aa(value: str, key: str) -> str:
    aa = str(value).strip().upper()
    if aa not in CANONICAL_AA:
        raise ValueError(f"{key} must be a canonical amino acid, got {value!r}")
    return aa


def parse_record(payload: dict[str, Any]) -> CanonicalMutationRecord:
    protein_id = _require_str(payload, "protein_id")
    sequence = _require_str(payload, "sequence").upper()
    mutation = _require_str(payload, "mutation").upper()
    match = MUTATION_RE.match(mutation)
    if match is None:
        raise ValueError(f"Invalid mutation format: {mutation}")
    wt, position_text, mut = match.groups()
    position = int(position_text)
    explicit_position_raw = payload.get("position")
    if explicit_position_raw is None:
        raise ValueError("Missing required field: position")
    explicit_position = int(explicit_position_raw)
    if explicit_position != position:
        raise ValueError(
            f"Mutation string position {position} disagrees with explicit position {explicit_position}"
        )
    if position < 1 or position > len(sequence):
        raise ValueError(
            f"Mutation position {position} is outside sequence length {len(sequence)}"
        )
    wt = _normalize_aa(payload.get("wt", wt), "wt")
    mut = _normalize_aa(payload.get("mut", mut), "mut")
    if wt != match.group(1) or mut != match.group(3):
        raise ValueError("Mutation string and explicit wt/mut fields disagree")

    structure_paths_payload = payload.get("structure_paths")
    if not isinstance(structure_paths_payload, dict):
        raise ValueError("structure_paths must be an object")
    structure_paths = StructurePaths(
        experimental=_optional_path(structure_paths_payload.get("experimental")),
        openfold=_optional_path(structure_paths_payload.get("openfold")),
        foldx=_optional_path(structure_paths_payload.get("foldx")),
    )
    experimental_ddg_raw = payload.get("experimental_ddg")
    experimental_ddg = None if experimental_ddg_raw is None else float(experimental_ddg_raw)

    record = CanonicalMutationRecord(
        protein_id=protein_id,
        sequence=sequence,
        mutation=mutation,
        position=position,
        wt=wt,
        mut=mut,
        structure_paths=structure_paths,
        experimental_ddg=experimental_ddg,
    )
    validate_record(record)
    return record


def _optional_path(value: Any) -> str | None:
    if value in {None, "", "null"}:
        return None
    return str(value)


def validate_record(record: CanonicalMutationRecord) -> CanonicalMutationRecord:
    if record.position < 1 or record.position > len(record.sequence):
        raise ValueError(
            f"Position {record.position} is outside sequence length {len(record.sequence)}"
        )
    observed = record.sequence[record.position - 1]
    if observed != record.wt:
        raise ValueError(
            f"Sequence residue mismatch for {record.mutation}: expected {record.wt}, found {observed}"
        )
    expected_mutation = f"{record.wt}{record.position}{record.mut}"
    if record.mutation != expected_mutation:
        raise ValueError(
            f"Mutation field {record.mutation} disagrees with parsed components {expected_mutation}"
        )
    return record


def load_canonical_records(path: str | Path) -> list[CanonicalMutationRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Canonical dataset must be a JSON list")
    return [parse_record(item) for item in payload]


def write_canonical_records(path: str | Path, records: list[CanonicalMutationRecord]) -> None:
    Path(path).write_text(
        json.dumps([record.to_dict() for record in records], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def demo_records() -> list[CanonicalMutationRecord]:
    return [
        parse_record({
            "protein_id": "demo-1",
            "sequence": "ACDEFGHIK",
            "mutation": "D3N",
            "position": 3,
            "wt": "D",
            "mut": "N",
            "structure_paths": {"experimental": None, "openfold": None, "foldx": None},
            "experimental_ddg": -1.2,
        }),
        parse_record({
            "protein_id": "demo-2",
            "sequence": "MKTLLA",
            "mutation": "T3A",
            "position": 3,
            "wt": "T",
            "mut": "A",
            "structure_paths": {"experimental": None, "openfold": None, "foldx": None},
            "experimental_ddg": 0.5,
        }),
        parse_record({
            "protein_id": "demo-3",
            "sequence": "QQQLLMN",
            "mutation": "L5P",
            "position": 5,
            "wt": "L",
            "mut": "P",
            "structure_paths": {"experimental": None, "openfold": None, "foldx": None},
            "experimental_ddg": None,
        }),
    ]
