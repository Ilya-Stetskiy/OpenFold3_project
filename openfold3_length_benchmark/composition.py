from __future__ import annotations

import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from biotite.structure.io import pdbx

from .interop import default_mmcif_cache_dir
from .models import EntryComposition, ExcludedComponent, ProteinChainRecord


PROTEIN_POLYMER_TYPES = {"polypeptide(L)", "polypeptide(D)"}
DNA_POLYMER_TYPES = {"polydeoxyribonucleotide"}
RNA_POLYMER_TYPES = {"polyribonucleotide"}
HYBRID_POLYMER_TYPES = {"polydeoxyribonucleotide/polyribonucleotide hybrid"}


def normalize_pdb_id(value: str) -> str:
    cleaned = str(value).strip().upper()
    if not cleaned:
        raise ValueError("PDB ID is empty")
    if not re.fullmatch(r"[A-Z0-9]{4}", cleaned):
        raise ValueError(f"Unsupported PDB ID format: {value!r}")
    return cleaned


def parse_pdb_ids(pdb_ids: str | Iterable[str], max_entries: int | None = None) -> list[str]:
    if isinstance(pdb_ids, str):
        raw_items = [item for item in re.split(r"[\s,;]+", pdb_ids) if item.strip()]
    else:
        raw_items = []
        for item in pdb_ids:
            raw_items.extend([part for part in re.split(r"[\s,;]+", str(item)) if part.strip()])

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        pdb_id = normalize_pdb_id(raw_item)
        if pdb_id in seen:
            continue
        seen.add(pdb_id)
        normalized.append(pdb_id)

    if max_entries is not None:
        normalized = normalized[: max(0, int(max_entries))]
    return normalized


def download_mmcif(
    pdb_id: str,
    *,
    cache_dir: str | Path | None = None,
    session: requests.Session | None = None,
    force: bool = False,
) -> Path:
    normalized_id = normalize_pdb_id(pdb_id)
    cache_root = Path(cache_dir or default_mmcif_cache_dir()).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    canonical_path = cache_root / f"{normalized_id}.cif"
    lower_path = cache_root / f"{normalized_id.lower()}.cif"
    for candidate in (canonical_path, lower_path):
        if candidate.exists() and not force:
            if candidate != canonical_path:
                shutil.copy2(candidate, canonical_path)
            return canonical_path

    client = session or requests.Session()
    response = client.get(
        f"https://files.rcsb.org/download/{normalized_id}.cif",
        timeout=60,
    )
    response.raise_for_status()
    canonical_path.write_text(response.text, encoding="utf-8")
    return canonical_path


def _first_cif_block(cif_path: Path) -> pdbx.CIFBlock:
    cif_file = pdbx.CIFFile.read(cif_path)
    return cif_file[list(cif_file.keys())[0]]


def _clean_sequence(raw_sequence: str) -> str:
    return "".join(str(raw_sequence).split())


def _split_chain_ids(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def _entity_metadata(block: pdbx.CIFBlock) -> dict[str, dict[str, str]]:
    if "entity" not in block:
        return {}

    entity = block["entity"]
    entity_ids = entity["id"].as_array()
    entity_types = entity["type"].as_array()
    descriptions = (
        entity["pdbx_description"].as_array()
        if "pdbx_description" in entity
        else ["?"] * len(entity_ids)
    )

    metadata: dict[str, dict[str, str]] = {}
    for entity_id, entity_type, description in zip(
        entity_ids,
        entity_types,
        descriptions,
        strict=True,
    ):
        metadata[str(entity_id)] = {
            "entity_type": str(entity_type),
            "description": str(description),
        }
    return metadata


def _struct_asym_by_entity(block: pdbx.CIFBlock) -> dict[str, tuple[str, ...]]:
    if "struct_asym" not in block:
        return {}

    struct_asym = block["struct_asym"]
    entity_ids = struct_asym["entity_id"].as_array()
    chain_ids = struct_asym["id"].as_array()

    collected: dict[str, list[str]] = defaultdict(list)
    for entity_id, chain_id in zip(entity_ids, chain_ids, strict=True):
        collected[str(entity_id)].append(str(chain_id))
    return {entity_id: tuple(chain_ids) for entity_id, chain_ids in collected.items()}


def _classify_polymer(polymer_type: str) -> str:
    if polymer_type in DNA_POLYMER_TYPES:
        return "dna_polymer"
    if polymer_type in RNA_POLYMER_TYPES:
        return "rna_polymer"
    if polymer_type in HYBRID_POLYMER_TYPES:
        return "hybrid_polymer"
    return "other_polymer"


def extract_entry_composition(
    cif_path: str | Path,
    *,
    pdb_id: str | None = None,
) -> EntryComposition:
    resolved_path = Path(cif_path).expanduser().resolve()
    normalized_id = normalize_pdb_id(pdb_id or resolved_path.stem)
    block = _first_cif_block(resolved_path)
    entity_meta = _entity_metadata(block)
    label_chain_ids_by_entity = _struct_asym_by_entity(block)

    protein_chains: list[ProteinChainRecord] = []
    excluded_components: list[ExcludedComponent] = []
    grouped_sequences: dict[str, dict[str, object]] = {}
    polymer_entity_ids: set[str] = set()

    if "entity_poly" in block:
        entity_poly = block["entity_poly"]
        entity_ids = entity_poly["entity_id"].as_array()
        polymer_types = entity_poly["type"].as_array()
        sequences = entity_poly["pdbx_seq_one_letter_code_can"].as_array()
        strand_ids = entity_poly["pdbx_strand_id"].as_array()

        for entity_id, polymer_type, sequence, strands_raw in zip(
            entity_ids,
            polymer_types,
            sequences,
            strand_ids,
            strict=True,
        ):
            entity_id_str = str(entity_id)
            polymer_type_str = str(polymer_type)
            polymer_entity_ids.add(entity_id_str)
            sequence_clean = _clean_sequence(sequence)
            chain_ids = _split_chain_ids(strands_raw)
            label_chain_ids = label_chain_ids_by_entity.get(entity_id_str, ())
            if not chain_ids:
                chain_ids = list(label_chain_ids)

            description = entity_meta.get(entity_id_str, {}).get("description")
            if polymer_type_str in PROTEIN_POLYMER_TYPES:
                if not sequence_clean:
                    raise ValueError(
                        f"PDB entry {normalized_id} protein entity {entity_id_str} has empty sequence"
                    )
                if not chain_ids:
                    raise ValueError(
                        f"PDB entry {normalized_id} protein entity {entity_id_str} has no chain IDs"
                    )
                for chain_id in chain_ids:
                    protein_chains.append(
                        ProteinChainRecord(
                            entity_id=entity_id_str,
                            chain_id=chain_id,
                            sequence=sequence_clean,
                            length=len(sequence_clean),
                            description=description,
                            label_chain_ids=label_chain_ids,
                        )
                    )
                group = grouped_sequences.setdefault(
                    sequence_clean,
                    {
                        "chain_ids": [],
                        "description": description,
                    },
                )
                group["chain_ids"].extend(chain_ids)
                continue

            excluded_components.append(
                ExcludedComponent(
                    entity_id=entity_id_str,
                    category=_classify_polymer(polymer_type_str),
                    description=description or polymer_type_str,
                    chain_ids=tuple(chain_ids or label_chain_ids),
                )
            )

    for entity_id, metadata in entity_meta.items():
        entity_type = metadata["entity_type"]
        if entity_type == "polymer":
            if entity_id in polymer_entity_ids:
                continue
            excluded_components.append(
                ExcludedComponent(
                    entity_id=entity_id,
                    category="polymer_unspecified",
                    description=metadata["description"],
                    chain_ids=label_chain_ids_by_entity.get(entity_id, ()),
                )
            )
            continue
        excluded_components.append(
            ExcludedComponent(
                entity_id=entity_id,
                category=entity_type.replace(" ", "_"),
                description=metadata["description"],
                chain_ids=label_chain_ids_by_entity.get(entity_id, ()),
            )
        )

    if not protein_chains:
        return EntryComposition.failed(
            normalized_id,
            f"No protein polymer chains were found in entry {normalized_id}",
        )

    protein_chains = sorted(protein_chains, key=lambda item: item.chain_id)
    chain_lengths = {chain.chain_id: chain.length for chain in protein_chains}
    molecules = []
    for sequence, group in sorted(
        grouped_sequences.items(),
        key=lambda item: sorted(set(item[1]["chain_ids"]))[0],
    ):
        molecules.append(
            {
                "molecule_type": "protein",
                "chain_ids": sorted(set(str(chain_id) for chain_id in group["chain_ids"])),
                "sequence": sequence,
            }
        )

    return EntryComposition(
        pdb_id=normalized_id,
        source_path=resolved_path,
        molecules=molecules,
        protein_chains=protein_chains,
        chain_lengths=chain_lengths,
        total_protein_length=sum(chain_lengths.values()),
        excluded_components=excluded_components,
    )


def collect_entry_compositions(
    pdb_ids: str | Iterable[str],
    *,
    cache_dir: str | Path | None = None,
    session: requests.Session | None = None,
    max_entries: int | None = None,
) -> list[EntryComposition]:
    compositions: list[EntryComposition] = []
    for pdb_id in parse_pdb_ids(pdb_ids, max_entries=max_entries):
        try:
            cif_path = download_mmcif(
                pdb_id,
                cache_dir=cache_dir,
                session=session,
            )
            composition = extract_entry_composition(cif_path, pdb_id=pdb_id)
        except Exception as exc:
            composition = EntryComposition.failed(pdb_id, str(exc))
        compositions.append(composition)
    return compositions


def compositions_to_dataframe(compositions: list[EntryComposition]) -> pd.DataFrame:
    rows = [composition.to_preview_row() for composition in compositions]
    columns = [
        "pdb_id",
        "status",
        "total_protein_length",
        "chain_count",
        "chain_ids",
        "chain_lengths",
        "molecule_count",
        "excluded_component_count",
        "excluded_components",
        "reference_path",
        "failure_reason",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def preview_entries(
    pdb_ids: str | Iterable[str],
    *,
    cache_dir: str | Path | None = None,
    session: requests.Session | None = None,
    max_entries: int | None = None,
) -> pd.DataFrame:
    compositions = collect_entry_compositions(
        pdb_ids,
        cache_dir=cache_dir,
        session=session,
        max_entries=max_entries,
    )
    return compositions_to_dataframe(compositions)
