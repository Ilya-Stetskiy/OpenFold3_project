#!/usr/bin/env python
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

"""Compute RMSD between an OpenFold prediction and an experimental structure.

This script is intended as a minimal benchmark utility for validating OpenFold
predictions against experimental PDB/mmCIF structures.

Example:
    python scripts/dev/compute_rmsd_vs_pdb.py \
        --pred path/to/prediction.pdb \
        --ref path/to/reference.cif \
        --atom-set ca \
        --output-json rmsd_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from biotite.structure import filter_amino_acids
from biotite.structure.io import pdb, pdbx


PROTEIN_BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True, help="Predicted structure")
    parser.add_argument("--ref", type=Path, required=True, help="Reference structure")
    parser.add_argument(
        "--atom-set",
        choices=("ca", "backbone", "all-heavy"),
        default="ca",
        help="Atoms to use for alignment and RMSD",
    )
    parser.add_argument(
        "--chains",
        type=str,
        default=None,
        help="Comma-separated chain IDs to use in both structures",
    )
    parser.add_argument(
        "--pred-chains",
        type=str,
        default=None,
        help="Comma-separated chain IDs to use only in the prediction",
    )
    parser.add_argument(
        "--ref-chains",
        type=str,
        default=None,
        help="Comma-separated chain IDs to use only in the reference",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path for the summary",
    )
    parser.add_argument(
        "--fail-on-low-match",
        type=int,
        default=10,
        help="Fail if fewer than this many atoms are matched",
    )
    return parser.parse_args()


def parse_chain_list(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    chain_ids = [item.strip() for item in raw_value.split(",") if item.strip()]
    return chain_ids or None


def load_structure(path: Path):
    suffix = path.suffix.lower()

    if suffix == ".pdb":
        pdb_file = pdb.PDBFile.read(path)
        return pdb.get_structure(
            pdb_file,
            model=1,
            altloc="occupancy",
            extra_fields=["b_factor", "occupancy", "charge"],
        )

    if suffix in {".cif", ".bcif"}:
        cif_class = pdbx.BinaryCIFFile if suffix == ".bcif" else pdbx.CIFFile
        cif_file = cif_class.read(path)
        return pdbx.get_structure(
            cif_file,
            model=1,
            altloc="occupancy",
            use_author_fields=True,
            extra_fields=["b_factor", "occupancy", "charge"],
        )

    raise ValueError(f"Unsupported structure format: {path}")


def get_annotation_or_default(atom_array, name: str, default_value: str) -> np.ndarray:
    try:
        values = getattr(atom_array, name)
    except AttributeError:
        values = np.full(atom_array.array_length(), default_value, dtype=object)
    return np.asarray(values)


def filter_atom_array(atom_array, atom_set: str, chain_ids: list[str] | None):
    mask = filter_amino_acids(atom_array)
    mask &= atom_array.element != "H"

    if atom_set == "ca":
        mask &= atom_array.atom_name == "CA"
    elif atom_set == "backbone":
        mask &= np.isin(atom_array.atom_name, list(PROTEIN_BACKBONE_ATOMS))
    elif atom_set == "all-heavy":
        pass
    else:
        raise ValueError(f"Unsupported atom set: {atom_set}")

    if chain_ids is not None:
        mask &= np.isin(atom_array.chain_id.astype(str), chain_ids)

    return atom_array[mask]


def build_atom_index(atom_array) -> dict[tuple[str, int, str, str], int]:
    ins_codes = get_annotation_or_default(atom_array, "ins_code", "")
    chain_ids = atom_array.chain_id.astype(str)
    atom_names = atom_array.atom_name.astype(str)
    res_ids = atom_array.res_id.astype(int)

    return {
        (chain_id, int(res_id), str(ins_code), atom_name): idx
        for idx, (chain_id, res_id, ins_code, atom_name) in enumerate(
            zip(chain_ids, res_ids, ins_codes, atom_names, strict=True)
        )
    }


def get_common_keys(pred_atom_array, ref_atom_array) -> list[tuple[str, int, str, str]]:
    pred_keys = build_atom_index(pred_atom_array)
    ref_keys = build_atom_index(ref_atom_array)
    common_keys = sorted(set(pred_keys) & set(ref_keys))
    return common_keys


def select_matched_coordinates(pred_atom_array, ref_atom_array) -> tuple[np.ndarray, np.ndarray]:
    pred_index = build_atom_index(pred_atom_array)
    ref_index = build_atom_index(ref_atom_array)
    common_keys = sorted(set(pred_index) & set(ref_index))

    pred_coords = pred_atom_array.coord[[pred_index[key] for key in common_keys]]
    ref_coords = ref_atom_array.coord[[ref_index[key] for key in common_keys]]
    return pred_coords, ref_coords


def compute_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    deltas = coords_a - coords_b
    return float(np.sqrt(np.mean(np.sum(deltas * deltas, axis=1))))


def kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mobile_centroid = mobile.mean(axis=0)
    reference_centroid = reference.mean(axis=0)

    mobile_centered = mobile - mobile_centroid
    reference_centered = reference - reference_centroid

    covariance = mobile_centered.T @ reference_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    return mobile_centered @ rotation + reference_centroid


def summarize_match_coverage(
    pred_atom_array,
    ref_atom_array,
    matched_count: int,
) -> dict[str, int]:
    return {
        "pred_filtered_atom_count": int(pred_atom_array.array_length()),
        "ref_filtered_atom_count": int(ref_atom_array.array_length()),
        "matched_atom_count": int(matched_count),
    }


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def chain_summary(atom_array) -> list[str]:
    return sorted({str(chain_id) for chain_id in atom_array.chain_id.tolist()})


def main() -> None:
    args = parse_args()

    shared_chains = parse_chain_list(args.chains)
    pred_chains = parse_chain_list(args.pred_chains) or shared_chains
    ref_chains = parse_chain_list(args.ref_chains) or shared_chains

    pred_structure = load_structure(args.pred)
    ref_structure = load_structure(args.ref)

    pred_filtered = filter_atom_array(
        pred_structure,
        atom_set=args.atom_set,
        chain_ids=pred_chains,
    )
    ref_filtered = filter_atom_array(
        ref_structure,
        atom_set=args.atom_set,
        chain_ids=ref_chains,
    )

    common_keys = get_common_keys(pred_filtered, ref_filtered)
    if len(common_keys) < args.fail_on_low_match:
        raise RuntimeError(
            "Too few matched atoms for a meaningful RMSD calculation: "
            f"{len(common_keys)} matched, threshold is {args.fail_on_low_match}"
        )

    pred_coords, ref_coords = select_matched_coordinates(pred_filtered, ref_filtered)
    pred_aligned = kabsch_align(pred_coords, ref_coords)

    summary = {
        "pred_path": str(args.pred.resolve()),
        "ref_path": str(args.ref.resolve()),
        "atom_set": args.atom_set,
        "pred_chains": pred_chains,
        "ref_chains": ref_chains,
        "pred_available_chains": chain_summary(pred_structure),
        "ref_available_chains": chain_summary(ref_structure),
        "coverage": summarize_match_coverage(
            pred_atom_array=pred_filtered,
            ref_atom_array=ref_filtered,
            matched_count=len(common_keys),
        ),
        "rmsd": {
            "before_superposition": compute_rmsd(pred_coords, ref_coords),
            "after_superposition": compute_rmsd(pred_aligned, ref_coords),
        },
    }

    if args.output_json is not None:
        dump_json(args.output_json, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
