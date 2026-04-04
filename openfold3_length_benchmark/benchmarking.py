from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from of_notebook_lib.analysis import collect_samples


def _tokenize_cif_row(line: str) -> list[str]:
    tokens: list[str] = []
    current = []
    quote: str | None = None
    for char in line.strip():
        if quote is not None:
            if char == quote:
                tokens.append("".join(current))
                current = []
                quote = None
            else:
                current.append(char)
            continue
        if char in {"'", '"'}:
            if current:
                tokens.append("".join(current))
                current = []
            quote = char
            continue
        if char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


@dataclass(frozen=True)
class AtomSiteRecord:
    chain_id: str
    residue_id: str
    atom_name: str
    coord: tuple[float, float, float]


def _parse_pdb_atoms(path: Path) -> list[AtomSiteRecord]:
    records: list[AtomSiteRecord] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.startswith(("ATOM", "HETATM")):
            continue
        chain_id = raw_line[21].strip() or "?"
        residue_id = raw_line[22:26].strip() or "?"
        insertion_code = raw_line[26].strip()
        if insertion_code:
            residue_id = f"{residue_id}{insertion_code}"
        records.append(
            AtomSiteRecord(
                chain_id=chain_id,
                residue_id=residue_id,
                atom_name=raw_line[12:16].strip(),
                coord=(
                    float(raw_line[30:38].strip()),
                    float(raw_line[38:46].strip()),
                    float(raw_line[46:54].strip()),
                ),
            )
        )
    return records


def _parse_cif_atoms(path: Path) -> list[AtomSiteRecord]:
    headers: list[str] = []
    records: list[AtomSiteRecord] = []
    in_atom_site_loop = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "loop_":
            headers = []
            in_atom_site_loop = False
            continue
        if line.startswith("_atom_site."):
            headers.append(line.removeprefix("_atom_site."))
            in_atom_site_loop = True
            continue
        if in_atom_site_loop and headers and (line == "#" or line.startswith("_")):
            break
        if not in_atom_site_loop or not headers:
            continue

        parts = _tokenize_cif_row(line)
        if len(parts) != len(headers):
            continue
        row = dict(zip(headers, parts, strict=True))
        auth_chain = row.get("auth_asym_id", "?")
        label_chain = row.get("label_asym_id", "?")
        chain_id = auth_chain if auth_chain not in {"?", "."} else label_chain
        auth_seq_id = row.get("auth_seq_id", "?")
        label_seq_id = row.get("label_seq_id", "?")
        residue_id = auth_seq_id if auth_seq_id not in {"?", "."} else label_seq_id
        insertion_code = row.get("pdbx_PDB_ins_code")
        if insertion_code not in {None, "?", "."}:
            residue_id = f"{residue_id}{insertion_code}"
        records.append(
            AtomSiteRecord(
                chain_id=chain_id,
                residue_id=str(residue_id),
                atom_name=row.get("auth_atom_id", row.get("label_atom_id", "?")),
                coord=(
                    float(row["Cartn_x"]),
                    float(row["Cartn_y"]),
                    float(row["Cartn_z"]),
                ),
            )
        )
    return records


def parse_structure_atoms(path: str | Path) -> list[AtomSiteRecord]:
    structure_path = Path(path)
    if structure_path.suffix.lower() == ".pdb":
        return _parse_pdb_atoms(structure_path)
    return _parse_cif_atoms(structure_path)


def _atom_filter(atom_set: str) -> set[str] | None:
    normalized = atom_set.strip().lower()
    if normalized == "ca":
        return {"CA"}
    if normalized == "backbone":
        return {"N", "CA", "C", "O"}
    if normalized == "all":
        return None
    raise ValueError(f"Unsupported atom_set: {atom_set!r}")


def _indexed_coords(
    atoms: list[AtomSiteRecord],
    *,
    atom_names: set[str] | None,
) -> dict[tuple[str, str, str], np.ndarray]:
    indexed: dict[tuple[str, str, str], np.ndarray] = {}
    for atom in atoms:
        if atom_names is not None and atom.atom_name not in atom_names:
            continue
        indexed[(atom.chain_id, atom.residue_id, atom.atom_name)] = np.asarray(atom.coord, dtype=float)
    return indexed


def _kabsch_superpose(
    mobile: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_mobile = mobile - mobile_center
    centered_target = target - target_center

    rmsd_before = float(np.sqrt(np.mean(np.sum((centered_mobile - centered_target) ** 2, axis=1))))

    covariance = centered_mobile.T @ centered_target
    v_matrix, _, w_transpose = np.linalg.svd(covariance)
    determinant = np.linalg.det(v_matrix @ w_transpose)
    correction = np.eye(3)
    if determinant < 0:
        correction[-1, -1] = -1.0
    rotation = v_matrix @ correction @ w_transpose
    aligned_mobile = centered_mobile @ rotation + target_center
    rmsd_after = float(np.sqrt(np.mean(np.sum((aligned_mobile - target) ** 2, axis=1))))
    return aligned_mobile, rmsd_before, rmsd_after


def compute_structure_rmsd(
    pred_path: str | Path,
    ref_path: str | Path,
    *,
    atom_set: str = "ca",
) -> dict[str, Any]:
    atom_names = _atom_filter(atom_set)
    pred_atoms = parse_structure_atoms(pred_path)
    ref_atoms = parse_structure_atoms(ref_path)
    pred_index = _indexed_coords(pred_atoms, atom_names=atom_names)
    ref_index = _indexed_coords(ref_atoms, atom_names=atom_names)
    shared_keys = sorted(set(pred_index) & set(ref_index))
    if not shared_keys:
        raise ValueError(f"No shared atoms found between {pred_path} and {ref_path}")

    pred_coords = np.vstack([pred_index[key] for key in shared_keys])
    ref_coords = np.vstack([ref_index[key] for key in shared_keys])
    _, rmsd_before, rmsd_after = _kabsch_superpose(pred_coords, ref_coords)

    return {
        "rmsd_before_superposition": rmsd_before,
        "rmsd_after_superposition": rmsd_after,
        "coverage": {
            "matched_atom_count": int(len(shared_keys)),
            "pred_filtered_atom_count": int(len(pred_index)),
            "ref_filtered_atom_count": int(len(ref_index)),
        },
    }


def run_rmsd_benchmark(
    *,
    pred_root: str | Path,
    ref_dir: str | Path,
    output_dir: str | Path,
    atom_set: str = "ca",
) -> Path:
    pred_root = Path(pred_root)
    ref_dir = Path(ref_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for sample in collect_samples(pred_root):
        if sample.model_path is None:
            failures.append(
                {
                    "query": sample.query_name,
                    "sample": sample.sample_name,
                    "reason": "model_path_missing",
                }
            )
            continue

        query_name = sample.query_name.upper()
        ref_path = ref_dir / f"{query_name}.cif"
        if not ref_path.exists():
            ref_path = ref_dir / f"{query_name}.pdb"
        if not ref_path.exists():
            failures.append(
                {
                    "query": sample.query_name,
                    "sample": sample.sample_name,
                    "reason": f"reference_missing:{query_name}",
                }
            )
            continue

        try:
            rmsd_result = compute_structure_rmsd(sample.model_path, ref_path, atom_set=atom_set)
        except Exception as exc:
            failures.append(
                {
                    "query": sample.query_name,
                    "sample": sample.sample_name,
                    "reason": str(exc),
                }
            )
            continue

        rows.append(
            {
                "query": sample.query_name,
                "seed": sample.seed_name,
                "sample": sample.sample_name,
                "pred_path": str(sample.model_path),
                "ref_path": str(ref_path),
                "rmsd_before_superposition": rmsd_result["rmsd_before_superposition"],
                "rmsd_after_superposition": rmsd_result["rmsd_after_superposition"],
                "coverage": rmsd_result["coverage"],
                "aggregated_confidence": {
                    "avg_plddt": sample.avg_plddt,
                    "ptm": sample.ptm,
                    "iptm": sample.iptm,
                    "sample_ranking_score": sample.sample_ranking_score,
                    "gpde": sample.gpde,
                    "has_clash": sample.has_clash,
                },
            }
        )

    rows_path = output_dir / "rmsd_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "n_samples": len(rows) + len(failures),
        "n_successful": len(rows),
        "n_failed": len(failures),
        "atom_set": atom_set,
        "failures": failures,
    }
    (output_dir / "rmsd_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "benchmark_rmsd.log").write_text(
        "\n".join(
            [
                f"atom_set={atom_set}",
                f"samples_total={summary['n_samples']}",
                f"samples_successful={summary['n_successful']}",
                f"samples_failed={summary['n_failed']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if not rows:
        raise ValueError(f"No RMSD rows were produced under {pred_root}")
    return output_dir
