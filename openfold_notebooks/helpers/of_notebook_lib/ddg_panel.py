from __future__ import annotations

import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

NOTEBOOK_ROOT = Path(__file__).resolve().parents[2]
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

from .display import validate_molecules
from .query_builders import CANONICAL_AA
from openfold3_length_benchmark.composition import (
    collect_entry_compositions,
    compositions_to_dataframe,
    normalize_pdb_id,
)


DEFAULT_SAFE_PPI_TARGET = {
    "pdb_id": "1BRS",
    "target_id": "barnase_barstar",
    "mutable_chain_id": "D",
    "description": "Barnase-barstar safe protein-protein complex.",
}


@dataclass(frozen=True, slots=True)
class PanelVisualRow:
    job_id: str
    panel_id: str
    target_id: str
    chain_id: str
    position_1based: int
    from_residue: str
    to_residue: str
    mutant_query_structure_path: Path | None
    wt_structure_path: Path | None
    foldx_mutant_model_path: Path | None


@dataclass(frozen=True, slots=True)
class FoldxPanelVisualRow:
    case_id: str
    chain_id: str
    position_1based: int
    from_residue: str
    to_residue: str
    wt_structure_path: Path | None
    foldx_mutant_model_path: Path | None
    foldx_binding_ddg_kcal_mol: float | None
    foldx_stability_ddg_kcal_mol: float | None
    foldx_score_kcal_mol: float | None


def parse_positions_spec(positions_text: str, *, sequence_length: int) -> tuple[int, ...]:
    values: list[int] = []
    for token in positions_text.replace("\n", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
            if start > end:
                raise ValueError(f"Invalid range: {token}")
            values.extend(range(start, end + 1))
            continue
        values.append(int(token))
    unique = sorted(set(values))
    invalid = [value for value in unique if value < 1 or value > sequence_length]
    if invalid:
        raise ValueError(
            f"Positions are outside sequence length {sequence_length}: {invalid}"
        )
    return tuple(unique)


def find_chain_sequence(molecules: list[dict], chain_id: str) -> str:
    target_chain = str(chain_id)
    for molecule in molecules:
        chain_ids = [str(value) for value in molecule.get("chain_ids", [])]
        if target_chain in chain_ids and molecule.get("sequence"):
            return str(molecule["sequence"]).upper()
    raise ValueError(f"Could not find protein chain {target_chain}")


def resolve_positions(
    *,
    positions_mode: str,
    positions_text: str,
    sequence_length: int,
) -> tuple[int, ...]:
    if positions_mode == "all_chain_positions":
        return tuple(range(1, sequence_length + 1))
    if positions_mode == "explicit_list":
        return parse_positions_spec(positions_text, sequence_length=sequence_length)
    raise ValueError(f"Unsupported positions_mode: {positions_mode}")


def resolve_experiment_molecules(
    *,
    pdb_id: str,
    pdb_cache_dir: str | Path | None = None,
) -> tuple[list[dict], pd.DataFrame, dict[str, object]]:
    normalized_id = normalize_pdb_id(pdb_id)
    compositions = collect_entry_compositions(
        normalized_id,
        cache_dir=pdb_cache_dir,
        max_entries=1,
    )
    if not compositions:
        raise ValueError(f"Could not load composition for {normalized_id}")
    composition = compositions[0]
    if composition.status != "ok":
        raise ValueError(
            f"Failed to load {normalized_id}: {composition.issue or composition.status}"
        )
    molecules = composition.molecules
    issues = validate_molecules(molecules)
    if issues:
        raise ValueError("; ".join(issues))
    metadata = {
        "pdb_id": composition.pdb_id,
        "reference_path": (
            None if composition.source_path is None else str(composition.source_path)
        ),
        "chain_ids": ",".join(composition.chain_ids),
        "total_protein_length": composition.total_protein_length,
    }
    return molecules, compositions_to_dataframe(compositions), metadata


def resolve_foldx_chain_context(
    *,
    pdb_id: str,
    mutable_chain_id: str,
    pdb_cache_dir: str | Path | None = None,
) -> dict[str, object]:
    from openfold3.benchmark.cif_utils import parse_structure_records
    from openfold3.benchmark.structure_source import (
        CANONICAL_AA_3_TO_1,
        extract_protein_sequence,
        resolve_structure_source,
    )

    resolved = resolve_structure_source(pdb_id=pdb_id, cache_dir=pdb_cache_dir)
    sequence = extract_protein_sequence(resolved.source_path, mutable_chain_id)
    residue_ids: list[str] = []
    sequence_positions: list[int] = []
    residue_min_partner_distances: list[float | None] = []
    residue_is_interface: list[bool] = []
    seen: set[tuple[str, str]] = set()
    all_atoms = parse_structure_records(resolved.source_path)
    partner_atoms = [
        atom for atom in all_atoms if atom.chain_id != mutable_chain_id
    ]
    for atom in all_atoms:
        key = (atom.chain_id, atom.residue_id)
        if key in seen or atom.chain_id != mutable_chain_id:
            continue
        seen.add(key)
        if atom.residue_name.upper() not in CANONICAL_AA_3_TO_1:
            continue
        sequence_positions.append(len(sequence_positions) + 1)
        residue_ids.append(str(atom.residue_id))
        residue_atoms = [
            residue_atom
            for residue_atom in all_atoms
            if residue_atom.chain_id == mutable_chain_id
            and residue_atom.residue_id == atom.residue_id
        ]
        min_distance = None if not partner_atoms else min(
            math.dist(
                (left.x, left.y, left.z),
                (right.x, right.y, right.z),
            )
            for left in residue_atoms
            for right in partner_atoms
        )
        residue_min_partner_distances.append(min_distance)
        residue_is_interface.append(
            min_distance is not None and min_distance <= 8.0
        )
    return {
        "source_path": resolved.source_path,
        "pdb_id": resolved.pdb_id,
        "chain_id": mutable_chain_id,
        "sequence": sequence,
        "sequence_length": len(sequence),
        "sequence_positions": tuple(sequence_positions),
        "residue_ids": tuple(residue_ids),
        "residue_min_partner_distances": tuple(residue_min_partner_distances),
        "residue_is_interface": tuple(residue_is_interface),
        "first_residue_id": None if not residue_ids else residue_ids[0],
        "last_residue_id": None if not residue_ids else residue_ids[-1],
    }


def build_panel_preview(
    target_id: str,
    molecules: list[dict],
    mutable_chain_id: str,
    positions: tuple[int, ...],
) -> pd.DataFrame:
    sequence = find_chain_sequence(molecules, mutable_chain_id)
    rows = []
    for position in positions:
        wt_residue = sequence[position - 1]
        mutant_count = len(CANONICAL_AA) - 1
        rows.append(
            {
                "target_id": target_id,
                "chain_id": mutable_chain_id,
                "position_1based": position,
                "wt_residue": wt_residue,
                "mutation_panel": f"{mutable_chain_id}_{wt_residue}{position}",
                "mutant_count": mutant_count,
                "mutants_preview": ",".join(
                    residue for residue in CANONICAL_AA if residue != wt_residue
                ),
            }
        )
    return pd.DataFrame(rows)


def render_info_card(title: str, items: list[tuple[str, object]], *, accent: str) -> str:
    lines = [
        f"<div style='border-left: 6px solid {accent}; padding: 12px 16px; margin: 10px 0; background: #f8f9fa;'>",
        f"<div style='font-weight: 700; margin-bottom: 8px;'>{title}</div>",
        "<table style='border-collapse: collapse;'>",
    ]
    for key, value in items:
        lines.append(
            "<tr>"
            f"<td style='padding: 2px 12px 2px 0; font-weight: 600; vertical-align: top;'>{key}</td>"
            f"<td style='padding: 2px 0; vertical-align: top;'>{value}</td>"
            "</tr>"
        )
    lines.extend(["</table>", "</div>"])
    return "".join(lines)


def build_run_name(*, pdb_id: str, mutable_chain_id: str) -> str:
    return f"{normalize_pdb_id(pdb_id).lower()}_{mutable_chain_id.lower()}_ddg_panel"


def build_foldx_run_name(*, pdb_id: str, mutable_chain_id: str) -> str:
    return f"{normalize_pdb_id(pdb_id).lower()}_{mutable_chain_id.lower()}_foldx_panel"


def summarize_panel_preview(
    *,
    target_id: str,
    pdb_id: str,
    mutable_chain_id: str,
    positions: tuple[int, ...],
    molecules: list[dict],
) -> dict[str, object]:
    sequence = find_chain_sequence(molecules, mutable_chain_id)
    return {
        "target_id": target_id,
        "pdb_id": normalize_pdb_id(pdb_id),
        "mutable_chain_id": mutable_chain_id,
        "sequence_length": len(sequence),
        "positions_count": len(positions),
        "planned_mutants": len(positions) * (len(CANONICAL_AA) - 1),
    }


def _first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def _viewer_format(path: Path) -> str:
    return "pdb" if path.suffix.lower() == ".pdb" else "mmcif"


def _load_py3dmol() -> Any | None:
    try:
        import py3Dmol
    except Exception:
        return None
    return py3Dmol


def _read_method_details_map(
    conn: sqlite3.Connection,
    *,
    target_id: str,
) -> dict[str, dict[str, dict[str, object]]]:
    rows = list(
        conn.execute(
            """
            SELECT j.job_id, mr.method, mr.details_json
            FROM jobs j
            JOIN method_results mr ON mr.job_id = j.job_id
            WHERE j.target_id = ?
            """,
            (target_id,),
        )
    )
    details_by_job: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        details_json = row["details_json"]
        try:
            details = json.loads(details_json) if details_json else {}
        except json.JSONDecodeError:
            details = {}
        details_by_job.setdefault(str(row["job_id"]), {})[str(row["method"])] = details
    return details_by_job


def load_panel_visual_rows(state_db_path: str | Path) -> list[PanelVisualRow]:
    conn = sqlite3.connect(Path(state_db_path))
    conn.row_factory = sqlite3.Row
    try:
        wt_row = conn.execute("SELECT * FROM wt_baseline LIMIT 1").fetchone()
        if wt_row is None:
            raise ValueError(f"No wt_baseline rows found in {state_db_path}")
        target_id = str(wt_row["target_id"])
        wt_structure_path = (
            None
            if wt_row["structure_path"] is None
            else Path(str(wt_row["structure_path"]))
        )
        details_by_job = _read_method_details_map(conn, target_id=target_id)
        job_rows = list(
            conn.execute(
                """
                SELECT *
                FROM jobs
                WHERE target_id = ?
                ORDER BY position_1based, to_residue, job_id
                """,
                (target_id,),
            )
        )
    finally:
        conn.close()

    visual_rows: list[PanelVisualRow] = []
    for row in job_rows:
        details = details_by_job.get(str(row["job_id"]), {}).get("foldx", {})
        foldx_path = _first_existing(
            [
                Path(str(details.get("mutant_model_path")))
                if details.get("mutant_model_path")
                else None,
                Path(str(details.get("prepared_input_pdb_path")))
                if details.get("prepared_input_pdb_path")
                else None,
            ]
        )
        visual_rows.append(
            PanelVisualRow(
                job_id=str(row["job_id"]),
                panel_id=str(row["panel_id"]),
                target_id=str(row["target_id"]),
                chain_id=str(row["chain_id"]),
                position_1based=int(row["position_1based"]),
                from_residue=str(row["from_residue"]),
                to_residue=str(row["to_residue"]),
                mutant_query_structure_path=(
                    None
                    if row["structure_path"] is None
                    else Path(str(row["structure_path"]))
                ),
                wt_structure_path=wt_structure_path,
                foldx_mutant_model_path=foldx_path,
            )
        )
    return visual_rows


def load_foldx_panel_visual_rows(summary_json_path: str | Path) -> list[FoldxPanelVisualRow]:
    payload = json.loads(Path(summary_json_path).read_text(encoding="utf-8"))
    wt_structure_path = (
        None
        if payload.get("source_path") is None
        else Path(str(payload["source_path"]))
    )
    rows: list[FoldxPanelVisualRow] = []
    for row in payload.get("rows", []):
        rows.append(
            FoldxPanelVisualRow(
                case_id=str(row["case_id"]),
                chain_id=str(row["chain_id"]),
                position_1based=int(row["position_1based"]),
                from_residue=str(row["from_residue"]),
                to_residue=str(row["to_residue"]),
                wt_structure_path=wt_structure_path,
                foldx_mutant_model_path=(
                    None
                    if row.get("mutant_structure_path") is None
                    else Path(str(row["mutant_structure_path"]))
                ),
                foldx_binding_ddg_kcal_mol=(
                    None
                    if row.get("foldx_binding_ddg_kcal_mol") is None
                    else float(row["foldx_binding_ddg_kcal_mol"])
                ),
                foldx_stability_ddg_kcal_mol=(
                    None
                    if row.get("foldx_stability_ddg_kcal_mol") is None
                    else float(row["foldx_stability_ddg_kcal_mol"])
                ),
                foldx_score_kcal_mol=(
                    None
                    if row.get("foldx_score_kcal_mol") is None
                    else float(row["foldx_score_kcal_mol"])
                ),
            )
        )
    return rows


def _render_structure_view(
    structure_path: Path | None,
    *,
    title: str,
    width: int,
    height: int,
    chain_id: str,
    position_1based: int,
    missing_message: str,
) -> str:
    if structure_path is None or not structure_path.exists():
        return f"<div><h4>{title}</h4><p>{missing_message}</p></div>"
    py3Dmol = _load_py3dmol()
    if py3Dmol is None:
        return (
            f"<div><h4>{title}</h4>"
            f"<p>py3Dmol is unavailable.</p>"
            f"<p>structure: {structure_path}</p></div>"
        )
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(
        structure_path.read_text(encoding="utf-8"),
        _viewer_format(structure_path),
    )
    viewer.setStyle({"model": 0}, {"cartoon": {"color": "#cbd5e1"}})
    viewer.setStyle({"model": 0, "chain": chain_id}, {"cartoon": {"color": "#0f766e"}})
    viewer.addStyle(
        {"model": 0, "chain": chain_id, "resi": str(position_1based)},
        {"stick": {"colorscheme": "greenCarbon", "radius": 0.25}},
    )
    viewer.zoomTo({"model": 0, "chain": chain_id, "resi": str(position_1based)})
    return viewer._make_html()  # noqa: SLF001


def render_panel_structure_comparison_html(
    row: PanelVisualRow,
    *,
    width: int = 420,
    height: int = 320,
) -> str:
    wt_html = _render_structure_view(
        row.wt_structure_path,
        title="WT complex",
        width=width,
        height=height,
        chain_id=row.chain_id,
        position_1based=row.position_1based,
        missing_message="WT structure is unavailable.",
    )
    foldx_html = _render_structure_view(
        row.foldx_mutant_model_path,
        title="FoldX local mutant",
        width=width,
        height=height,
        chain_id=row.chain_id,
        position_1based=row.position_1based,
        missing_message="FoldX local mutant structure is unavailable.",
    )
    mutant_html = _render_structure_view(
        row.mutant_query_structure_path,
        title="OpenFold mutant complex",
        width=width,
        height=height,
        chain_id=row.chain_id,
        position_1based=row.position_1based,
        missing_message="Mutant query structure is unavailable.",
    )
    label = f"{row.chain_id}:{row.from_residue}{row.position_1based}{row.to_residue}"
    return (
        f"<div><p><strong>{label}</strong> job_id={row.job_id}</p>"
        f"<div style='display:flex; gap:16px; align-items:flex-start;'>"
        f"<div style='flex:1'>{wt_html}</div>"
        f"<div style='flex:1'>{foldx_html}</div>"
        f"<div style='flex:1'>{mutant_html}</div>"
        f"</div></div>"
    )


def render_foldx_structure_comparison_html(
    row: FoldxPanelVisualRow,
    *,
    width: int = 460,
    height: int = 340,
) -> str:
    wt_html = _render_structure_view(
        row.wt_structure_path,
        title="WT complex",
        width=width,
        height=height,
        chain_id=row.chain_id,
        position_1based=row.position_1based,
        missing_message="WT structure is unavailable.",
    )
    foldx_html = _render_structure_view(
        row.foldx_mutant_model_path,
        title="FoldX mutant",
        width=width,
        height=height,
        chain_id=row.chain_id,
        position_1based=row.position_1based,
        missing_message="FoldX mutant structure is unavailable.",
    )
    label = f"{row.chain_id}:{row.from_residue}{row.position_1based}{row.to_residue}"
    score_text = "NA" if row.foldx_score_kcal_mol is None else f"{row.foldx_score_kcal_mol:.4f}"
    binding_text = (
        "NA"
        if row.foldx_binding_ddg_kcal_mol is None
        else f"{row.foldx_binding_ddg_kcal_mol:.4f}"
    )
    stability_text = (
        "NA"
        if row.foldx_stability_ddg_kcal_mol is None
        else f"{row.foldx_stability_ddg_kcal_mol:.4f}"
    )
    return (
        f"<div><p><strong>{label}</strong> selected={score_text} kcal/mol; "
        f"binding={binding_text}; stability={stability_text}</p>"
        f"<div style='display:flex; gap:16px; align-items:flex-start;'>"
        f"<div style='flex:1'>{wt_html}</div>"
        f"<div style='flex:1'>{foldx_html}</div>"
        f"</div></div>"
    )


def preview_panel_input(
    *,
    pdb_id: str,
    mutable_chain_id: str,
    positions_mode: str,
    positions_text: str,
    pdb_cache_dir: str | Path | None = None,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, dict[str, object], tuple[int, ...]]:
    molecules, pdb_preview_df, metadata = resolve_experiment_molecules(
        pdb_id=pdb_id,
        pdb_cache_dir=pdb_cache_dir,
    )
    sequence = find_chain_sequence(molecules, mutable_chain_id)
    positions = resolve_positions(
        positions_mode=positions_mode,
        positions_text=positions_text,
        sequence_length=len(sequence),
    )
    preview_df = build_panel_preview(
        target_id=build_run_name(pdb_id=pdb_id, mutable_chain_id=mutable_chain_id),
        molecules=molecules,
        mutable_chain_id=mutable_chain_id,
        positions=positions,
    )
    return (
        molecules,
        pdb_preview_df,
        preview_df,
        summarize_panel_preview(
            target_id=build_run_name(pdb_id=pdb_id, mutable_chain_id=mutable_chain_id),
            pdb_id=pdb_id,
            mutable_chain_id=mutable_chain_id,
            positions=positions,
            molecules=molecules,
        ),
        positions,
    )


def preview_foldx_panel_input(
    *,
    pdb_id: str,
    mutable_chain_id: str,
    positions_mode: str,
    positions_text: str,
    pdb_cache_dir: str | Path | None = None,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, dict[str, object], tuple[int, ...]]:
    molecules, pdb_preview_df, _ = resolve_experiment_molecules(
        pdb_id=pdb_id,
        pdb_cache_dir=pdb_cache_dir,
    )
    chain_context = resolve_foldx_chain_context(
        pdb_id=pdb_id,
        mutable_chain_id=mutable_chain_id,
        pdb_cache_dir=pdb_cache_dir,
    )
    positions = resolve_positions(
        positions_mode=positions_mode,
        positions_text=positions_text,
        sequence_length=int(chain_context["sequence_length"]),
    )
    preview_df = build_panel_preview(
        target_id=build_foldx_run_name(
            pdb_id=pdb_id,
            mutable_chain_id=mutable_chain_id,
        ),
        molecules=[{"molecule_type": "protein", "chain_ids": [mutable_chain_id], "sequence": chain_context["sequence"]}],
        mutable_chain_id=mutable_chain_id,
        positions=positions,
    )
    residue_id_by_position = {
        int(position): str(residue_id)
        for position, residue_id in zip(
            chain_context["sequence_positions"],
            chain_context["residue_ids"],
            strict=True,
        )
    }
    interface_by_position = {
        int(position): bool(is_interface)
        for position, is_interface in zip(
            chain_context["sequence_positions"],
            chain_context["residue_is_interface"],
            strict=True,
        )
    }
    min_distance_by_position = {
        int(position): (
            None if min_distance is None else float(min_distance)
        )
        for position, min_distance in zip(
            chain_context["sequence_positions"],
            chain_context["residue_min_partner_distances"],
            strict=True,
        )
    }
    if not preview_df.empty:
        preview_df["residue_id"] = preview_df["position_1based"].map(residue_id_by_position)
        preview_df["is_interface_8a"] = preview_df["position_1based"].map(interface_by_position)
        preview_df["min_partner_atom_distance_a"] = preview_df["position_1based"].map(
            min_distance_by_position
        )
        preview_df = preview_df[
            [
                "target_id",
                "chain_id",
                "position_1based",
                "residue_id",
                "is_interface_8a",
                "min_partner_atom_distance_a",
                "wt_residue",
                "mutation_panel",
                "mutant_count",
                "mutants_preview",
            ]
        ]
    return (
        molecules,
        pdb_preview_df,
        preview_df,
        {
            "target_id": build_foldx_run_name(
                pdb_id=pdb_id,
                mutable_chain_id=mutable_chain_id,
            ),
            "pdb_id": normalize_pdb_id(pdb_id),
            "mutable_chain_id": mutable_chain_id,
            "sequence_length": int(chain_context["sequence_length"]),
            "positions_count": len(positions),
            "planned_mutants": len(positions) * (len(CANONICAL_AA) - 1),
            "resolved_structure_path": str(chain_context["source_path"]),
            "first_residue_id": chain_context["first_residue_id"],
            "last_residue_id": chain_context["last_residue_id"],
            "interface_positions": sum(bool(value) for value in chain_context["residue_is_interface"]),
        },
        positions,
    )
