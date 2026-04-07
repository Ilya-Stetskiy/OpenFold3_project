from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from .cif_utils import parse_structure_records
from .harness import DdgBenchmarkHarness
from .local_edit import run_local_mutation_case
from .models import MutationInput
from .structure_source import CANONICAL_AA_3_TO_1, resolve_structure_source

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True, slots=True)
class FoldxPanelMutationRow:
    case_id: str
    pdb_id: str | None
    chain_id: str
    position_1based: int
    from_residue: str
    to_residue: str
    mutation_id: str
    local_edit_status: str
    foldx_score_kcal_mol: float | None
    runtime_seconds: float | None
    failure_reason: str | None
    source_path: str
    mutant_structure_path: str | None
    report_path: str


@dataclass(frozen=True, slots=True)
class FoldxPanelRunResult:
    output_root: Path
    rows: tuple[FoldxPanelMutationRow, ...]
    summary_json_path: Path
    rows_csv_path: Path
    ranking_csv_path: Path


def _slug_case_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _extract_chain_sequences(structure_path: Path) -> dict[str, str]:
    from .structure_source import extract_protein_sequence
    from .cif_utils import summarize_structure

    summary = summarize_structure(structure_path)
    sequences: dict[str, str] = {}
    for chain_id in summary.chain_ids:
        try:
            sequences[chain_id] = extract_protein_sequence(structure_path, chain_id)
        except ValueError:
            continue
    return sequences


def _resolved_chain_sites(structure_path: Path, chain_id: str) -> list[tuple[str, str]]:
    sites: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for atom in parse_structure_records(structure_path):
        key = (atom.chain_id, atom.residue_id)
        if key in seen or atom.chain_id != chain_id:
            continue
        seen.add(key)
        residue_name_1 = CANONICAL_AA_3_TO_1.get(atom.residue_name.upper())
        if residue_name_1 is None:
            continue
        sites.append((str(atom.residue_id), residue_name_1))
    return sites


def build_foldx_panel_mutations(
    *,
    structure_path: str | Path | None = None,
    pdb_id: str | None = None,
    chain_id: str,
    positions: tuple[int, ...],
    cache_dir: str | Path | None = None,
    session: requests.Session | None = None,
) -> tuple[Path, str | None, list[MutationInput]]:
    resolved = resolve_structure_source(
        structure_path=structure_path,
        pdb_id=pdb_id,
        cache_dir=cache_dir,
        session=session,
    )
    resolved_sites = _resolved_chain_sites(resolved.source_path, chain_id)
    if not resolved_sites:
        raise ValueError(f"Could not find protein chain {chain_id} in {resolved.source_path}")
    mutations: list[MutationInput] = []
    for position in positions:
        if position < 1 or position > len(resolved_sites):
            raise ValueError(
                f"Position {position} is outside chain {chain_id} length {len(resolved_sites)}"
            )
        residue_id, wt_residue = resolved_sites[position - 1]
        if not str(residue_id).isdigit():
            raise ValueError(
                f"Residue id {chain_id}:{residue_id} is not an integer residue id and is unsupported in FoldX v1"
            )
        structure_position = int(residue_id)
        for to_residue in CANONICAL_AA:
            if to_residue == wt_residue:
                continue
            mutations.append(
                MutationInput(
                    chain_id=chain_id,
                    from_residue=wt_residue,
                    position_1based=structure_position,
                    to_residue=to_residue,
                )
            )
    return resolved.source_path, resolved.pdb_id, mutations


def _extract_foldx_score(payload: dict[str, Any]) -> float | None:
    report = payload.get("harness_report") or {}
    results = report.get("results") or []
    for result in results:
        if result.get("method") == "foldx":
            score = result.get("score")
            return None if score is None else float(score)
    return None


def _row_from_payload(payload: dict[str, Any]) -> FoldxPanelMutationRow:
    mutation = payload["mutation"]
    mutation_id = mutation.get("mutation_id")
    if mutation_id is None:
        mutation_id = (
            f"{mutation['chain_id']}_{str(mutation['from_residue']).upper()}"
            f"{int(mutation['position_1based'])}{str(mutation['to_residue']).upper()}"
        )
    row_pdb_id = None
    if payload.get("source_kind") == "pdb_id":
        row_pdb_id = Path(str(payload["source_path"])).stem.upper()
    return FoldxPanelMutationRow(
        case_id=str(payload["case_id"]),
        pdb_id=row_pdb_id,
        chain_id=str(mutation["chain_id"]),
        position_1based=int(mutation["position_1based"]),
        from_residue=str(mutation["from_residue"]),
        to_residue=str(mutation["to_residue"]),
        mutation_id=str(mutation_id),
        local_edit_status=str(payload["local_edit_status"]),
        foldx_score_kcal_mol=_extract_foldx_score(payload),
        runtime_seconds=(
            None
            if payload.get("runtime_seconds") is None
            else float(payload["runtime_seconds"])
        ),
        failure_reason=payload.get("failure_reason"),
        source_path=str(payload["source_path"]),
        mutant_structure_path=payload.get("mutant_structure_path"),
        report_path=str(payload["report_path"]),
    )


def _load_cached_row(case_root: Path) -> FoldxPanelMutationRow | None:
    result_path = case_root / "local_edit_result.json"
    if not result_path.exists():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return _row_from_payload(payload)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _progress_iter(items: list[MutationInput], *, enabled: bool, description: str):
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm
    except Exception:
        return items
    return tqdm(items, desc=description, unit="mutation")


def _ranking_rows(rows: list[FoldxPanelMutationRow]) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if row.local_edit_status == "ok" and row.foldx_score_kcal_mol is not None]
    ordered = sorted(
        ok_rows,
        key=lambda row: (float(row.foldx_score_kcal_mol), row.position_1based, row.to_residue),
    )
    ranking: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        ranking.append(
            {
                "global_rank": index,
                "case_id": row.case_id,
                "mutation_id": row.mutation_id,
                "chain_id": row.chain_id,
                "position_1based": row.position_1based,
                "from_residue": row.from_residue,
                "to_residue": row.to_residue,
                "foldx_score_kcal_mol": row.foldx_score_kcal_mol,
                "mutant_structure_path": row.mutant_structure_path,
                "report_path": row.report_path,
            }
        )
    return ranking


def run_foldx_panel(
    *,
    output_root: str | Path,
    chain_id: str,
    positions: tuple[int, ...],
    structure_path: str | Path | None = None,
    pdb_id: str | None = None,
    cache_dir: str | Path | None = None,
    session: requests.Session | None = None,
    harness: DdgBenchmarkHarness | None = None,
    show_progress: bool = True,
) -> FoldxPanelRunResult:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases_root = output_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)

    source_path, resolved_pdb_id, mutations = build_foldx_panel_mutations(
        structure_path=structure_path,
        pdb_id=pdb_id,
        chain_id=chain_id,
        positions=positions,
        cache_dir=cache_dir,
        session=session,
    )

    rows: list[FoldxPanelMutationRow] = []
    for mutation in _progress_iter(
        mutations,
        enabled=show_progress,
        description=f"FoldX {chain_id} panel",
    ):
        case_id = _slug_case_id(
            f"{(resolved_pdb_id or Path(source_path).stem).lower()}_{mutation.mutation_id.lower()}"
        )
        case_root = cases_root / case_id
        cached = _load_cached_row(case_root)
        if cached is not None:
            rows.append(cached)
            continue
        result = run_local_mutation_case(
            mutation=mutation,
            work_dir=cases_root,
            structure_path=source_path,
            case_id=case_id,
            cache_dir=cache_dir,
            session=session,
            harness=harness,
        )
        payload = json.loads((result.report_path.parent / "local_edit_result.json").read_text(encoding="utf-8"))
        rows.append(_row_from_payload(payload))

    row_dicts = [asdict(row) for row in rows]
    ranking_rows = _ranking_rows(rows)
    summary_payload = {
        "pdb_id": resolved_pdb_id,
        "source_path": str(source_path),
        "chain_id": chain_id,
        "positions": list(positions),
        "total_mutations": len(rows),
        "successful_mutations": sum(row.local_edit_status == "ok" for row in rows),
        "rows": row_dicts,
        "ranking": ranking_rows,
    }
    summary_json_path = output_root / "summary.json"
    rows_csv_path = output_root / "rows.csv"
    ranking_csv_path = output_root / "ranking.csv"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_csv(rows_csv_path, row_dicts)
    _write_csv(ranking_csv_path, ranking_rows)
    return FoldxPanelRunResult(
        output_root=output_root,
        rows=tuple(rows),
        summary_json_path=summary_json_path,
        rows_csv_path=rows_csv_path,
        ranking_csv_path=ranking_csv_path,
    )
