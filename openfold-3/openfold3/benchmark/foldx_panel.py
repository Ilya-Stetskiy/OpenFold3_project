from __future__ import annotations

import csv
import io
import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import nullcontext, redirect_stderr, redirect_stdout
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
    foldx_binding_ddg_kcal_mol: float | None
    foldx_stability_ddg_kcal_mol: float | None
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


RANKING_METRICS = {
    "binding_ddg": "foldx_binding_ddg_kcal_mol",
    "stability_ddg": "foldx_stability_ddg_kcal_mol",
}


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


def _resolve_ranking_metric_name(ranking_metric: str) -> str:
    resolved = RANKING_METRICS.get(ranking_metric)
    if resolved is None:
        supported = ", ".join(sorted(RANKING_METRICS))
        raise ValueError(
            f"Unsupported ranking_metric {ranking_metric!r}. Expected one of: {supported}"
        )
    return resolved


def _extract_foldx_result(payload: dict[str, Any]) -> dict[str, Any] | None:
    report = payload.get("harness_report") or {}
    results = report.get("results") or []
    for result in results:
        if result.get("method") == "foldx":
            return result
    return None


def _extract_foldx_metrics(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    result = _extract_foldx_result(payload)
    if result is None:
        return None, None
    details = result.get("details") or {}
    binding_ddg = None
    mutant_interaction = details.get("mutant_interaction_energy")
    wt_interaction = details.get("wt_interaction_energy")
    if mutant_interaction is not None and wt_interaction is not None:
        binding_ddg = float(mutant_interaction) - float(wt_interaction)
    elif result.get("score") is not None:
        binding_ddg = float(result["score"])
    stability_ddg = details.get("buildmodel_total_energy_change")
    return (
        binding_ddg,
        None if stability_ddg is None else float(stability_ddg),
    )


def _row_score(
    *,
    binding_ddg: float | None,
    stability_ddg: float | None,
    ranking_metric_name: str,
) -> float | None:
    metric_values = {
        "foldx_binding_ddg_kcal_mol": binding_ddg,
        "foldx_stability_ddg_kcal_mol": stability_ddg,
    }
    return metric_values[ranking_metric_name]


def _row_from_payload(
    payload: dict[str, Any],
    *,
    ranking_metric_name: str,
) -> FoldxPanelMutationRow:
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
    binding_ddg, stability_ddg = _extract_foldx_metrics(payload)
    return FoldxPanelMutationRow(
        case_id=str(payload["case_id"]),
        pdb_id=row_pdb_id,
        chain_id=str(mutation["chain_id"]),
        position_1based=int(mutation["position_1based"]),
        from_residue=str(mutation["from_residue"]),
        to_residue=str(mutation["to_residue"]),
        mutation_id=str(mutation_id),
        local_edit_status=str(payload["local_edit_status"]),
        foldx_binding_ddg_kcal_mol=binding_ddg,
        foldx_stability_ddg_kcal_mol=stability_ddg,
        foldx_score_kcal_mol=_row_score(
            binding_ddg=binding_ddg,
            stability_ddg=stability_ddg,
            ranking_metric_name=ranking_metric_name,
        ),
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


def _load_cached_row(
    case_root: Path,
    *,
    ranking_metric_name: str,
) -> FoldxPanelMutationRow | None:
    result_path = case_root / "local_edit_result.json"
    if not result_path.exists():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return _row_from_payload(payload, ranking_metric_name=ranking_metric_name)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is None:
        return 1
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    return num_workers


def _progress_bar(*, total: int, enabled: bool, description: str):
    if not enabled:
        return nullcontext(None)
    try:
        from tqdm import tqdm
    except Exception:
        return nullcontext(None)
    return tqdm(total=total, desc=description, unit="mutation", dynamic_ncols=True)


def _ranking_rows(
    rows: list[FoldxPanelMutationRow],
    *,
    ranking_metric: str,
) -> list[dict[str, Any]]:
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
                "ranking_metric": ranking_metric,
                "foldx_binding_ddg_kcal_mol": row.foldx_binding_ddg_kcal_mol,
                "foldx_stability_ddg_kcal_mol": row.foldx_stability_ddg_kcal_mol,
                "foldx_score_kcal_mol": row.foldx_score_kcal_mol,
                "mutant_structure_path": row.mutant_structure_path,
                "report_path": row.report_path,
            }
        )
    return ranking


def _run_mutation_payload(
    *,
    mutation: MutationInput,
    cases_root: Path,
    source_path: Path,
    case_id: str,
    cache_dir: str | Path | None,
    session: requests.Session | None,
    harness: DdgBenchmarkHarness | None,
) -> dict[str, Any]:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        result = run_local_mutation_case(
            mutation=mutation,
            work_dir=cases_root,
            structure_path=source_path,
            case_id=case_id,
            cache_dir=cache_dir,
            session=session,
            harness=harness,
        )
    return json.loads(
        (result.report_path.parent / "local_edit_result.json").read_text(encoding="utf-8")
    )


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
    num_workers: int | None = 1,
    ranking_metric: str = "stability_ddg",
) -> FoldxPanelRunResult:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases_root = output_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    resolved_num_workers = _resolve_num_workers(num_workers)
    ranking_metric_name = _resolve_ranking_metric_name(ranking_metric)

    source_path, resolved_pdb_id, mutations = build_foldx_panel_mutations(
        structure_path=structure_path,
        pdb_id=pdb_id,
        chain_id=chain_id,
        positions=positions,
        cache_dir=cache_dir,
        session=session,
    )

    rows_by_case_id: dict[str, FoldxPanelMutationRow] = {}
    uncached_jobs: list[tuple[str, MutationInput]] = []
    for mutation in mutations:
        case_id = _slug_case_id(
            f"{(resolved_pdb_id or Path(source_path).stem).lower()}_{mutation.mutation_id.lower()}"
        )
        case_root = cases_root / case_id
        cached = _load_cached_row(case_root, ranking_metric_name=ranking_metric_name)
        if cached is not None:
            rows_by_case_id[case_id] = cached
            continue
        uncached_jobs.append((case_id, mutation))

    rows: list[FoldxPanelMutationRow] = []
    with _progress_bar(
        total=len(mutations),
        enabled=show_progress,
        description=f"FoldX {chain_id} panel",
    ) as progress:
        if progress is not None and rows_by_case_id:
            progress.update(len(rows_by_case_id))
        if resolved_num_workers == 1 or len(uncached_jobs) <= 1:
            for case_id, mutation in uncached_jobs:
                payload = _run_mutation_payload(
                    mutation=mutation,
                    cases_root=cases_root,
                    source_path=source_path,
                    case_id=case_id,
                    cache_dir=cache_dir,
                    session=session,
                    harness=harness,
                )
                rows_by_case_id[case_id] = _row_from_payload(
                    payload,
                    ranking_metric_name=ranking_metric_name,
                )
                if progress is not None:
                    progress.update(1)
        else:
            future_to_case_id: dict[Future[dict[str, Any]], str] = {}
            with ThreadPoolExecutor(
                max_workers=min(resolved_num_workers, len(uncached_jobs)),
                thread_name_prefix="foldx-panel",
            ) as executor:
                for case_id, mutation in uncached_jobs:
                    future = executor.submit(
                        _run_mutation_payload,
                        mutation=mutation,
                        cases_root=cases_root,
                        source_path=source_path,
                        case_id=case_id,
                        cache_dir=cache_dir,
                        session=session,
                        harness=harness,
                    )
                    future_to_case_id[future] = case_id
                for future in as_completed(future_to_case_id):
                    case_id = future_to_case_id[future]
                    rows_by_case_id[case_id] = _row_from_payload(
                        future.result(),
                        ranking_metric_name=ranking_metric_name,
                    )
                    if progress is not None:
                        progress.update(1)

    for mutation in mutations:
        case_id = _slug_case_id(
            f"{(resolved_pdb_id or Path(source_path).stem).lower()}_{mutation.mutation_id.lower()}"
        )
        rows.append(rows_by_case_id[case_id])

    row_dicts = [asdict(row) for row in rows]
    ranking_rows = _ranking_rows(rows, ranking_metric=ranking_metric)
    summary_payload = {
        "pdb_id": resolved_pdb_id,
        "source_path": str(source_path),
        "chain_id": chain_id,
        "positions": list(positions),
        "ranking_metric": ranking_metric,
        "num_workers": resolved_num_workers,
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
