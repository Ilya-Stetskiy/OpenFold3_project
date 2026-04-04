from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from of_notebook_lib.analysis import collect_samples


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _sample_index(output_dir: Path) -> dict[str, object]:
    return {sample.sample_name: sample for sample in collect_samples(output_dir)}


def _viewer_format(path: Path) -> str:
    return "pdb" if path.suffix.lower() == ".pdb" else "mmcif"


def _view_row_records(result) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    results_df = result.results_df[result.results_df["status"] == "ok"].copy()
    for row in results_df.to_dict(orient="records"):
        predict_run_dir = Path(str(row["predict_run_dir"]))
        output_dir = predict_run_dir / "output"
        sample_name = row.get("model_selected_sample")
        sample = _sample_index(output_dir).get(sample_name)
        model_path = None if sample is None else getattr(sample, "model_path", None)
        if model_path is None:
            model_path = _first_existing(
                [
                    Path(str(row["predict_summary_dir"])) / "best_by_sample_ranking_score" / f"{sample_name}_model.cif",
                    Path(str(row["predict_summary_dir"])) / "best_by_sample_ranking_score" / f"{sample_name}_model.pdb",
                ]
            )

        rows.append(
            {
                "pdb_id": row["pdb_id"],
                "total_protein_length": row["total_protein_length"],
                "model_selected_rmsd": row["model_selected_rmsd"],
                "avg_plddt": row["avg_plddt"],
                "reference_path": Path(str(row["reference_path"])),
                "model_path": None if model_path is None else Path(model_path),
                "sample_name": sample_name,
            }
        )
    return rows


def display_result_structures(
    result,
    *,
    pdb_ids: str | Iterable[str] | None = None,
    max_items: int = 3,
    width: int = 520,
    height: int = 360,
) -> None:
    try:
        from IPython.display import HTML, Markdown, display
    except Exception as exc:
        raise RuntimeError(
            "Inline notebook display requires IPython in the kernel environment."
        ) from exc

    records = _view_row_records(result)
    if pdb_ids:
        selected_ids = (
            {item.strip().upper() for item in pdb_ids.split()} if isinstance(pdb_ids, str) else {str(item).strip().upper() for item in pdb_ids}
        )
        records = [record for record in records if str(record["pdb_id"]).upper() in selected_ids]
    if max_items is not None:
        records = records[:max_items]

    if not records:
        display(Markdown("No successful structures were available for inline viewing."))
        return

    try:
        import py3Dmol
    except Exception:
        lines = ["Inline 3D viewer is unavailable (`py3Dmol` not installed).", ""]
        for record in records:
            lines.extend(
                [
                    f"- `{record['pdb_id']}` length={record['total_protein_length']} RMSD={record['model_selected_rmsd']}",
                    f"  reference: `{record['reference_path']}`",
                    f"  predicted: `{record['model_path']}`",
                ]
            )
        display(Markdown("\n".join(lines)))
        return

    for record in records:
        reference_path = record["reference_path"]
        model_path = record["model_path"]
        if model_path is None or not Path(model_path).exists():
            display(
                Markdown(
                    f"### {record['pdb_id']}\n"
                    f"Selected model file is missing.\n\n"
                    f"- reference: `{reference_path}`"
                )
            )
            continue

        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(reference_path.read_text(encoding="utf-8"), _viewer_format(reference_path))
        viewer.setStyle({"model": 0}, {"cartoon": {"color": "#94a3b8"}})
        viewer.addModel(Path(model_path).read_text(encoding="utf-8"), _viewer_format(Path(model_path)))
        viewer.setStyle({"model": 1}, {"cartoon": {"color": "#0f766e"}})
        viewer.zoomTo()

        display(
            Markdown(
                f"### {record['pdb_id']}\n"
                f"- length: `{record['total_protein_length']}`\n"
                f"- selected RMSD: `{record['model_selected_rmsd']}`\n"
                f"- selected sample: `{record['sample_name']}`\n"
                f"- avg pLDDT: `{record['avg_plddt']}`\n"
                f"- reference: `{reference_path}`\n"
                f"- predicted: `{model_path}`"
            )
        )
        display(HTML(viewer._make_html()))  # noqa: SLF001
