from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from of_notebook_lib.analysis import collect_samples


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _viewer_format(path: Path) -> str:
    return "pdb" if path.suffix.lower() == ".pdb" else "mmcif"


def _load_display_dependencies() -> tuple[Any, Any, Any]:
    try:
        from IPython.display import HTML, Markdown, display
    except Exception as exc:
        raise RuntimeError(
            "Inline notebook display requires IPython in the kernel environment."
        ) from exc
    return HTML, Markdown, display


def _load_py3dmol() -> Any | None:
    try:
        import py3Dmol
    except Exception:
        return None
    return py3Dmol


def _sample_records_by_name(output_dir: Path) -> dict[str, Any]:
    return {sample.sample_name: sample for sample in collect_samples(output_dir)}


def _resolve_model_path(sample_name: str | None, row: dict[str, object], sample_index: dict[str, Any]) -> Path | None:
    if not sample_name:
        return None
    sample = sample_index.get(sample_name)
    model_path = None if sample is None else getattr(sample, "model_path", None)
    if model_path is not None and Path(model_path).exists():
        return Path(model_path)
    summary_dir = Path(str(row["predict_summary_dir"]))
    return _first_existing(
        [
            summary_dir / "best_by_sample_ranking_score" / f"{sample_name}_model.cif",
            summary_dir / "best_by_sample_ranking_score" / f"{sample_name}_model.pdb",
        ]
    )


def _build_pdb_index(result) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    results_df = result.results_df[result.results_df["status"] == "ok"].copy()
    sample_points_df = getattr(result, "sample_points_df", None)

    for row in results_df.to_dict(orient="records"):
        pdb_id = str(row["pdb_id"])
        predict_run_dir = Path(str(row["predict_run_dir"]))
        output_dir = predict_run_dir / "output"
        samples_by_name = _sample_records_by_name(output_dir)

        sample_options: dict[str, dict[str, object]] = {}
        sample_frame = None
        if sample_points_df is not None and not sample_points_df.empty:
            sample_frame = sample_points_df[sample_points_df["pdb_id"] == pdb_id].copy()

        for sample_name, sample in samples_by_name.items():
            matching_row = None
            if sample_frame is not None and not sample_frame.empty:
                matches = sample_frame[sample_frame["sample"] == sample_name]
                if not matches.empty:
                    matching_row = matches.iloc[0].to_dict()
            sample_options[sample_name] = {
                "label": sample_name,
                "sample_name": sample_name,
                "seed": getattr(sample, "seed_name", None),
                "model_path": None if getattr(sample, "model_path", None) is None else Path(sample.model_path),
                "rmsd": None if matching_row is None else matching_row.get("rmsd_after_superposition"),
                "sample_ranking_score": getattr(sample, "sample_ranking_score", None),
                "avg_plddt": getattr(sample, "avg_plddt", None),
            }

        selected_sample = row.get("model_selected_sample")
        oracle_sample = row.get("oracle_sample")
        index[pdb_id] = {
            "pdb_id": pdb_id,
            "length": row.get("total_protein_length"),
            "reference_path": Path(str(row["reference_path"])),
            "selected_sample": selected_sample,
            "oracle_sample": oracle_sample,
            "sample_options": sample_options,
        }
    return index


def _coerce_choice(entry: dict[str, object], choice: str | None, fallback: str | None) -> str | None:
    sample_options = entry["sample_options"]
    if choice in {"selected", "oracle-best"}:
        resolved = entry["selected_sample"] if choice == "selected" else entry["oracle_sample"]
        return resolved if resolved in sample_options else fallback
    if choice in sample_options:
        return choice
    return fallback


def _option_labels(entry: dict[str, object]) -> list[tuple[str, str]]:
    options = [("selected", "selected"), ("oracle-best", "oracle-best")]
    sample_options = entry["sample_options"]
    for sample_name, sample in sorted(
        sample_options.items(),
        key=lambda item: (
            -(item[1].get("sample_ranking_score") or float("-inf")),
            item[0],
        ),
    ):
        seed = sample.get("seed") or "seed"
        rmsd = sample.get("rmsd")
        ranking = sample.get("sample_ranking_score")
        label = f"{sample_name} ({seed}"
        if ranking is not None:
            label += f", rank={ranking:.3f}"
        if rmsd is not None:
            label += f", rmsd={rmsd:.3f}"
        label += ")"
        options.append((label, sample_name))
    return options


def _render_single_view(record: dict[str, object], *, title: str, width: int, height: int) -> str:
    reference_path = Path(record["reference_path"])
    model_path = record["model_path"]
    if model_path is None or not Path(model_path).exists():
        return (
            f"<div><h4>{title}</h4><p>Model file is missing.</p>"
            f"<p>reference: {reference_path}</p></div>"
        )

    py3Dmol = _load_py3dmol()
    if py3Dmol is None:
        return (
            f"<div><h4>{title}</h4>"
            f"<p>py3Dmol is unavailable.</p>"
            f"<p>reference: {reference_path}</p>"
            f"<p>predicted: {model_path}</p></div>"
        )

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(reference_path.read_text(encoding="utf-8"), _viewer_format(reference_path))
    viewer.setStyle({"model": 0}, {"cartoon": {"color": "#94a3b8"}})
    viewer.addModel(Path(model_path).read_text(encoding="utf-8"), _viewer_format(Path(model_path)))
    viewer.setStyle({"model": 1}, {"cartoon": {"color": "#0f766e"}})
    viewer.zoomTo()
    return viewer._make_html()  # noqa: SLF001


def _render_compare_html(
    entry: dict[str, object],
    *,
    left_choice: str | None,
    right_choice: str | None,
    width: int,
    height: int,
) -> str:
    left_sample = _coerce_choice(
        entry,
        left_choice,
        entry["selected_sample"] or entry["oracle_sample"],
    )
    right_sample = _coerce_choice(
        entry,
        right_choice,
        entry["oracle_sample"] or entry["selected_sample"],
    )
    sample_options = entry["sample_options"]

    def build_record(sample_name: str | None) -> dict[str, object]:
        sample = {} if sample_name is None else sample_options.get(sample_name, {})
        return {
            "reference_path": entry["reference_path"],
            "model_path": sample.get("model_path"),
        }

    left_html = _render_single_view(
        build_record(left_sample),
        title=f"{entry['pdb_id']} left",
        width=width,
        height=height,
    )
    right_html = _render_single_view(
        build_record(right_sample),
        title=f"{entry['pdb_id']} right",
        width=width,
        height=height,
    )

    def metrics_block(sample_name: str | None, label: str) -> str:
        if sample_name is None or sample_name not in sample_options:
            return f"<div><strong>{label}</strong><br/>missing</div>"
        sample = sample_options[sample_name]
        return (
            f"<div><strong>{label}</strong><br/>"
            f"sample: {sample_name}<br/>"
            f"seed: {sample.get('seed') or 'NA'}<br/>"
            f"rank: {sample.get('sample_ranking_score')}<br/>"
            f"RMSD: {sample.get('rmsd')}<br/>"
            f"avg pLDDT: {sample.get('avg_plddt')}</div>"
        )

    return (
        f"<div>"
        f"<p><strong>{entry['pdb_id']}</strong> length={entry['length']} "
        f"selected={entry['selected_sample']} oracle-best={entry['oracle_sample']}</p>"
        f"<div style='display:flex; gap:16px; align-items:flex-start;'>"
        f"<div style='flex:1'>{metrics_block(left_sample, 'left')}</div>"
        f"<div style='flex:1'>{metrics_block(right_sample, 'right')}</div>"
        f"</div>"
        f"<div style='display:flex; gap:16px; align-items:flex-start; margin-top:12px;'>"
        f"<div style='flex:1'>{left_html}</div>"
        f"<div style='flex:1'>{right_html}</div>"
        f"</div>"
        f"</div>"
    )


def display_result_structures(
    result,
    *,
    pdb_ids: str | Iterable[str] | None = None,
    max_items: int = 3,
    width: int = 520,
    height: int = 360,
) -> None:
    HTML, Markdown, display = _load_display_dependencies()

    entries = list(_build_pdb_index(result).values())
    if pdb_ids:
        selected_ids = (
            {item.strip().upper() for item in pdb_ids.split()}
            if isinstance(pdb_ids, str)
            else {str(item).strip().upper() for item in pdb_ids}
        )
        entries = [entry for entry in entries if str(entry["pdb_id"]).upper() in selected_ids]
    if max_items is not None:
        entries = entries[:max_items]

    if not entries:
        display(Markdown("No successful structures were available for inline viewing."))
        return

    for entry in entries:
        html = _render_compare_html(
            entry,
            left_choice="selected",
            right_choice="oracle-best",
            width=width,
            height=height,
        )
        display(HTML(html))


def build_structure_browser(
    result,
    *,
    width: int = 520,
    height: int = 360,
):
    HTML, Markdown, display = _load_display_dependencies()
    try:
        import ipywidgets as widgets
    except Exception:
        display(
            Markdown(
                "Interactive structure browser requires `ipywidgets` in the notebook kernel."
            )
        )
        return None

    entries = _build_pdb_index(result)
    if not entries:
        display(Markdown("No successful structures were available for interactive viewing."))
        return None

    pdb_ids = sorted(entries)
    pdb_dropdown = widgets.Dropdown(
        options=pdb_ids,
        value=pdb_ids[0],
        description="PDB",
        layout=widgets.Layout(width="240px"),
    )
    left_dropdown = widgets.Dropdown(description="Left", layout=widgets.Layout(width="480px"))
    right_dropdown = widgets.Dropdown(description="Right", layout=widgets.Layout(width="480px"))
    output = widgets.Output()

    def sync_sample_options(selected_pdb: str) -> None:
        entry = entries[selected_pdb]
        options = _option_labels(entry)
        left_dropdown.options = options
        right_dropdown.options = options
        left_dropdown.value = "selected"
        right_dropdown.value = "oracle-best"

    def refresh(*_args) -> None:
        entry = entries[pdb_dropdown.value]
        with output:
            output.clear_output()
            display(
                HTML(
                    _render_compare_html(
                        entry,
                        left_choice=left_dropdown.value,
                        right_choice=right_dropdown.value,
                        width=width,
                        height=height,
                    )
                )
            )

    def on_pdb_change(change) -> None:  # noqa: ANN001
        if change.get("name") != "value":
            return
        sync_sample_options(change["new"])
        refresh()

    pdb_dropdown.observe(on_pdb_change, names="value")
    left_dropdown.observe(refresh, names="value")
    right_dropdown.observe(refresh, names="value")

    sync_sample_options(pdb_dropdown.value)
    refresh()

    ui = widgets.VBox(
        [
            widgets.HBox([pdb_dropdown]),
            widgets.HBox([left_dropdown, right_dropdown]),
            output,
        ]
    )
    display(ui)
    return ui
