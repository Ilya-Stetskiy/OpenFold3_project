from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def _load_display_dependencies():
    try:
        from IPython.display import Markdown, SVG, clear_output, display
    except Exception as exc:
        raise RuntimeError(
            "Notebook controls require IPython display helpers in the kernel environment."
        ) from exc
    return Markdown, SVG, clear_output, display


def display_case_timeline(result, case_key: str) -> object | None:
    Markdown, SVG, _clear_output, display = _load_display_dependencies()
    if result.case_results_df.empty:
        display(Markdown("No runtime benchmark results are available yet."))
        return None

    benchmark_mode, pdb_id = case_key.split(":", 1)
    frame = result.case_results_df[
        (result.case_results_df["benchmark_mode"] == benchmark_mode)
        & (result.case_results_df["pdb_id"] == pdb_id)
    ].copy()
    if frame.empty:
        display(Markdown(f"No rows matched `{case_key}`."))
        return None

    display(frame)
    for row in frame.to_dict(orient="records"):
        timeline_path = row.get("timeline_svg_path")
        if not timeline_path:
            continue
        path = Path(timeline_path)
        if not path.exists():
            continue
        display(Markdown(f"**{row['run_mode']}** timeline: `{path}`"))
        display(SVG(path.read_text(encoding="utf-8")))
    return frame


def build_notebook_controls(
    *,
    runtime,
    config_getter: Callable[[], dict[str, Any]],
    state: dict[str, Any] | None = None,
):
    state = state if state is not None else {}
    try:
        import ipywidgets as widgets
    except ImportError:
        return "ipywidgets is not installed. Use the preview/run cells below."

    try:
        Markdown, SVG, clear_output, display = _load_display_dependencies()
    except RuntimeError:
        return "IPython display helpers are unavailable. Use the preview/run cells below."

    from .orchestration import run_runtime_benchmark
    from .interop import preview_entries

    output = widgets.Output()
    preview_button = widgets.Button(description="Preview", button_style="info")
    run_button = widgets.Button(description="Run", button_style="success")
    refresh_button = widgets.Button(description="Refresh", button_style="")
    case_selector = widgets.Dropdown(description="Case", options=[("None", "")], value="")

    def _case_options(result) -> list[tuple[str, str]]:
        if result.case_results_df.empty:
            return [("None", "")]
        options = [("None", "")]
        case_keys = sorted(
            {
                f"{row['benchmark_mode']}:{row['pdb_id']}"
                for row in result.case_results_df.to_dict(orient="records")
            }
        )
        options.extend((case_key.replace(":", " / "), case_key) for case_key in case_keys)
        return options

    def _show_preview() -> None:
        config = config_getter()
        trace_ids = config.get("pipeline_trace_pdb_ids")
        union_ids = [config["pdb_ids_text"]]
        if trace_ids:
            union_ids.append(trace_ids)
        preview_df = preview_entries(
            " ".join(str(item) for item in union_ids if item),
            cache_dir=config.get("cache_dir"),
            max_entries=config.get("max_entries"),
        )
        state["preview_df"] = preview_df
        display(preview_df)

    def _show_run() -> None:
        config = config_getter()
        modes_value = config.get("modes", ("cold", "warm"))
        if isinstance(modes_value, str):
            modes = (modes_value,)
        else:
            modes = tuple(modes_value)
        result = run_runtime_benchmark(
            runtime=runtime,
            pdb_ids=config["pdb_ids_text"],
            modes=modes,
            sampling_interval_seconds=float(config.get("sampling_interval_seconds", 1.0)),
            output_root=config.get("output_root"),
            runner_yaml=config.get("runner_yaml"),
            max_entries=config.get("max_entries"),
            cache_dir=config.get("cache_dir"),
            pipeline_trace_pdb_ids=config.get("pipeline_trace_pdb_ids"),
        )
        state["result"] = result
        case_selector.options = _case_options(result)
        display(result.case_results_df)
        display(Markdown(f"Run root: `{result.run_root}`"))
        display(Markdown(f"Manifest: `{result.manifest_path}`"))
        display(Markdown(f"Summary: `{result.summary_path}`"))

    def _refresh() -> None:
        if "result" in state:
            result = state["result"]
            display(result.case_results_df)
            if case_selector.value:
                display_case_timeline(result, case_selector.value)
            else:
                display(Markdown("Select a case to show cold/warm timelines inline."))
            return
        if "preview_df" in state:
            display(state["preview_df"])
            return
        display(Markdown("No preview or run has been executed yet."))

    def _wrap(callback: Callable[[], None]) -> Callable[[Any], None]:
        def _handler(_button) -> None:
            with output:
                clear_output(wait=True)
                callback()

        return _handler

    preview_button.on_click(_wrap(_show_preview))
    run_button.on_click(_wrap(_show_run))
    refresh_button.on_click(_wrap(_refresh))
    case_selector.observe(lambda _change: _wrap(_refresh)(None), names="value")

    header = widgets.HTML(
        "<b>OpenFold3 runtime benchmark controls</b><br/>Preview entries, launch cold/warm runs, and inspect case timelines."
    )
    controls = widgets.HBox([preview_button, run_button, refresh_button])
    return widgets.VBox([header, controls, case_selector, output])
