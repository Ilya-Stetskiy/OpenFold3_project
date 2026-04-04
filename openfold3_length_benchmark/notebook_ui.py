from __future__ import annotations

from typing import Any, Callable

from .composition import preview_entries
from .orchestration import run_length_benchmark


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
        from IPython.display import clear_output, display
    except ImportError:
        return "IPython display helpers are unavailable. Use the preview/run cells below."

    output = widgets.Output()
    preview_button = widgets.Button(description="Preview", button_style="info")
    run_button = widgets.Button(description="Run", button_style="success")
    refresh_button = widgets.Button(description="Refresh", button_style="")

    def _show_preview() -> None:
        config = config_getter()
        preview_df = preview_entries(
            config["pdb_ids_text"],
            max_entries=config.get("max_entries"),
        )
        state["preview_df"] = preview_df
        display(preview_df)

    def _show_run() -> None:
        config = config_getter()
        result = run_length_benchmark(
            runtime=runtime,
            pdb_ids=config["pdb_ids_text"],
            atom_set=config.get("atom_set", "ca"),
            use_msa_server=config.get("use_msa_server", True),
            num_diffusion_samples=config.get("num_diffusion_samples", 1),
            num_model_seeds=config.get("num_model_seeds", 1),
            runner_yaml=config.get("runner_yaml"),
            output_root=config.get("output_root"),
            max_entries=config.get("max_entries"),
        )
        state["result"] = result
        display(result.results_df)
        print("Run root:", result.run_root)
        for label, path in result.plot_paths.items():
            print(f"{label}: {path}")

    def _refresh() -> None:
        if "result" in state:
            display(state["result"].results_df)
            print("Run root:", state["result"].run_root)
            return
        if "preview_df" in state:
            display(state["preview_df"])
            return
        print("No preview or run has been executed yet.")

    def _wrap(callback: Callable[[], None]) -> Callable[[Any], None]:
        def _handler(_button) -> None:
            with output:
                clear_output(wait=True)
                callback()

        return _handler

    preview_button.on_click(_wrap(_show_preview))
    run_button.on_click(_wrap(_show_run))
    refresh_button.on_click(_wrap(_refresh))

    header = widgets.HTML(
        "<b>Optional controls</b><br/>If widgets are unavailable, run the cells below in order."
    )
    controls = widgets.HBox([preview_button, run_button, refresh_button])
    return widgets.VBox([header, controls, output])
