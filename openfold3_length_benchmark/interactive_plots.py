from __future__ import annotations

from typing import Any


def _load_display_dependencies() -> tuple[Any, Any]:
    try:
        from IPython.display import Markdown, display
    except Exception as exc:
        raise RuntimeError(
            "Interactive notebook plots require IPython in the kernel environment."
        ) from exc
    return Markdown, display


def _pick_plotly_renderer(pio: Any) -> str | None:
    preferred_renderers = (
        "plotly_mimetype",
        "jupyterlab",
        "notebook_connected",
        "notebook",
        "iframe_connected",
        "iframe",
    )
    available = set(pio.renderers)
    for name in preferred_renderers:
        if name in available:
            return name
    return None


def display_interactive_scatter(result) -> object | None:
    Markdown, display = _load_display_dependencies()

    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception:
        display(
            Markdown(
                "Interactive Plotly scatter is unavailable because `plotly` is not installed in the notebook kernel. "
                "Use the SVG fallback below or install `plotly` in the same kernel environment."
            )
        )
        return None

    results_df = result.results_df.copy()
    sample_points_df = result.sample_points_df.copy()

    valid_selected = results_df[
        (results_df["status"] == "ok")
        & results_df["total_protein_length"].notna()
        & results_df["model_selected_rmsd"].notna()
    ].copy()
    valid_samples = sample_points_df[
        sample_points_df["total_protein_length"].notna()
        & sample_points_df["rmsd_after_superposition"].notna()
    ].copy()

    if valid_selected.empty and valid_samples.empty:
        display(Markdown("No successful rows were available for the interactive scatter."))
        return None

    fig = go.Figure()
    categories = [
        ("single_chain", "#0f766e", "#94a3b8", "Single-chain"),
        ("double_chain", "#c2410c", "#fdba74", "Double-chain"),
    ]

    for chain_group, selected_color, sample_color, label in categories:
        sample_group = valid_samples[valid_samples["chain_group"] == chain_group].copy()
        if not sample_group.empty:
            fig.add_trace(
                go.Scatter(
                    x=sample_group["total_protein_length"],
                    y=sample_group["rmsd_after_superposition"],
                    mode="markers",
                    name=f"{label} samples",
                    marker={"size": 7, "color": sample_color, "opacity": 0.45},
                    customdata=sample_group[
                        [
                            "pdb_id",
                            "sample",
                            "seed",
                            "sample_ranking_score",
                            "avg_plddt",
                        ]
                    ].values,
                    hovertemplate=(
                        "PDB: %{customdata[0]}<br>"
                        "Length: %{x}<br>"
                        "RMSD: %{y:.3f}<br>"
                        "Sample: %{customdata[1]}<br>"
                        "Seed: %{customdata[2]}<br>"
                        "Rank: %{customdata[3]}<br>"
                        "avg pLDDT: %{customdata[4]}<extra></extra>"
                    ),
                )
            )

        selected_group = valid_selected[valid_selected["chain_group"] == chain_group].copy()
        if not selected_group.empty:
            fig.add_trace(
                go.Scatter(
                    x=selected_group["total_protein_length"],
                    y=selected_group["model_selected_rmsd"],
                    mode="markers",
                    name=f"{label} selected",
                    marker={
                        "size": 11,
                        "color": selected_color,
                        "line": {"width": 2, "color": selected_color},
                        "opacity": 0.95,
                    },
                    customdata=selected_group[
                        [
                            "pdb_id",
                            "model_selected_sample",
                            "oracle_sample",
                            "sample_ranking_score",
                            "avg_plddt",
                        ]
                    ].values,
                    hovertemplate=(
                        "PDB: %{customdata[0]}<br>"
                        "Length: %{x}<br>"
                        "Selected RMSD: %{y:.3f}<br>"
                        "Selected sample: %{customdata[1]}<br>"
                        "Oracle-best sample: %{customdata[2]}<br>"
                        "Selected rank: %{customdata[3]}<br>"
                        "avg pLDDT: %{customdata[4]}<extra></extra>"
                    ),
                )
            )

            if len(selected_group) >= 2:
                slope, intercept = __import__("numpy").polyfit(
                    selected_group["total_protein_length"].astype(float),
                    selected_group["model_selected_rmsd"].astype(float),
                    1,
                )
                x_min = float(selected_group["total_protein_length"].min())
                x_max = float(selected_group["total_protein_length"].max())
                fig.add_trace(
                    go.Scatter(
                        x=[x_min, x_max],
                        y=[slope * x_min + intercept, slope * x_max + intercept],
                        mode="lines",
                        name=f"{label} trend",
                        line={"color": selected_color, "width": 2},
                        hoverinfo="skip",
                    )
                )

    fig.update_layout(
        title="Protein length vs OpenFold3 RMSD",
        xaxis_title="Total protein length",
        yaxis_title="RMSD after superposition",
        template="plotly_white",
        height=620,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    fig.update_traces(cliponaxis=False)

    renderer = _pick_plotly_renderer(pio)
    if renderer is not None:
        pio.renderers.default = renderer

    try:
        fig.show(renderer=renderer)
    except Exception as exc:
        display(
            Markdown(
                "Interactive Plotly scatter could not be rendered in this notebook frontend. "
                f"Falling back to SVG. Renderer: `{renderer or 'default'}`. Error: `{exc}`"
            )
        )
        return None

    return fig
