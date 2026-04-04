from __future__ import annotations

from typing import Any


def _load_display_dependencies() -> tuple[Any, Any, Any]:
    try:
        from IPython.display import HTML, Markdown, display
    except Exception as exc:
        raise RuntimeError(
            "Interactive notebook plots require IPython in the kernel environment."
        ) from exc
    return HTML, Markdown, display


def display_interactive_scatter(result) -> object | None:
    HTML, Markdown, display = _load_display_dependencies()

    try:
        import plotly.graph_objects as go
        from plotly.offline import plot as plotly_plot
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

    metric_specs = [
        {
            "key": "ca",
            "label": "CA",
            "selected_column": "model_selected_rmsd_ca"
            if "model_selected_rmsd_ca" in results_df.columns
            else "model_selected_rmsd",
            "sample_column": "rmsd_ca"
            if "rmsd_ca" in sample_points_df.columns
            else "rmsd_after_superposition",
            "symbol": "circle",
            "line_dash": "solid",
        },
        {
            "key": "backbone",
            "label": "Backbone",
            "selected_column": "model_selected_rmsd_backbone",
            "sample_column": "rmsd_backbone",
            "symbol": "diamond",
            "line_dash": "dash",
        },
    ]

    valid_selected_frames: dict[str, Any] = {}
    valid_sample_frames: dict[str, Any] = {}
    for metric in metric_specs:
        selected_column = metric["selected_column"]
        sample_column = metric["sample_column"]
        if selected_column in results_df.columns:
            valid_selected_frames[metric["key"]] = results_df[
                (results_df["status"] == "ok")
                & results_df["total_protein_length"].notna()
                & results_df[selected_column].notna()
            ].copy()
        if sample_column in sample_points_df.columns:
            valid_sample_frames[metric["key"]] = sample_points_df[
                sample_points_df["total_protein_length"].notna()
                & sample_points_df[sample_column].notna()
            ].copy()

    if not any(not frame.empty for frame in valid_selected_frames.values()) and not any(
        not frame.empty for frame in valid_sample_frames.values()
    ):
        display(Markdown("No successful rows were available for the interactive scatter."))
        return None

    fig = go.Figure()
    categories = [
        ("single_chain", "#0f766e", "#94a3b8", "Single-chain"),
        ("double_chain", "#c2410c", "#fdba74", "Double-chain"),
    ]

    for chain_group, selected_color, sample_color, label in categories:
        for metric in metric_specs:
            sample_group = valid_sample_frames.get(metric["key"])
            if sample_group is not None:
                sample_group = sample_group[sample_group["chain_group"] == chain_group].copy()
            if sample_group is not None and not sample_group.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sample_group["total_protein_length"],
                        y=sample_group[metric["sample_column"]],
                        mode="markers",
                        name=f"{label} {metric['label']} samples",
                        marker={
                            "size": 7,
                            "color": sample_color,
                            "opacity": 0.45,
                            "symbol": metric["symbol"],
                        },
                        visible=True if metric["key"] == "ca" else "legendonly",
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
                            f"Metric: {metric['label']}<br>"
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

            selected_group = valid_selected_frames.get(metric["key"])
            if selected_group is not None:
                selected_group = selected_group[
                    selected_group["chain_group"] == chain_group
                ].copy()
            if selected_group is not None and not selected_group.empty:
                fig.add_trace(
                    go.Scatter(
                        x=selected_group["total_protein_length"],
                        y=selected_group[metric["selected_column"]],
                        mode="markers",
                        name=f"{label} {metric['label']} selected",
                        marker={
                            "size": 11,
                            "color": selected_color,
                            "line": {"width": 2, "color": selected_color},
                            "opacity": 0.95,
                            "symbol": metric["symbol"],
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
                            f"Metric: {metric['label']}<br>"
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
                        selected_group[metric["selected_column"]].astype(float),
                        1,
                    )
                    x_min = float(selected_group["total_protein_length"].min())
                    x_max = float(selected_group["total_protein_length"].max())
                    fig.add_trace(
                        go.Scatter(
                            x=[x_min, x_max],
                            y=[slope * x_min + intercept, slope * x_max + intercept],
                            mode="lines",
                            name=f"{label} {metric['label']} trend",
                            line={
                                "color": selected_color,
                                "width": 2,
                                "dash": metric["line_dash"],
                            },
                            hoverinfo="skip",
                        )
                    )

    fig.update_layout(
        title="Protein length vs OpenFold3 RMSD (CA and Backbone)",
        xaxis_title="Total protein length",
        yaxis_title="RMSD after superposition",
        template="plotly_white",
        height=620,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    fig.update_traces(cliponaxis=False)
    try:
        html = plotly_plot(
            fig,
            include_plotlyjs="cdn",
            output_type="div",
            config={
                "responsive": True,
                "displaylogo": False,
                "toImageButtonOptions": {"format": "svg"},
            },
        )
        display(HTML(html))
    except Exception as exc:
        display(
            Markdown(
                "Interactive Plotly scatter could not be rendered in this notebook frontend. "
                f"Falling back to SVG. Error: `{exc}`"
            )
        )
        return None

    return fig
