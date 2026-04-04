from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


def _write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _svg_header(width: int, height: int, title: str) -> list[str]:
    escaped_title = html.escape(title)
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2937; }",
        ".axis { stroke: #94a3b8; stroke-width: 1; }",
        ".grid { stroke: #e2e8f0; stroke-width: 1; }",
        ".sample-point { fill: #94a3b8; opacity: 0.35; }",
        ".point { fill: #0f766e; opacity: 0.9; }",
        ".selected-ring { fill: none; stroke: #0f766e; stroke-width: 2; opacity: 0.95; }",
        ".bar { fill: #2563eb; opacity: 0.85; }",
        ".line { stroke: #dc2626; stroke-width: 2; }",
        "</style>",
        f'<text x="24" y="32" font-size="20" font-weight="700">{escaped_title}</text>',
    ]


def _write_placeholder_svg(path: str | Path, *, title: str, message: str) -> Path:
    lines = _svg_header(960, 240, title)
    lines.append(
        f'<text x="24" y="92" font-size="16">{html.escape(message)}</text>'
    )
    lines.append("</svg>")
    return _write_text(path, "\n".join(lines))


def _scale(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if domain_max == domain_min:
        return (range_min + range_max) / 2.0
    fraction = (value - domain_min) / (domain_max - domain_min)
    return range_min + fraction * (range_max - range_min)


def write_scatter_svg(
    results_df: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str = "Protein length vs OpenFold3 RMSD",
    regression: dict[str, float] | None = None,
    sample_points_df: pd.DataFrame | None = None,
) -> Path:
    selected = results_df[
        (results_df["status"] == "ok")
        & results_df["total_protein_length"].notna()
        & results_df["model_selected_rmsd"].notna()
    ].copy()
    samples = pd.DataFrame()
    if sample_points_df is not None and not sample_points_df.empty:
        samples = sample_points_df[
            sample_points_df["total_protein_length"].notna()
            & sample_points_df["rmsd_after_superposition"].notna()
        ].copy()
    if selected.empty and samples.empty:
        return _write_placeholder_svg(
            output_path,
            title=title,
            message="No successful rows with numeric length and RMSD were available.",
        )

    width = 960
    height = 540
    left = 90
    right = 40
    top = 70
    bottom = 70

    plot_width = width - left - right
    plot_height = height - top - bottom

    x_source = (
        samples["total_protein_length"].astype(float)
        if not samples.empty
        else selected["total_protein_length"].astype(float)
    )
    y_source = (
        samples["rmsd_after_superposition"].astype(float)
        if not samples.empty
        else selected["model_selected_rmsd"].astype(float)
    )
    x_values = x_source
    y_values = y_source
    x_padding = max((x_values.max() - x_values.min()) * 0.05, 1.0)
    y_padding = max((y_values.max() - y_values.min()) * 0.1, 0.1)
    x_min = float(x_values.min() - x_padding)
    x_max = float(x_values.max() + x_padding)
    y_min = max(0.0, float(y_values.min() - y_padding))
    y_max = float(y_values.max() + y_padding)

    lines = _svg_header(width, height, title)

    for tick in range(6):
        fraction = tick / 5.0
        y = top + fraction * plot_height
        x = left + fraction * plot_width
        y_value = y_max - fraction * (y_max - y_min)
        x_value = x_min + fraction * (x_max - x_min)

        lines.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" />'
        )
        lines.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" />'
        )
        lines.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" font-size="12" text-anchor="end">{y_value:.2f}</text>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{height - bottom + 24}" font-size="12" text-anchor="middle">{x_value:.0f}</text>'
        )

    lines.append(
        f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" />'
    )
    lines.append(
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" />'
    )
    lines.append(
        f'<text x="{left + plot_width / 2:.1f}" y="{height - 20}" font-size="14" text-anchor="middle">Total protein length</text>'
    )
    lines.append(
        f'<text x="24" y="{top + plot_height / 2:.1f}" font-size="14" transform="rotate(-90 24,{top + plot_height / 2:.1f})" text-anchor="middle">RMSD after superposition</text>'
    )

    if regression is not None:
        slope = regression.get("slope")
        intercept = regression.get("intercept")
        if slope is not None and intercept is not None:
            x1 = x_min
            y1 = slope * x1 + intercept
            x2 = x_max
            y2 = slope * x2 + intercept
            px1 = _scale(x1, x_min, x_max, left, width - right)
            py1 = _scale(y1, y_min, y_max, height - bottom, top)
            px2 = _scale(x2, x_min, x_max, left, width - right)
            py2 = _scale(y2, y_min, y_max, height - bottom, top)
            lines.append(
                f'<line class="line" x1="{px1:.1f}" y1="{py1:.1f}" x2="{px2:.1f}" y2="{py2:.1f}" />'
            )

    if not samples.empty:
        for row in samples.to_dict(orient="records"):
            x = _scale(
                float(row["total_protein_length"]),
                x_min,
                x_max,
                left,
                width - right,
            )
            y = _scale(
                float(row["rmsd_after_superposition"]),
                y_min,
                y_max,
                height - bottom,
                top,
            )
            label = html.escape(
                f"{row.get('pdb_id')} / {row.get('sample', 'sample')} / {row.get('seed') or 'seed'}"
            )
            lines.append(
                f'<circle class="sample-point" cx="{x:.1f}" cy="{y:.1f}" r="3"><title>{label}</title></circle>'
            )

    show_labels = len(selected) <= 18
    for row in selected.to_dict(orient="records"):
        x = _scale(float(row["total_protein_length"]), x_min, x_max, left, width - right)
        y = _scale(float(row["model_selected_rmsd"]), y_min, y_max, height - bottom, top)
        label = html.escape(str(row["pdb_id"]))
        lines.append(f'<circle class="point" cx="{x:.1f}" cy="{y:.1f}" r="5"><title>{label}</title></circle>')
        lines.append(f'<circle class="selected-ring" cx="{x:.1f}" cy="{y:.1f}" r="8"></circle>')
        if show_labels:
            lines.append(
                f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-size="11">{label}</text>'
            )

    legend_x = width - right - 220
    legend_y = top + 8
    lines.append(f'<circle class="sample-point" cx="{legend_x:.1f}" cy="{legend_y:.1f}" r="4"></circle>')
    lines.append(
        f'<text x="{legend_x + 12:.1f}" y="{legend_y + 4:.1f}" font-size="12">all samples / seeds</text>'
    )
    lines.append(f'<circle class="point" cx="{legend_x:.1f}" cy="{legend_y + 20:.1f}" r="5"></circle>')
    lines.append(
        f'<circle class="selected-ring" cx="{legend_x:.1f}" cy="{legend_y + 20:.1f}" r="8"></circle>'
    )
    lines.append(
        f'<text x="{legend_x + 12:.1f}" y="{legend_y + 24:.1f}" font-size="12">selected sample per PDB</text>'
    )

    lines.append("</svg>")
    return _write_text(output_path, "\n".join(lines))


def write_binned_svg(
    binned_df: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str = "Binned RMSD by protein length quantile",
) -> Path:
    if binned_df.empty:
        return _write_placeholder_svg(
            output_path,
            title=title,
            message="No successful rows were available to build quantile bins.",
        )

    width = 960
    height = 420
    left = 80
    right = 40
    top = 70
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    values = binned_df["mean_model_selected_rmsd"].astype(float)
    max_value = max(float(values.max()), 0.1)
    bar_count = len(binned_df)
    gap = 24
    bar_width = max(40.0, (plot_width - gap * (bar_count - 1)) / max(bar_count, 1))

    lines = _svg_header(width, height, title)
    lines.append(
        f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" />'
    )
    lines.append(
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" />'
    )
    lines.append(
        f'<text x="{left + plot_width / 2:.1f}" y="{height - 20}" font-size="14" text-anchor="middle">Length quantile bin</text>'
    )
    lines.append(
        f'<text x="24" y="{top + plot_height / 2:.1f}" font-size="14" transform="rotate(-90 24,{top + plot_height / 2:.1f})" text-anchor="middle">Mean model-selected RMSD</text>'
    )

    for tick in range(6):
        fraction = tick / 5.0
        y = top + fraction * plot_height
        y_value = max_value - fraction * max_value
        lines.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" />'
        )
        lines.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" font-size="12" text-anchor="end">{y_value:.2f}</text>'
        )

    for index, row in enumerate(binned_df.to_dict(orient="records")):
        bar_height = _scale(
            float(row["mean_model_selected_rmsd"]),
            0.0,
            max_value,
            0.0,
            plot_height,
        )
        x = left + index * (bar_width + gap)
        y = height - bottom - bar_height
        label = html.escape(str(row["length_bin"]))
        mean_value = float(row["mean_model_selected_rmsd"])
        lines.append(
            f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}">'
            f'<title>{label}: mean RMSD={mean_value:.3f}</title></rect>'
        )
        lines.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - bottom + 20}" font-size="11" text-anchor="middle">{label}</text>'
        )
        lines.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" font-size="11" text-anchor="middle">{mean_value:.2f}</text>'
        )

    lines.append("</svg>")
    return _write_text(output_path, "\n".join(lines))
