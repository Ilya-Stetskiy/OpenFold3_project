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
        ".point-cold { fill: #0f766e; opacity: 0.92; }",
        ".point-warm { fill: #c2410c; opacity: 0.92; }",
        ".line-cold { stroke: #0f766e; stroke-width: 2; }",
        ".line-warm { stroke: #c2410c; stroke-width: 2; }",
        ".line-mean { stroke: #0f766e; stroke-width: 2; }",
        ".line-max { stroke: #c2410c; stroke-width: 2; stroke-dasharray: 8 4; }",
        ".timeline-event { stroke: #64748b; stroke-width: 1; opacity: 0.7; }",
        ".timeline-cpu { stroke: #0f766e; stroke-width: 2; fill: none; }",
        ".timeline-rss { stroke: #2563eb; stroke-width: 2; fill: none; }",
        ".timeline-gpu { stroke: #c2410c; stroke-width: 2; fill: none; }",
        ".timeline-mem { stroke: #7c3aed; stroke-width: 2; fill: none; }",
        "</style>",
        f'<text x="24" y="30" font-size="20" font-weight="700">{escaped_title}</text>',
    ]


def _write_placeholder_svg(path: str | Path, *, title: str, message: str) -> Path:
    lines = _svg_header(960, 220, title)
    lines.append(f'<text x="24" y="90" font-size="16">{html.escape(message)}</text>')
    lines.append("</svg>")
    return _write_text(path, "\n".join(lines))


def _scale(
    value: float,
    domain_min: float,
    domain_max: float,
    range_min: float,
    range_max: float,
) -> float:
    if domain_max == domain_min:
        return (range_min + range_max) / 2.0
    fraction = (value - domain_min) / (domain_max - domain_min)
    return range_min + fraction * (range_max - range_min)


def write_metric_scatter_svg(
    results_df: pd.DataFrame,
    *,
    output_path: str | Path,
    y_column: str,
    title: str,
    y_label: str,
) -> Path:
    frame = results_df[
        (results_df["status"] == "ok")
        & (results_df["benchmark_mode"] == "size_sweep_core")
        & results_df["total_protein_length"].notna()
        & results_df[y_column].notna()
    ].copy()
    if frame.empty:
        return _write_placeholder_svg(
            output_path,
            title=title,
            message="No successful size-sweep rows were available for this metric.",
        )

    width = 960
    height = 540
    left = 90
    right = 40
    top = 70
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = frame["total_protein_length"].astype(float)
    y_values = frame[y_column].astype(float)
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
        f'<text x="24" y="{top + plot_height / 2:.1f}" font-size="14" transform="rotate(-90 24,{top + plot_height / 2:.1f})" text-anchor="middle">{html.escape(y_label)}</text>'
    )

    legend_y = 54
    legend_x = width - 220
    for offset, (run_mode, css_class) in enumerate(
        [("cold", "point-cold"), ("warm", "point-warm")]
    ):
        y = legend_y + offset * 20
        lines.append(
            f'<circle class="{css_class}" cx="{legend_x}" cy="{y}" r="5"></circle>'
        )
        lines.append(
            f'<text x="{legend_x + 12}" y="{y + 4}" font-size="12">{run_mode}</text>'
        )

    show_labels = len(frame) <= 18
    for row in frame.to_dict(orient="records"):
        css_class = "point-cold" if row["run_mode"] == "cold" else "point-warm"
        x = _scale(float(row["total_protein_length"]), x_min, x_max, left, width - right)
        y = _scale(float(row[y_column]), y_min, y_max, height - bottom, top)
        label = html.escape(f"{row['pdb_id']} ({row['run_mode']})")
        lines.append(f'<circle class="{css_class}" cx="{x:.1f}" cy="{y:.1f}" r="5"><title>{label}</title></circle>')
        if show_labels:
            lines.append(
                f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-size="11">{label}</text>'
            )

    lines.append("</svg>")
    return _write_text(output_path, "\n".join(lines))


def write_gpu_util_scatter_svg(
    results_df: pd.DataFrame,
    *,
    output_path: str | Path,
) -> Path:
    frame = results_df[
        (results_df["status"] == "ok")
        & (results_df["benchmark_mode"] == "size_sweep_core")
        & results_df["total_protein_length"].notna()
        & (
            results_df["mean_gpu_util_percent"].notna()
            | results_df["max_gpu_util_percent"].notna()
        )
    ].copy()
    if frame.empty:
        return _write_placeholder_svg(
            output_path,
            title="Protein length vs GPU utilization",
            message="No successful GPU telemetry rows were available for this metric.",
        )

    width = 960
    height = 540
    left = 90
    right = 40
    top = 70
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = frame["total_protein_length"].astype(float)
    y_values = pd.concat(
        [
            frame["mean_gpu_util_percent"].dropna().astype(float),
            frame["max_gpu_util_percent"].dropna().astype(float),
        ]
    )
    x_padding = max((x_values.max() - x_values.min()) * 0.05, 1.0)
    y_padding = max((y_values.max() - y_values.min()) * 0.1, 1.0)
    x_min = float(x_values.min() - x_padding)
    x_max = float(x_values.max() + x_padding)
    y_min = max(0.0, float(y_values.min() - y_padding))
    y_max = min(100.0, float(y_values.max() + y_padding))

    lines = _svg_header(width, height, "Protein length vs GPU utilization")
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
            f'<text x="{left - 12}" y="{y + 5:.1f}" font-size="12" text-anchor="end">{y_value:.1f}</text>'
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
        f'<text x="24" y="{top + plot_height / 2:.1f}" font-size="14" transform="rotate(-90 24,{top + plot_height / 2:.1f})" text-anchor="middle">GPU utilization (%)</text>'
    )
    lines.append('<line class="line-mean" x1="720" y1="50" x2="760" y2="50" />')
    lines.append('<text x="770" y="54" font-size="12">mean</text>')
    lines.append('<line class="line-max" x1="820" y1="50" x2="860" y2="50" />')
    lines.append('<text x="870" y="54" font-size="12">max</text>')

    for _, row in frame.iterrows():
        x = _scale(float(row["total_protein_length"]), x_min, x_max, left, width - right)
        if pd.notna(row["mean_gpu_util_percent"]):
            y = _scale(float(row["mean_gpu_util_percent"]), y_min, y_max, height - bottom, top)
            lines.append(f'<circle class="point-cold" cx="{x:.1f}" cy="{y:.1f}" r="4"></circle>')
        if pd.notna(row["max_gpu_util_percent"]):
            y = _scale(float(row["max_gpu_util_percent"]), y_min, y_max, height - bottom, top)
            lines.append(f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" fill="#c2410c"></rect>')

    lines.append("</svg>")
    return _write_text(output_path, "\n".join(lines))


def _line_path(
    values: list[tuple[float, float]],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    if not values:
        return ""
    parts: list[str] = []
    for index, (x_value, y_value) in enumerate(values):
        x = _scale(x_value, x_min, x_max, left, left + width)
        y = _scale(y_value, y_min, y_max, top + height, top)
        parts.append(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}")
    return " ".join(parts)


def write_timeline_svg(
    samples_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    if samples_df.empty:
        return _write_placeholder_svg(
            output_path,
            title=title,
            message="No telemetry samples were collected for this case.",
        )

    ticks = samples_df.sort_values(by=["sample_seq", "pid"]).drop_duplicates(
        subset=["sample_seq"],
        keep="first",
    )
    if ticks.empty:
        return _write_placeholder_svg(
            output_path,
            title=title,
            message="No aggregate telemetry rows were available for this case.",
        )

    width = 1200
    height = 820
    left = 90
    right = 40
    top = 70
    panel_height = 150
    panel_gap = 30
    plot_width = width - left - right
    x_min = float(ticks["relative_seconds"].min())
    x_max = float(max(ticks["relative_seconds"].max(), x_min + 1e-6))

    panels = [
        (
            "CPU %",
            "timeline-cpu",
            ticks[["relative_seconds", "tree_total_cpu_percent"]]
            .dropna()
            .astype(float)
            .values.tolist(),
        ),
        (
            "RSS (GiB)",
            "timeline-rss",
            [
                [float(row["relative_seconds"]), float(row["tree_total_rss_bytes"]) / (1024**3)]
                for row in ticks.to_dict(orient="records")
            ],
        ),
        (
            "GPU util %",
            "timeline-gpu",
            ticks[["relative_seconds", "gpu_util_percent_max"]]
            .dropna()
            .astype(float)
            .values.tolist(),
        ),
        (
            "GPU mem (GiB)",
            "timeline-mem",
            [
                [float(row["relative_seconds"]), float(row["gpu_memory_used_total_mb"]) / 1024.0]
                for row in ticks.to_dict(orient="records")
                if row.get("gpu_memory_used_total_mb") is not None
            ],
        ),
    ]

    lines = _svg_header(width, height, title)
    for panel_index, (label, css_class, values) in enumerate(panels):
        panel_top = top + panel_index * (panel_height + panel_gap)
        panel_bottom = panel_top + panel_height
        if values:
            y_values = [item[1] for item in values]
            y_min = min(y_values)
            y_max = max(y_values)
            if y_min == y_max:
                y_max = y_min + 1.0
        else:
            y_min = 0.0
            y_max = 1.0

        lines.append(
            f'<line class="axis" x1="{left}" y1="{panel_bottom}" x2="{width - right}" y2="{panel_bottom}" />'
        )
        lines.append(
            f'<line class="axis" x1="{left}" y1="{panel_top}" x2="{left}" y2="{panel_bottom}" />'
        )
        lines.append(
            f'<text x="{left - 12}" y="{panel_top + 14}" font-size="12" text-anchor="end">{html.escape(label)}</text>'
        )

        for tick in range(5):
            fraction = tick / 4.0
            y = panel_top + fraction * panel_height
            y_value = y_max - fraction * (y_max - y_min)
            lines.append(
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" />'
            )
            lines.append(
                f'<text x="{left - 12}" y="{y + 4:.1f}" font-size="11" text-anchor="end">{y_value:.2f}</text>'
            )

        path_data = _line_path(
            [(float(x), float(y)) for x, y in values],
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            left=left,
            top=panel_top,
            width=plot_width,
            height=panel_height,
        )
        if path_data:
            lines.append(f'<path class="{css_class}" d="{path_data}"></path>')

        if not events_df.empty:
            for _, event_row in events_df.iterrows():
                event_x = _scale(
                    float(event_row["relative_seconds"]),
                    x_min,
                    x_max,
                    left,
                    width - right,
                )
                lines.append(
                    f'<line class="timeline-event" x1="{event_x:.1f}" y1="{panel_top}" x2="{event_x:.1f}" y2="{panel_bottom}" />'
                )
                if panel_index == 0:
                    label_text = html.escape(
                        f"{event_row.get('stage', '?')}:{event_row.get('event', '?')}"
                    )
                    lines.append(
                        f'<text x="{event_x + 3:.1f}" y="{panel_top + 12:.1f}" font-size="10" transform="rotate(-90 {event_x + 3:.1f},{panel_top + 12:.1f})">{label_text}</text>'
                    )

    for tick in range(6):
        fraction = tick / 5.0
        x = left + fraction * plot_width
        x_value = x_min + fraction * (x_max - x_min)
        lines.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + 4 * panel_height + 3 * panel_gap}" />'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{height - 20}" font-size="12" text-anchor="middle">{x_value:.1f}s</text>'
        )

    lines.append("</svg>")
    return _write_text(output_path, "\n".join(lines))
