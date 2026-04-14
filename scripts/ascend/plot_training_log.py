#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import sys
from pathlib import Path


STEP_PATTERN = re.compile(r"step:(?P<step>\d+)\s+-\s+(?P<body>.*)")
METRIC_PATTERN = re.compile(r"(?P<key>[^:]+?):(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
DEFAULT_METRIC_CANDIDATES = [
    "val-core/hiyouga/geometry3k/acc/mean@1",
    "val-aux/hiyouga/geometry3k/reward/mean@1",
    "critic/rewards/mean",
    "critic/score/mean",
    "actor/entropy",
    "perf/throughput",
]
SERIES_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def parse_step_metrics(log_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for raw_line in log_path.read_text(errors="replace").splitlines():
        line = _strip_ansi(raw_line)
        match = STEP_PATTERN.search(line)
        if not match:
            continue
        row: dict[str, float] = {"step": float(match.group("step"))}
        for metric_match in METRIC_PATTERN.finditer(match.group("body")):
            key = metric_match.group("key").strip()
            if key.startswith("- "):
                key = key[2:].strip()
            row[key] = float(metric_match.group("value"))
        rows.append(row)
    return rows


def choose_metrics(rows: list[dict[str, float]], requested_metrics: list[str]) -> list[str]:
    if not rows:
        return []

    available_keys = {
        key
        for row in rows
        for key in row
        if key not in {"step", "training/global_step"}
    }
    if requested_metrics:
        chosen = [metric for metric in requested_metrics if metric in available_keys]
    else:
        chosen = [metric for metric in DEFAULT_METRIC_CANDIDATES if metric in available_keys]
        if not chosen:
            chosen = sorted(available_keys)[:4]
    return chosen


def write_csv(rows: list[dict[str, float]], csv_path: Path, metrics: list[str]) -> None:
    fieldnames = ["step", "training/global_step", *metrics]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def render_svg(rows: list[dict[str, float]], metrics: list[str], title: str) -> str:
    width = 1200
    height = 720
    margin_left = 80
    margin_right = 30
    margin_top = 60
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_values = [row.get("training/global_step", row["step"]) for row in rows]
    x_min = min(x_values)
    x_max = max(x_values)
    if x_min == x_max:
        x_max = x_min + 1.0

    y_values = [row[metric] for metric in metrics for row in rows if metric in row]
    y_min = min(y_values)
    y_max = max(y_values)
    if math.isclose(y_min, y_max):
        delta = 1.0 if y_min == 0 else abs(y_min) * 0.05
        y_min -= delta
        y_max += delta

    def x_to_px(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_width

    def y_to_px(y: float) -> float:
        return margin_top + plot_height - (y - y_min) / (y_max - y_min) * plot_height

    x_ticks = _linspace(x_min, x_max, 6)
    y_ticks = _linspace(y_min, y_max, 6)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="32" font-family="sans-serif" font-size="22" font-weight="bold">{html.escape(title)}</text>',
    ]

    for x_tick in x_ticks:
        x = x_to_px(x_tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#374151">{_fmt_tick(x_tick)}</text>'
        )
    for y_tick in y_ticks:
        y = y_to_px(y_tick)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#374151">{_fmt_tick(y_tick)}</text>'
        )

    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#111827">global step</text>'
    )
    parts.append(
        f'<text x="22" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 22 {margin_top + plot_height / 2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#111827">metric value</text>'
    )

    legend_y = margin_top - 8
    legend_x = margin_left
    for idx, metric in enumerate(metrics):
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        metric_points = [
            (x_to_px(row.get("training/global_step", row["step"])), y_to_px(row[metric]))
            for row in rows
            if metric in row
        ]
        if len(metric_points) >= 2:
            path = " ".join(
                [f"M {metric_points[0][0]:.2f} {metric_points[0][1]:.2f}"]
                + [f"L {x:.2f} {y:.2f}" for x, y in metric_points[1:]]
            )
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        elif len(metric_points) == 1:
            x, y = metric_points[0]
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')

        label_x = legend_x + idx * 250
        parts.append(f'<line x1="{label_x}" y1="{legend_y}" x2="{label_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(
            f'<text x="{label_x + 30}" y="{legend_y + 4}" font-family="sans-serif" font-size="12" fill="#111827">{html.escape(metric)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract verl training metrics from a log file and draw an SVG curve.")
    parser.add_argument("--log", required=True, type=Path, help="Training log path.")
    parser.add_argument("--output", required=True, type=Path, help="Output SVG path.")
    parser.add_argument("--csv", type=Path, help="Optional CSV export path.")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric key to plot. Can be passed multiple times. Defaults to a small recommended set if omitted.",
    )
    args = parser.parse_args()

    rows = parse_step_metrics(args.log)
    if not rows:
        print(f"No step metrics found in {args.log}", file=sys.stderr)
        return 1

    metrics = choose_metrics(rows, args.metric)
    if not metrics:
        print("No requested metrics were found in the log.", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    title = f"Training Curves: {args.log.name}"
    args.output.write_text(render_svg(rows, metrics, title))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, args.csv, metrics)

    print(f"Parsed {len(rows)} step line(s).")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Wrote SVG: {args.output}")
    if args.csv:
        print(f"Wrote CSV: {args.csv}")
    return 0


def _fmt_tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _linspace(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    step = (end - start) / (count - 1)
    return [start + idx * step for idx in range(count)]


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


if __name__ == "__main__":
    raise SystemExit(main())
