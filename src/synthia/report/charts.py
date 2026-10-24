"""Inline SVG charts for the report. No JavaScript, no external assets; every chart has a table twin.

Marks follow one spec: bars at most 24px thick with a 4px rounded data end, hairline solid gridlines, text in
text tokens rather than series colours, a legend whenever two series share a chart, and native ``<title>``
tooltips on every mark. Colours come from CSS variables declared in the template so dark mode swaps them in one
place; heatmap cells are the one exception because their colour encodes a value.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence

import numpy as np

SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b"]
DIVERGING_NEGATIVE = "#2a78d6"
DIVERGING_POSITIVE = "#e34948"
DIVERGING_MID = "#f0efec"


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _mix(a: str, b: str, t: float) -> str:
    ra, ga, ba = _hex_to_rgb(a)
    rb, gb, bb = _hex_to_rgb(b)
    t = min(max(t, 0.0), 1.0)
    return f"#{round(ra + (rb - ra) * t):02x}{round(ga + (gb - ga) * t):02x}{round(ba + (bb - ba) * t):02x}"


def sequential_color(value: float, vmax: float) -> str:
    """One hue, light to dark, for a magnitude in [0, vmax]."""
    if vmax <= 0 or not math.isfinite(value):
        return SEQUENTIAL[0]
    pos = min(max(value / vmax, 0.0), 1.0) * (len(SEQUENTIAL) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(SEQUENTIAL) - 1)
    return _mix(SEQUENTIAL[lo], SEQUENTIAL[hi], pos - lo)


def diverging_color(value: float) -> str:
    """Blue for negative, red for positive, neutral grey at zero; value in [-1, 1]."""
    if not math.isfinite(value):
        return DIVERGING_MID
    if value < 0:
        return _mix(DIVERGING_MID, DIVERGING_NEGATIVE, min(-value, 1.0))
    return _mix(DIVERGING_MID, DIVERGING_POSITIVE, min(value, 1.0))


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def bar_chart_with_ci(
    rows: Sequence[tuple[str, float, float, float]], title: str, x_max: float | None = None, unit: str = ""
) -> str:
    """Horizontal bars (one series) with a whisker for the confidence interval and the value at the tip."""
    if not rows:
        return ""
    label_w, plot_w, row_h, pad = 150, 420, 26, 12
    height = pad * 2 + row_h * len(rows) + 24
    width = label_w + plot_w + 90
    top = max([r[3] for r in rows] + [1e-9]) if x_max is None else x_max
    top = max(top * 1.08, 1e-9)
    scale = plot_w / top
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">']
    ticks = _nice_ticks(top, 4)
    for t in ticks:
        x = label_w + t * scale
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{height - 24}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{height - 8}" text-anchor="middle">{_fmt(t, 2)}</text>')
    for i, (label, value, low, high) in enumerate(rows):
        y = pad + i * row_h
        bar_w = max(value * scale, 0.0)
        parts.append(f'<text class="label" x="{label_w - 8}" y="{y + 17}" text-anchor="end">{_esc(label)}</text>')
        parts.append(
            f'<path class="series-1" d="{_rounded_bar(label_w, y + 3, bar_w, 20, 4)}">'
            f"<title>{_esc(label)}: {_fmt(value)} [{_fmt(low)}, {_fmt(high)}]{_esc(unit)}</title></path>"
        )
        x1, x2 = label_w + low * scale, label_w + high * scale
        parts.append(f'<line class="whisker" x1="{x1:.1f}" y1="{y + 13}" x2="{x2:.1f}" y2="{y + 13}"/>')
        parts.append(f'<text class="value" x="{max(label_w + bar_w, x2) + 8:.1f}" y="{y + 17}">{_fmt(value)}</text>')
    parts.append(f'<line class="axis" x1="{label_w}" y1="{pad}" x2="{label_w}" y2="{height - 24}"/>')
    parts.append("</svg>")
    return "".join(parts)


def _rounded_bar(x: float, y: float, w: float, h: float, r: float) -> str:
    """Bar growing to the right: square at the baseline, rounded at the data end."""
    if w <= r:
        return f"M{x:.1f},{y:.1f} h{max(w, 0.5):.1f} v{h:.1f} h{-max(w, 0.5):.1f} Z"
    return (
        f"M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} a{r},{r} 0 0 1 {-r},{r} "
        f"h{-(w - r):.1f} Z"
    )


def _rounded_column(x: float, y: float, w: float, h: float, r: float) -> str:
    """Column growing upward from the baseline: square at the base, rounded at the data end."""
    if h <= r:
        return f"M{x:.1f},{y + h:.1f} v{-max(h, 0.5):.1f} h{w:.1f} v{max(h, 0.5):.1f} Z"
    return (
        f"M{x:.1f},{y + h:.1f} v{-(h - r):.1f} a{r},{r} 0 0 1 {r},{-r} h{w - 2 * r:.1f} a{r},{r} 0 0 1 {r},{r} "
        f"v{h - r:.1f} Z"
    )


def _nice_ticks(top: float, count: int) -> list[float]:
    if top <= 0:
        return [0.0]
    raw = top / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((m for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), default=10) * magnitude
    return [i * step for i in range(int(top // step) + 1)]


def grouped_columns(labels: Sequence[str], real: Sequence[float], synthetic: Sequence[float], title: str) -> str:
    """Two-series column chart (real then synthetic per bin) with a 2px surface gap between neighbours."""
    n = len(labels)
    if n == 0:
        return ""
    plot_h, pad_top, pad_bottom, left = 120, 10, 34, 36
    slot = max(12.0, min(40.0, 420 / n))
    width = left + slot * n + 12
    height = pad_top + plot_h + pad_bottom
    top = max(list(real) + list(synthetic) + [1e-9]) * 1.1
    col_w = max((slot - 6) / 2, 2.0)
    parts = [f'<svg class="chart" viewBox="0 0 {width:.0f} {height}" role="img" aria-label="{_esc(title)}">']
    for t in _nice_ticks(top, 3):
        y = pad_top + plot_h - t / top * plot_h
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - 12:.0f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 4}" y="{y + 4:.1f}" text-anchor="end">{_pct(t)}</text>')
    for i, label in enumerate(labels):
        x0 = left + i * slot + 2
        for k, (value, cls, name) in enumerate(
            ((real[i], "series-1", "real"), (synthetic[i], "series-2", "synthetic"))
        ):
            h = max(value / top * plot_h, 0.0)
            x = x0 + k * (col_w + 2)
            parts.append(
                f'<path class="{cls}" d="{_rounded_column(x, pad_top + plot_h - h, col_w, h, min(4, col_w / 2))}">'
                f"<title>{_esc(name)} {_esc(label)}: {_pct(value)}</title></path>"
            )
        if n <= 12 or i % max(1, n // 8) == 0:
            tick_x = x0 + col_w + 1
            parts.append(
                f'<text class="tick" x="{tick_x:.1f}" y="{height - 18}" text-anchor="middle">'
                f"{_esc(_short(label))}</text>"
            )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{pad_top + plot_h}" x2="{width - 12:.0f}" y2="{pad_top + plot_h}"/>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _pct(value: float) -> str:
    return f"{100 * value:.0f}%" if value >= 0.1 or value == 0 else f"{100 * value:.1f}%"


def _short(label: str) -> str:
    return label if len(label) <= 7 else label[:6] + "."


def heatmap(matrix: np.ndarray, labels: Sequence[str], title: str, diverging: bool, vmax: float = 1.0) -> str:
    """Square cell grid; diverging for signed associations, one-hue sequential for absolute deltas."""
    n = len(labels)
    cell, left, top_pad = 22, 118, 100
    width, height = left + cell * n + 10, top_pad + cell * n + 10
    parts = [f'<svg class="chart heat" viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">']
    for j, label in enumerate(labels):
        x = left + j * cell + cell / 2
        parts.append(
            f'<text class="tick" transform="translate({x:.1f},{top_pad - 6}) rotate(-60)" text-anchor="start">'
            f"{_esc(label)}</text>"
        )
        parts.append(
            f'<text class="tick" x="{left - 6}" y="{top_pad + j * cell + cell / 2 + 4:.1f}" text-anchor="end">'
            f"{_esc(label)}</text>"
        )
    for i in range(n):
        for j in range(n):
            value = float(matrix[i, j])
            color = diverging_color(value) if diverging else sequential_color(abs(value), vmax)
            parts.append(
                f'<rect x="{left + j * cell + 1}" y="{top_pad + i * cell + 1}" width="{cell - 2}" height="{cell - 2}" '
                f'rx="2" fill="{color}"><title>{_esc(labels[i])} x {_esc(labels[j])}: {_fmt(value)}</title></rect>'
            )
    parts.append("</svg>")
    return "".join(parts)


def matrix_table(matrix: np.ndarray, labels: Sequence[str]) -> str:
    """Table twin of a heatmap."""
    head = "".join(f"<th>{_esc(label)}</th>" for label in labels)
    rows = []
    for i, label in enumerate(labels):
        cells = "".join(f"<td>{_fmt(float(matrix[i, j]), 2)}</td>" for j in range(len(labels)))
        rows.append(f"<tr><th>{_esc(label)}</th>{cells}</tr>")
    return f'<table class="matrix"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
