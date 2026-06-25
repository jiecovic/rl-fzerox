# src/rl_fzerox/ui/watch/view/panels/visuals/live_chart/axis.py
"""Axis range and label helpers for Watch live charts.

The plot renderer uses these helpers to keep value scaling separate from
pygame drawing. Summaries reuse the formatter for reference-line labels.
"""

from __future__ import annotations

from rl_fzerox.ui.watch.view.panels.visuals.live_chart.elements import (
    _PlotReferenceLine,
    _PlotSeries,
)
from rl_fzerox.ui.watch.view.panels.visuals.live_chart.geometry import _float_tuple_stats


def _plot_value_range(
    *,
    series: tuple[_PlotSeries, ...],
    reference_lines: tuple[_PlotReferenceLine, ...],
) -> tuple[float, float]:
    y_min = 0.0
    y_max = 0.0
    for line in series:
        if line.y_values:
            stats = _float_tuple_stats(line.y_values)
            y_min = min(y_min, stats.minimum)
            y_max = max(y_max, stats.maximum)
    for line in reference_lines:
        y_min = min(y_min, line.value)
        y_max = max(y_max, line.value)
    return y_min, y_max


def _format_axis_value(value: float) -> str:
    abs_value = abs(value)
    if abs_value < 1e-9:
        return "0"
    if abs_value < 0.001:
        return f"{value:.1e}"
    if abs_value >= 1000.0:
        return f"{value / 1000.0:.1f}k"
    if abs_value >= 100.0:
        return f"{value:.0f}"
    if abs_value >= 10.0:
        return f"{value:.1f}"
    if abs_value >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"
