# src/rl_fzerox/ui/watch/view/panels/visuals/live_chart/elements.py
"""Small chart element records shared by live-chart drawing and summaries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _PlotSeries:
    y_values: tuple[float, ...]
    color: tuple[int, int, int]
    label: str | None = None


@dataclass(frozen=True, slots=True)
class _PlotReferenceLine:
    value: float
    color: tuple[int, int, int]
    label: str | None = None


@dataclass(frozen=True, slots=True)
class _PlotLegendItem:
    label: str
    color: tuple[int, int, int]
    width: int
