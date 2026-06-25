# src/rl_fzerox/ui/watch/view/panels/visuals/live_chart/style.py
"""Shared dimensions and colors for Watch live chart cards."""

from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.view.screen.theme import PALETTE


@dataclass(frozen=True, slots=True)
class _LiveChartStyle:
    block_gap: int = 16
    block_padding: int = 10
    title_gap: int = 4
    plot_gap: int = 8
    plot_height: int = 184
    plot_min_width: int = 120
    plot_max_buckets: int = 240
    plot_axis_width: int = 42
    axis_label_gap: int = 3
    legend_margin: int = 6
    legend_padding: int = 5
    legend_swatch_width: int = 12
    legend_swatch_gap: int = 4
    legend_item_gap: int = 8
    legend_row_gap: int = 3
    radius: int = 8
    line_width: int = 2
    step_reward_color: tuple[int, int, int] = (111, 211, 255)
    progress_color: tuple[int, int, int] = PALETTE.text_accent
    position_multiplier_color: tuple[int, int, int] = (255, 196, 91)
    combined_multiplier_color: tuple[int, int, int] = (195, 165, 255)
    average_speed_color: tuple[int, int, int] = (255, 196, 91)
    edge_ratio_color: tuple[int, int, int] = (255, 132, 132)
    outside_edge_excess_ratio_color: tuple[int, int, int] = (175, 220, 135)
    height_above_ground_color: tuple[int, int, int] = (190, 165, 255)
    min_height_color: tuple[int, int, int] = (255, 196, 91)
    grid_color: tuple[int, int, int] = PALETTE.panel_border
    zero_line_color: tuple[int, int, int] = PALETTE.text_muted
    reference_line_dash: int = 8
    reference_line_gap: int = 5
    reference_tick_length: int = 6
    reference_label_padding: int = 2
    block_fill: tuple[int, int, int] = (16, 20, 26)


LIVE_CHART_STYLE = _LiveChartStyle()
