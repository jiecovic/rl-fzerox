# src/rl_fzerox/ui/watch/view/panels/visuals/live.py
from __future__ import annotations

from rl_fzerox.ui.watch.live_series import EpisodeLiveSeriesSnapshot
from rl_fzerox.ui.watch.view.panels.core.format import _int_info
from rl_fzerox.ui.watch.view.panels.rendering.text import _fit_text
from rl_fzerox.ui.watch.view.panels.visuals.live_chart.geometry import _plot_points
from rl_fzerox.ui.watch.view.panels.visuals.live_chart.plot import (
    _draw_chart_block,
    _plot_legend_rows,
    _PlotSeries,
)
from rl_fzerox.ui.watch.view.panels.visuals.live_chart.style import LIVE_CHART_STYLE
from rl_fzerox.ui.watch.view.panels.visuals.live_chart.summaries import (
    _edge_ratio_summary,
    _height_above_ground_reference_lines,
    _height_above_ground_summary,
    _ko_star_events_summary,
    _multiplier_range,
    _multiplier_summary,
    _outside_edge_excess_ratio_summary,
    _speed_reference_lines,
    _speed_summary,
    _step_reward_summary,
)
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import (
    PygameModule,
    PygameSurface,
    ViewerFonts,
)

__all__ = (
    "LIVE_CHART_STYLE",
    "_draw_live_tab",
    "_ko_star_events_summary",
    "_PlotSeries",
    "_plot_legend_rows",
    "_plot_points",
    "_speed_summary",
)


def _draw_live_tab(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    height: int,
    info: dict[str, object],
    live_series: EpisodeLiveSeriesSnapshot | None,
) -> None:
    env_step = _int_info(info, "episode_step")
    title_surface = fonts.section.render("Live episode", True, PALETTE.text_primary)
    subtitle_surface = fonts.small.render(
        f"decision-step traces for the current episode · frame {env_step}",
        True,
        PALETTE.text_muted,
    )
    screen.blit(title_surface, (x, y))
    subtitle_y = y + title_surface.get_height() + LIVE_CHART_STYLE.title_gap
    screen.blit(subtitle_surface, (x, subtitle_y))
    ko_events_y = subtitle_y + subtitle_surface.get_height() + LIVE_CHART_STYLE.title_gap
    ko_events_surface = fonts.small.render(
        _fit_text(fonts.small, _ko_star_events_summary(live_series), width),
        True,
        PALETTE.text_muted,
    )
    screen.blit(ko_events_surface, (x, ko_events_y))
    chart_y = ko_events_y + ko_events_surface.get_height() + LAYOUT.section_gap

    header_height = chart_y - y
    plot_height = _plot_height_for_rows(
        fonts,
        available_height=max(0, height - header_height),
        row_count=3,
    )
    block_height = _chart_block_height(fonts, plot_height=plot_height)
    block_width = (width - LIVE_CHART_STYLE.block_gap) // 2
    right_x = x + block_width + LIVE_CHART_STYLE.block_gap
    second_row_y = chart_y + block_height + LIVE_CHART_STYLE.block_gap
    third_row_y = second_row_y + block_height + LIVE_CHART_STYLE.block_gap
    _draw_chart_block(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=x,
        y=chart_y,
        width=block_width,
        height=block_height,
        title="Speed",
        summary=_speed_summary(live_series),
        x_values=() if live_series is None else live_series.env_steps,
        y_values=() if live_series is None else live_series.speed_kph,
        color=LIVE_CHART_STYLE.progress_color,
        fixed_range=None,
        zero_line=False,
        plot_height=plot_height,
        reference_lines=_speed_reference_lines(live_series),
    )
    _draw_chart_block(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=right_x,
        y=chart_y,
        width=block_width,
        height=block_height,
        title="Step reward",
        summary=_step_reward_summary(live_series),
        x_values=() if live_series is None else live_series.env_steps,
        y_values=() if live_series is None else live_series.step_rewards,
        color=LIVE_CHART_STYLE.step_reward_color,
        fixed_range=None,
        zero_line=True,
        plot_height=plot_height,
    )
    _draw_chart_block(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=x,
        y=second_row_y,
        width=block_width,
        height=block_height,
        title="Edge ratio",
        summary=_edge_ratio_summary(live_series),
        x_values=() if live_series is None else live_series.env_steps,
        y_values=() if live_series is None else live_series.edge_ratio,
        color=LIVE_CHART_STYLE.edge_ratio_color,
        fixed_range=(-1.0, 1.0),
        zero_line=True,
        plot_height=plot_height,
    )
    _draw_chart_block(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=right_x,
        y=second_row_y,
        width=block_width,
        height=block_height,
        title="Outside edge excess ratio",
        summary=_outside_edge_excess_ratio_summary(live_series),
        x_values=() if live_series is None else live_series.env_steps,
        y_values=() if live_series is None else live_series.outside_edge_excess_ratio,
        color=LIVE_CHART_STYLE.outside_edge_excess_ratio_color,
        fixed_range=None,
        zero_line=False,
        plot_height=plot_height,
    )
    _draw_chart_block(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=x,
        y=third_row_y,
        width=block_width,
        height=block_height,
        title="Height over ground",
        summary=_height_above_ground_summary(live_series),
        x_values=() if live_series is None else live_series.env_steps,
        y_values=() if live_series is None else live_series.height_above_ground,
        color=LIVE_CHART_STYLE.height_above_ground_color,
        fixed_range=None,
        zero_line=True,
        plot_height=plot_height,
        reference_lines=_height_above_ground_reference_lines(live_series),
    )
    _draw_chart_block(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=right_x,
        y=third_row_y,
        width=block_width,
        height=block_height,
        title="Progress multipliers",
        summary=_multiplier_summary(live_series),
        x_values=() if live_series is None else live_series.env_steps,
        y_values=() if live_series is None else live_series.progress_speed_multiplier,
        color=LIVE_CHART_STYLE.progress_color,
        series_label="speed",
        fixed_range=_multiplier_range(live_series),
        zero_line=False,
        plot_height=plot_height,
        extra_series=(
            _PlotSeries(
                y_values=() if live_series is None else live_series.position_progress_multiplier,
                color=LIVE_CHART_STYLE.position_multiplier_color,
                label="position",
            ),
            _PlotSeries(
                y_values=(
                    () if live_series is None else live_series.progress_speed_position_multiplier
                ),
                color=LIVE_CHART_STYLE.combined_multiplier_color,
                label="total",
            ),
        ),
    )


def _plot_height_for_rows(
    fonts: ViewerFonts,
    *,
    available_height: int,
    row_count: int,
) -> int:
    non_plot_height = _chart_block_non_plot_height(fonts)
    gap_height = max(0, row_count - 1) * LIVE_CHART_STYLE.block_gap
    usable_height = max(0, available_height - gap_height)
    fit_height = (usable_height // max(1, row_count)) - non_plot_height
    return max(82, min(LIVE_CHART_STYLE.plot_height, fit_height))


def _chart_block_height(fonts: ViewerFonts, *, plot_height: int) -> int:
    return _chart_block_non_plot_height(fonts) + plot_height


def _chart_block_non_plot_height(fonts: ViewerFonts) -> int:
    title_height = fonts.small.render("Step reward", True, PALETTE.text_primary).get_height()
    summary_height = fonts.small.render(
        "waiting for episode steps",
        True,
        PALETTE.text_muted,
    ).get_height()
    return (
        (2 * LIVE_CHART_STYLE.block_padding)
        + title_height
        + LIVE_CHART_STYLE.title_gap
        + summary_height
        + LIVE_CHART_STYLE.plot_gap
    )
