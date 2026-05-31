# src/rl_fzerox/ui/watch/view/panels/visuals/live.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.view.live_episode import EpisodeLiveSeriesSnapshot
from rl_fzerox.ui.watch.view.panels.core.format import _int_info
from rl_fzerox.ui.watch.view.panels.rendering.text import _fit_text
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import (
    PygameModule,
    PygameRect,
    PygameSurface,
    ViewerFonts,
)


@dataclass(frozen=True, slots=True)
class _LiveChartStyle:
    block_gap: int = 16
    block_padding: int = 10
    title_gap: int = 4
    plot_gap: int = 8
    plot_height: int = 184
    plot_min_width: int = 120
    radius: int = 8
    line_width: int = 2
    return_color: tuple[int, int, int] = (111, 211, 255)
    progress_color: tuple[int, int, int] = PALETTE.text_accent
    position_multiplier_color: tuple[int, int, int] = (255, 196, 91)
    combined_multiplier_color: tuple[int, int, int] = (195, 165, 255)
    lateral_offset_color: tuple[int, int, int] = (255, 132, 132)
    outside_edge_excess_ratio_color: tuple[int, int, int] = (175, 220, 135)
    future_segment_distance_color: tuple[int, int, int] = (190, 165, 255)
    grid_color: tuple[int, int, int] = PALETTE.panel_border
    zero_line_color: tuple[int, int, int] = PALETTE.text_muted
    block_fill: tuple[int, int, int] = (16, 20, 26)


LIVE_CHART_STYLE = _LiveChartStyle()


@dataclass(frozen=True, slots=True)
class _PlotSeries:
    y_values: tuple[float, ...]
    color: tuple[int, int, int]


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
    chart_y = subtitle_y + subtitle_surface.get_height() + LAYOUT.section_gap

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
    )
    _draw_chart_block(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=right_x,
        y=chart_y,
        width=block_width,
        height=block_height,
        title="Episode return",
        summary=_return_summary(live_series),
        x_values=() if live_series is None else live_series.env_steps,
        y_values=() if live_series is None else live_series.returns,
        color=LIVE_CHART_STYLE.return_color,
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
        title="Lateral offset",
        summary=_lateral_offset_summary(live_series),
        x_values=() if live_series is None else live_series.env_steps,
        y_values=() if live_series is None else live_series.lateral_offset,
        color=LIVE_CHART_STYLE.lateral_offset_color,
        fixed_range=None,
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
        title="Future segment distance",
        summary=_future_segment_distance_summary(live_series),
        x_values=() if live_series is None else live_series.env_steps,
        y_values=() if live_series is None else live_series.future_local_nearest_segment_distance,
        color=LIVE_CHART_STYLE.future_segment_distance_color,
        fixed_range=None,
        zero_line=False,
        plot_height=plot_height,
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
        fixed_range=_multiplier_range(live_series),
        zero_line=False,
        plot_height=plot_height,
        extra_series=(
            _PlotSeries(
                y_values=() if live_series is None else live_series.position_progress_multiplier,
                color=LIVE_CHART_STYLE.position_multiplier_color,
            ),
            _PlotSeries(
                y_values=(
                    () if live_series is None else live_series.progress_speed_position_multiplier
                ),
                color=LIVE_CHART_STYLE.combined_multiplier_color,
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
    title_height = fonts.small.render("Episode return", True, PALETTE.text_primary).get_height()
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


def _draw_chart_block(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    summary: str,
    x_values: tuple[int, ...],
    y_values: tuple[float, ...],
    color: tuple[int, int, int],
    fixed_range: tuple[float, float] | None,
    zero_line: bool,
    plot_height: int,
    extra_series: tuple[_PlotSeries, ...] = (),
) -> None:
    rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(
        screen,
        LIVE_CHART_STYLE.block_fill,
        rect,
        border_radius=LIVE_CHART_STYLE.radius,
    )
    pygame.draw.rect(
        screen,
        PALETTE.panel_border,
        rect,
        width=1,
        border_radius=LIVE_CHART_STYLE.radius,
    )

    inner_x = x + LIVE_CHART_STYLE.block_padding
    inner_y = y + LIVE_CHART_STYLE.block_padding
    inner_width = width - (2 * LIVE_CHART_STYLE.block_padding)
    title_surface = fonts.small.render(
        _fit_text(fonts.small, title, inner_width),
        True,
        PALETTE.text_primary,
    )
    summary_surface = fonts.small.render(
        _fit_text(fonts.small, summary, inner_width),
        True,
        PALETTE.text_muted,
    )
    screen.blit(title_surface, (inner_x, inner_y))
    summary_y = inner_y + title_surface.get_height() + LIVE_CHART_STYLE.title_gap
    screen.blit(summary_surface, (inner_x, summary_y))

    plot_y = summary_y + summary_surface.get_height() + LIVE_CHART_STYLE.plot_gap
    plot_rect = pygame.Rect(
        inner_x,
        plot_y,
        max(LIVE_CHART_STYLE.plot_min_width, inner_width),
        plot_height,
    )
    _draw_plot_background(pygame=pygame, screen=screen, rect=plot_rect)
    if not x_values or not y_values:
        waiting_surface = fonts.small.render(
            "waiting for episode steps",
            True,
            PALETTE.text_muted,
        )
        waiting_x = plot_rect.x + max(0, (plot_rect.width - waiting_surface.get_width()) // 2)
        waiting_y = plot_rect.y + max(0, (plot_rect.height - waiting_surface.get_height()) // 2)
        screen.blit(waiting_surface, (waiting_x, waiting_y))
        return
    _draw_plot_series(
        pygame=pygame,
        screen=screen,
        rect=plot_rect,
        x_values=x_values,
        series=(_PlotSeries(y_values=y_values, color=color), *extra_series),
        fixed_range=fixed_range,
        zero_line=zero_line,
    )


def _draw_plot_background(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    rect: PygameRect,
) -> None:
    pygame.draw.rect(screen, PALETTE.panel_background, rect, border_radius=6)
    pygame.draw.rect(screen, PALETTE.panel_border, rect, width=1, border_radius=6)


def _draw_plot_series(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    rect: PygameRect,
    x_values: tuple[int, ...],
    series: tuple[_PlotSeries, ...],
    fixed_range: tuple[float, float] | None,
    zero_line: bool,
) -> None:
    valid_series = tuple(line for line in series if len(line.y_values) == len(x_values))
    if not valid_series:
        return
    x_start = x_values[0]
    x_end = max(x_values[-1], x_start + 1)
    if fixed_range is None:
        series_values = tuple(value for line in valid_series for value in line.y_values)
        y_min = min(min(series_values), 0.0)
        y_max = max(max(series_values), 0.0)
        if y_min == y_max:
            span = max(1.0, abs(y_min) * 0.1)
            y_min -= span
            y_max += span
    else:
        y_min, y_max = fixed_range
    span = max(1e-6, y_max - y_min)
    mid_y = rect.y + rect.height // 2
    pygame.draw.line(
        screen,
        LIVE_CHART_STYLE.grid_color,
        (rect.x, mid_y),
        (rect.x + rect.width, mid_y),
        width=1,
    )
    if zero_line and y_min <= 0.0 <= y_max:
        zero_y = (
            rect.y + rect.height - 1 - int(round((0.0 - y_min) / span * max(1, rect.height - 1)))
        )
        pygame.draw.line(
            screen,
            LIVE_CHART_STYLE.zero_line_color,
            (rect.x, zero_y),
            (rect.x + rect.width, zero_y),
            width=1,
        )
    for line in valid_series:
        points = [
            (
                rect.x + int(round((step - x_start) / (x_end - x_start) * max(1, rect.width - 1))),
                rect.y
                + rect.height
                - 1
                - int(round((value - y_min) / span * max(1, rect.height - 1))),
            )
            for step, value in zip(x_values, line.y_values, strict=False)
        ]
        if len(points) == 1:
            pygame.draw.circle(screen, line.color, points[0], 3)
        else:
            pygame.draw.lines(screen, line.color, False, points, LIVE_CHART_STYLE.line_width)


def _speed_summary(live_series: EpisodeLiveSeriesSnapshot | None) -> str:
    if live_series is None or not live_series.speed_kph:
        return "now - · progress -"
    current_speed = live_series.speed_kph[-1]
    current_progress = live_series.current_progress * 100.0
    return f"now {current_speed:.1f} km/h · progress {current_progress:.1f}%"


def _return_summary(live_series: EpisodeLiveSeriesSnapshot | None) -> str:
    if live_series is None or not live_series.returns:
        return "now - · max progress -"
    current_return = live_series.current_return
    max_progress = live_series.max_progress * 100.0
    return f"now {current_return:+.4f} · max progress {max_progress:.1f}%"


def _multiplier_summary(live_series: EpisodeLiveSeriesSnapshot | None) -> str:
    if live_series is None or not live_series.progress_speed_position_multiplier:
        return "speed - · position - · total -"
    speed = live_series.progress_speed_multiplier[-1]
    position = live_series.position_progress_multiplier[-1]
    total = live_series.progress_speed_position_multiplier[-1]
    return f"speed {speed:.2f}x · position {position:.2f}x · total {total:.2f}x"


def _multiplier_range(live_series: EpisodeLiveSeriesSnapshot | None) -> tuple[float, float] | None:
    if live_series is None:
        return None
    values = (
        *live_series.progress_speed_multiplier,
        *live_series.position_progress_multiplier,
        *live_series.progress_speed_position_multiplier,
        1.0,
    )
    if not values:
        return None
    lower = max(0.0, min(values))
    upper = max(values)
    if lower == upper:
        return (max(0.0, lower - 0.1), upper + 0.1)
    padding = max(0.05, (upper - lower) * 0.1)
    return (max(0.0, lower - padding), upper + padding)


def _lateral_offset_summary(live_series: EpisodeLiveSeriesSnapshot | None) -> str:
    if live_series is None or not live_series.lateral_offset:
        return "now - · max |x| -"
    current_offset = live_series.lateral_offset[-1]
    max_absolute_offset = max(abs(value) for value in live_series.lateral_offset)
    return f"now {current_offset:+.1f} · max |x| {max_absolute_offset:.1f}"


def _outside_edge_excess_ratio_summary(
    live_series: EpisodeLiveSeriesSnapshot | None,
) -> str:
    if live_series is None or not live_series.outside_edge_excess_ratio:
        return "now - · max -"
    current_excess = live_series.outside_edge_excess_ratio[-1]
    max_excess = max(live_series.outside_edge_excess_ratio)
    return f"now {current_excess:.3f} · max {max_excess:.3f}"


def _future_segment_distance_summary(live_series: EpisodeLiveSeriesSnapshot | None) -> str:
    if live_series is None or not live_series.future_local_nearest_segment_distance:
        return "seg - · distance -"
    current_distance = live_series.future_local_nearest_segment_distance[-1]
    segment_index = live_series.current_future_local_nearest_segment_index
    segment_label = "-" if segment_index is None else str(segment_index)
    return f"seg {segment_label} · distance {current_distance:.1f}"
