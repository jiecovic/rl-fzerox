# src/rl_fzerox/ui/watch/view/panels/visuals/live.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.view.live_episode import EpisodeLiveSeriesSnapshot
from rl_fzerox.ui.watch.view.panels.core.format import _int_info
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
    grid_color: tuple[int, int, int] = PALETTE.panel_border
    zero_line_color: tuple[int, int, int] = PALETTE.text_muted
    block_fill: tuple[int, int, int] = (16, 20, 26)


LIVE_CHART_STYLE = _LiveChartStyle()


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
    del height
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

    block_height = _chart_block_height(fonts)
    _draw_chart_block(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=x,
        y=chart_y,
        width=width,
        height=block_height,
        title="Speed",
        summary=_speed_summary(live_series),
        x_values=() if live_series is None else live_series.env_steps,
        y_values=() if live_series is None else live_series.speed_kph,
        color=LIVE_CHART_STYLE.progress_color,
        fixed_range=None,
        zero_line=False,
    )
    _draw_chart_block(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=x,
        y=chart_y + block_height + LIVE_CHART_STYLE.block_gap,
        width=width,
        height=block_height,
        title="Episode return",
        summary=_return_summary(live_series),
        x_values=() if live_series is None else live_series.env_steps,
        y_values=() if live_series is None else live_series.returns,
        color=LIVE_CHART_STYLE.return_color,
        fixed_range=None,
        zero_line=True,
    )


def _chart_block_height(fonts: ViewerFonts) -> int:
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
        + LIVE_CHART_STYLE.plot_height
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
    title_surface = fonts.small.render(title, True, PALETTE.text_primary)
    summary_surface = fonts.small.render(summary, True, PALETTE.text_muted)
    screen.blit(title_surface, (inner_x, inner_y))
    summary_y = inner_y + title_surface.get_height() + LIVE_CHART_STYLE.title_gap
    screen.blit(summary_surface, (inner_x, summary_y))

    plot_y = summary_y + summary_surface.get_height() + LIVE_CHART_STYLE.plot_gap
    plot_rect = pygame.Rect(
        inner_x,
        plot_y,
        max(LIVE_CHART_STYLE.plot_min_width, inner_width),
        LIVE_CHART_STYLE.plot_height,
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
        y_values=y_values,
        color=color,
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
    y_values: tuple[float, ...],
    color: tuple[int, int, int],
    fixed_range: tuple[float, float] | None,
    zero_line: bool,
) -> None:
    if len(x_values) != len(y_values):
        return
    x_start = x_values[0]
    x_end = max(x_values[-1], x_start + 1)
    if fixed_range is None:
        y_min = min(min(y_values), 0.0)
        y_max = max(max(y_values), 0.0)
        if y_min == y_max:
            span = max(1.0, abs(y_min) * 0.1)
            y_min -= span
            y_max += span
    else:
        y_min, y_max = fixed_range
    span = max(1e-6, y_max - y_min)
    points = [
        (
            rect.x + int(round((step - x_start) / (x_end - x_start) * max(1, rect.width - 1))),
            rect.y
            + rect.height
            - 1
            - int(round((value - y_min) / span * max(1, rect.height - 1))),
        )
        for step, value in zip(x_values, y_values, strict=False)
    ]
    mid_y = rect.y + rect.height // 2
    pygame.draw.line(
        screen,
        LIVE_CHART_STYLE.grid_color,
        (rect.x, mid_y),
        (rect.x + rect.width, mid_y),
        width=1,
    )
    if zero_line and y_min <= 0.0 <= y_max:
        zero_y = rect.y + rect.height - 1 - int(
            round((0.0 - y_min) / span * max(1, rect.height - 1))
        )
        pygame.draw.line(
            screen,
            LIVE_CHART_STYLE.zero_line_color,
            (rect.x, zero_y),
            (rect.x + rect.width, zero_y),
            width=1,
        )
    if len(points) == 1:
        pygame.draw.circle(screen, color, points[0], 3)
        return
    pygame.draw.lines(screen, color, False, points, LIVE_CHART_STYLE.line_width)


def _speed_summary(live_series: EpisodeLiveSeriesSnapshot | None) -> str:
    if live_series is None or not live_series.speed_kph:
        return "speed - · progress - · return -"
    current_speed = live_series.speed_kph[-1]
    current_progress = live_series.current_progress * 100.0
    current_return = live_series.current_return
    return (
        f"speed {current_speed:.1f} km/h · progress {current_progress:.1f}% · "
        f"return {current_return:+.4f}"
    )


def _return_summary(live_series: EpisodeLiveSeriesSnapshot | None) -> str:
    if live_series is None or not live_series.returns:
        return "return - · progress max -"
    current_return = live_series.current_return
    max_progress = live_series.max_progress * 100.0
    return f"return {current_return:+.4f} · progress max {max_progress:.1f}%"
