# src/rl_fzerox/ui/watch/view/panels/visuals/live_chart/plot.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.view.panels.rendering.text import _fit_text
from rl_fzerox.ui.watch.view.panels.visuals.live_chart.geometry import (
    _float_tuple_stats,
    _plot_points,
)
from rl_fzerox.ui.watch.view.panels.visuals.live_chart.style import LIVE_CHART_STYLE
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import (
    PygameModule,
    PygameRect,
    PygameSurface,
    ViewerFonts,
)


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
    series_label: str | None = None,
    fixed_range: tuple[float, float] | None,
    zero_line: bool,
    plot_height: int,
    extra_series: tuple[_PlotSeries, ...] = (),
    reference_lines: tuple[_PlotReferenceLine, ...] = (),
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

    series = (_PlotSeries(y_values=y_values, color=color, label=series_label), *extra_series)
    plot_width = max(
        LIVE_CHART_STYLE.plot_min_width,
        inner_width - LIVE_CHART_STYLE.plot_axis_width,
    )
    legend_rows = _plot_legend_rows(
        fonts=fonts,
        rect_width=plot_width,
        series=series,
    )
    legend_height = _plot_legend_height(fonts=fonts, rows=legend_rows)
    legend_area_height = legend_height + LIVE_CHART_STYLE.legend_margin if legend_rows else 0
    legend_gap = LIVE_CHART_STYLE.plot_gap if legend_rows else 0
    plot_y = summary_y + summary_surface.get_height() + LIVE_CHART_STYLE.plot_gap
    if legend_rows:
        legend_rect = pygame.Rect(
            inner_x + LIVE_CHART_STYLE.plot_axis_width,
            plot_y,
            plot_width,
            legend_area_height,
        )
        _draw_plot_legend(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            rect=legend_rect,
            rows=legend_rows,
        )
        plot_y += legend_area_height + legend_gap
    visible_plot_height = max(60, plot_height - legend_area_height - legend_gap)
    plot_rect = pygame.Rect(
        inner_x + LIVE_CHART_STYLE.plot_axis_width,
        plot_y,
        plot_width,
        visible_plot_height,
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
        fonts=fonts,
        rect=plot_rect,
        x_values=x_values,
        series=series,
        fixed_range=fixed_range,
        zero_line=zero_line,
        reference_lines=reference_lines,
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
    fonts: ViewerFonts,
    rect: PygameRect,
    x_values: tuple[int, ...],
    series: tuple[_PlotSeries, ...],
    fixed_range: tuple[float, float] | None,
    zero_line: bool,
    reference_lines: tuple[_PlotReferenceLine, ...],
) -> None:
    valid_series = tuple(line for line in series if len(line.y_values) == len(x_values))
    if not valid_series:
        return
    x_start = x_values[0]
    x_end = max(x_values[-1], x_start + 1)
    if fixed_range is None:
        y_min, y_max = _plot_value_range(
            series=valid_series,
            reference_lines=reference_lines,
        )
        if y_min == y_max:
            span = max(1.0, abs(y_min) * 0.1)
            y_min -= span
            y_max += span
    else:
        y_min, y_max = fixed_range
    span = max(1e-6, y_max - y_min)
    _draw_y_axis_labels(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        rect=rect,
        y_min=y_min,
        y_max=y_max,
        zero_line=zero_line,
    )
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
    for line in reference_lines:
        reference_y = (
            rect.y
            + rect.height
            - 1
            - int(round((line.value - y_min) / span * max(1, rect.height - 1)))
        )
        _draw_reference_line(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            rect=rect,
            y=reference_y,
            line=line,
        )
    for line in valid_series:
        points = _plot_points(
            x_values=x_values,
            y_values=line.y_values,
            rect=rect,
            x_start=x_start,
            x_end=x_end,
            y_min=y_min,
            span=span,
        )
        if len(points) == 1:
            pygame.draw.circle(screen, line.color, points[0], 3)
        else:
            pygame.draw.lines(screen, line.color, False, points, LIVE_CHART_STYLE.line_width)


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


def _draw_reference_line(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    rect: PygameRect,
    y: int,
    line: _PlotReferenceLine,
) -> None:
    _draw_horizontal_dashed_line(
        pygame=pygame,
        screen=screen,
        y=y,
        x_start=rect.x,
        x_end=rect.x + rect.width,
        color=line.color,
    )
    pygame.draw.line(
        screen,
        line.color,
        (rect.x - LIVE_CHART_STYLE.reference_tick_length, y),
        (rect.x, y),
        width=2,
    )
    if line.label is None:
        return
    label_width = LIVE_CHART_STYLE.plot_axis_width - LIVE_CHART_STYLE.reference_tick_length - 3
    label_surface = fonts.small.render(
        _fit_text(fonts.small, line.label, label_width),
        True,
        line.color,
    )
    label_x = rect.x - LIVE_CHART_STYLE.plot_axis_width + 2
    label_y = y - label_surface.get_height() // 2
    label_y = max(rect.y, min(rect.bottom - label_surface.get_height(), label_y))
    padding = LIVE_CHART_STYLE.reference_label_padding
    label_backdrop = pygame.Rect(
        label_x - padding,
        label_y - padding,
        label_surface.get_width() + (2 * padding),
        label_surface.get_height() + (2 * padding),
    )
    pygame.draw.rect(screen, LIVE_CHART_STYLE.block_fill, label_backdrop, border_radius=3)
    screen.blit(label_surface, (label_x, label_y))


def _draw_horizontal_dashed_line(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    y: int,
    x_start: int,
    x_end: int,
    color: tuple[int, int, int],
) -> None:
    x = x_start
    while x < x_end:
        dash_end = min(x + LIVE_CHART_STYLE.reference_line_dash, x_end)
        pygame.draw.line(screen, color, (x, y), (dash_end, y), width=1)
        x = dash_end + LIVE_CHART_STYLE.reference_line_gap


def _draw_plot_legend(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    rect: PygameRect,
    rows: tuple[tuple[_PlotLegendItem, ...], ...],
) -> None:
    if not rows:
        return

    label_height = fonts.small.render("Hg", True, PALETTE.text_primary).get_height()
    max_legend_width = max(1, rect.width - (2 * LIVE_CHART_STYLE.legend_margin))
    content_width = max(
        sum(item.width for item in row) + LIVE_CHART_STYLE.legend_item_gap * max(0, len(row) - 1)
        for row in rows
    )
    legend_width = min(
        max_legend_width,
        content_width + (2 * LIVE_CHART_STYLE.legend_padding),
    )
    legend_height = (
        (2 * LIVE_CHART_STYLE.legend_padding)
        + len(rows) * label_height
        + max(0, len(rows) - 1) * LIVE_CHART_STYLE.legend_row_gap
    )
    legend_rect = pygame.Rect(
        rect.right - LIVE_CHART_STYLE.legend_margin - legend_width,
        rect.y + LIVE_CHART_STYLE.legend_margin,
        legend_width,
        legend_height,
    )
    pygame.draw.rect(screen, LIVE_CHART_STYLE.block_fill, legend_rect, border_radius=5)
    pygame.draw.rect(screen, PALETTE.panel_border, legend_rect, width=1, border_radius=5)

    row_y = legend_rect.y + LIVE_CHART_STYLE.legend_padding
    for row in rows:
        item_x = legend_rect.x + LIVE_CHART_STYLE.legend_padding
        for item in row:
            line_y = row_y + label_height // 2
            pygame.draw.line(
                screen,
                item.color,
                (item_x, line_y),
                (item_x + LIVE_CHART_STYLE.legend_swatch_width, line_y),
                width=LIVE_CHART_STYLE.line_width,
            )
            label_surface = fonts.small.render(item.label, True, item.color)
            label_x = (
                item_x + LIVE_CHART_STYLE.legend_swatch_width + LIVE_CHART_STYLE.legend_swatch_gap
            )
            screen.blit(label_surface, (label_x, row_y))
            item_x += item.width + LIVE_CHART_STYLE.legend_item_gap
        row_y += label_height + LIVE_CHART_STYLE.legend_row_gap


def _plot_legend_height(
    *,
    fonts: ViewerFonts,
    rows: tuple[tuple[_PlotLegendItem, ...], ...],
) -> int:
    if not rows:
        return 0
    label_height = fonts.small.render("Hg", True, PALETTE.text_primary).get_height()
    return (
        (2 * LIVE_CHART_STYLE.legend_padding)
        + len(rows) * label_height
        + max(0, len(rows) - 1) * LIVE_CHART_STYLE.legend_row_gap
    )


def _plot_legend_rows(
    *,
    fonts: ViewerFonts,
    rect_width: int,
    series: tuple[_PlotSeries, ...],
) -> tuple[tuple[_PlotLegendItem, ...], ...]:
    content_width = max(
        1,
        rect_width - (2 * LIVE_CHART_STYLE.legend_margin) - (2 * LIVE_CHART_STYLE.legend_padding),
    )
    rows: list[list[_PlotLegendItem]] = [[]]
    current_width = 0
    for line in series:
        if line.label is None:
            continue
        label_width = fonts.small.render(line.label, True, line.color).get_width()
        item = _PlotLegendItem(
            label=line.label,
            color=line.color,
            width=(
                LIVE_CHART_STYLE.legend_swatch_width
                + LIVE_CHART_STYLE.legend_swatch_gap
                + label_width
            ),
        )
        next_width = (
            item.width
            if current_width == 0
            else current_width + LIVE_CHART_STYLE.legend_item_gap + item.width
        )
        if rows[-1] and next_width > content_width:
            rows.append([])
            current_width = 0
            next_width = item.width
        rows[-1].append(item)
        current_width = next_width
    return tuple(tuple(row) for row in rows if row)


def _draw_y_axis_labels(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    rect: PygameRect,
    y_min: float,
    y_max: float,
    zero_line: bool,
) -> None:
    span = max(1e-6, y_max - y_min)
    values = (
        (y_max, 0.0, y_min)
        if zero_line and y_min < 0.0 < y_max
        else (y_max, (y_min + y_max) / 2.0, y_min)
    )
    label_x = rect.x - LIVE_CHART_STYLE.plot_axis_width + 2
    label_width = LIVE_CHART_STYLE.plot_axis_width - 5
    occupied: list[tuple[int, int]] = []
    for value in values:
        label = _format_axis_value(value)
        surface = fonts.small.render(
            _fit_text(fonts.small, label, label_width), True, PALETTE.text_muted
        )
        label_y = (
            rect.y
            + rect.height
            - 1
            - int(round((value - y_min) / span * max(1, rect.height - 1)))
            - surface.get_height() // 2
        )
        label_y = max(rect.y, min(rect.bottom - surface.get_height(), label_y))
        label_bottom = label_y + surface.get_height()
        if any(
            label_y <= bottom + LIVE_CHART_STYLE.axis_label_gap
            and label_bottom >= top - LIVE_CHART_STYLE.axis_label_gap
            for top, bottom in occupied
        ):
            continue
        occupied.append((label_y, label_bottom))
        screen.blit(surface, (label_x, label_y))
        tick_y = label_y + surface.get_height() // 2
        pygame.draw.line(
            screen,
            LIVE_CHART_STYLE.grid_color,
            (rect.x - 4, tick_y),
            (rect.x, tick_y),
            width=1,
        )


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
