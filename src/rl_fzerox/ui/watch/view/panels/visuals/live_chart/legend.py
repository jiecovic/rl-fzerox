# src/rl_fzerox/ui/watch/view/panels/visuals/live_chart/legend.py
"""Legend layout and drawing for compact Watch live charts."""

from __future__ import annotations

from rl_fzerox.ui.watch.view.panels.visuals.live_chart.elements import (
    _PlotLegendItem,
    _PlotSeries,
)
from rl_fzerox.ui.watch.view.panels.visuals.live_chart.style import LIVE_CHART_STYLE
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import (
    PygameModule,
    PygameRect,
    PygameSurface,
    ViewerFonts,
)


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
