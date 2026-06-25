# src/rl_fzerox/ui/watch/view/panels/visuals/cnn.py
"""Render CNN activation pages inside the Watch side panel.

The runtime captures activation snapshots; this module plans layer/stat layouts
and draws them with pygame, including layer tabs and unused-tile overlays.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.runtime.policy.cnn import (
    CnnActivationChannelStats,
    CnnActivationLayer,
    CnnActivationSnapshot,
)
from rl_fzerox.ui.watch.view.panels.rendering.tab_bar import _draw_text_tabs
from rl_fzerox.ui.watch.view.panels.visuals.cnn_grid import _draw_grid_overlay
from rl_fzerox.ui.watch.view.panels.visuals.cnn_layout import (
    _layer_label,
    _layer_page_label,
    _layer_pages,
    _plan_layer_layout,
)
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import (
    MouseRect,
    PygameModule,
    PygameSurface,
    ViewerFonts,
)


@dataclass(frozen=True, slots=True)
class _CnnStatsLayoutConfig:
    max_columns: int = 16
    cell_size: int = 16
    cell_gap: int = 3
    min_cell_size: int = 10
    summary_gap: int = 5
    detail_gap: int = 3
    grid_top_gap: int = 7
    weakest_channel_count: int = 6


_STATS_LAYOUT = _CnnStatsLayoutConfig()


def _draw_cnn_tab(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    activations: CnnActivationSnapshot | None,
    selected_layer_page_index: int = 0,
) -> tuple[int, tuple[MouseRect | None, ...]]:
    y = _draw_heading(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=x,
        y=y,
        width=width,
        activations=activations,
    )
    y = _draw_mode_hint(screen=screen, fonts=fonts, x=x, y=y, activations=activations)
    if activations is None:
        y = _draw_note(
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            text="Open this tab while a policy is running to capture CNN activations.",
        )
        return y, ()
    if activations.error is not None:
        y = _draw_note(
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            text=f"Activation capture failed: {activations.error}",
        )
        return y, ()
    if not activations.layers:
        y = _draw_note(
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            text="No F-Zero X CNN image extractor found for this policy.",
        )
        return y, ()

    layer_pages = _layer_pages(activations.layers)
    selected_page_index = selected_layer_page_index % len(layer_pages) if layer_pages else 0
    layer_tab_rects: tuple[MouseRect | None, ...] = ()
    if len(layer_pages) > 1:
        y, layer_tab_rects = _draw_text_tabs(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            width=width,
            labels=tuple(_layer_page_label(page) for page in layer_pages),
            selected_index=selected_page_index,
            hint_text=f"Layers {selected_page_index + 1}/{len(layer_pages)}",
        )
        y += LAYOUT.title_section_gap

    selected_layers = layer_pages[selected_page_index]
    if activations.normalization == "stats":
        return (
            _draw_stats_layers(
                pygame=pygame,
                screen=screen,
                fonts=fonts,
                x=x,
                y=y,
                width=width,
                layers=selected_layers,
            ),
            layer_tab_rects,
        )

    layer_layout = _plan_layer_layout(
        pygame=pygame,
        fonts=fonts,
        width=width,
        available_height=max(1, screen.get_height() - y - LAYOUT.panel_padding),
        layers=selected_layers,
    )
    current_y = y
    for planned in layer_layout:
        label_x = x + planned.x_offset
        label_y = y + planned.y_offset
        screen.blit(planned.label_surface, (label_x, label_y))
        image_y = label_y + planned.label_surface.get_height() + LAYOUT.section_title_gap

        surface = planned.surface
        if planned.target_size != surface.get_size():
            surface = pygame.transform.scale(surface, planned.target_size)
        screen.blit(surface, (label_x, image_y))
        _draw_grid_overlay(
            pygame=pygame,
            screen=screen,
            x=label_x,
            y=image_y,
            size=planned.target_size,
            grid_shape=planned.layer.grid_shape,
            used_tiles=planned.layer.rendered_channel_count,
        )
        current_y = max(current_y, image_y + planned.target_size[1] + LAYOUT.section_gap)
    return current_y, layer_tab_rects


def _draw_heading(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    activations: CnnActivationSnapshot | None,
) -> int:
    title = fonts.section.render("CNN Activations", True, PALETTE.text_primary)
    screen.blit(title, (x, y))
    mode_label = _normalization_label(activations)
    if mode_label is not None:
        mode_surface = fonts.small.render(mode_label, True, PALETTE.text_muted)
        screen.blit(mode_surface, (x + width - mode_surface.get_width(), y + 2))
    y += title.get_height() + LAYOUT.section_title_gap
    pygame.draw.line(screen, PALETTE.panel_border, (x, y), (x + width, y), width=1)
    return y + LAYOUT.section_rule_gap


def _normalization_label(activations: CnnActivationSnapshot | None) -> str | None:
    if activations is None:
        return "C cycles CNN view"
    if activations.normalization == "layer_percentile":
        return "view: layer strength"
    if activations.normalization == "stats":
        return "view: dead check"
    return "view: channel structure"


def _draw_mode_hint(
    *,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    activations: CnnActivationSnapshot | None,
) -> int:
    hint = _mode_hint(activations)
    if hint is None:
        return y
    surface = fonts.small.render(hint, True, PALETTE.text_muted)
    screen.blit(surface, (x, y))
    return y + surface.get_height() + LAYOUT.section_title_gap


def _mode_hint(activations: CnnActivationSnapshot | None) -> str | None:
    if activations is None:
        return None
    if activations.normalization == "layer_percentile":
        return "Shared 0..p99 scale per layer: brighter tile = stronger channel."
    if activations.normalization == "stats":
        return "Dead-channel check, not a spatial feature-map image."
    return "Each channel is autoscaled independently: use this for spatial structure."


def _draw_stats_layers(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    layers: tuple[CnnActivationLayer, ...],
) -> int:
    current_y = y
    for layer in layers:
        current_y = _draw_stats_layer(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=x,
            y=current_y,
            width=width,
            layer=layer,
        )
        current_y += LAYOUT.section_gap
    return current_y


def _draw_stats_layer(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    layer: CnnActivationLayer,
) -> int:
    label_surface = fonts.small.render(_layer_label(layer), True, PALETTE.text_muted)
    screen.blit(label_surface, (x, y))
    current_y = y + label_surface.get_height() + _STATS_LAYOUT.summary_gap

    summary_surface = fonts.small.render(_stats_summary(layer), True, PALETTE.text_primary)
    screen.blit(summary_surface, (x, current_y))
    current_y += summary_surface.get_height() + _STATS_LAYOUT.detail_gap

    detail_surface = fonts.small.render(_weak_channel_summary(layer), True, PALETTE.text_muted)
    screen.blit(detail_surface, (x, current_y))
    current_y += detail_surface.get_height() + _STATS_LAYOUT.detail_gap

    legend_surface = fonts.small.render(_stats_legend(), True, PALETTE.text_muted)
    screen.blit(legend_surface, (x, current_y))
    current_y += legend_surface.get_height() + _STATS_LAYOUT.grid_top_gap

    if not layer.channel_stats:
        return _draw_note(
            screen=screen,
            fonts=fonts,
            x=x,
            y=current_y,
            text="No channel statistics available for this layer.",
        )
    return _draw_channel_stats_grid(
        pygame=pygame,
        screen=screen,
        x=x,
        y=current_y,
        width=width,
        stats=layer.channel_stats,
    )


def _stats_summary(layer: CnnActivationLayer) -> str:
    stats = layer.channel_stats
    if not stats:
        return "stats unavailable"

    dead_count = sum(1 for stat in stats if stat.dead)
    active_percent = _mean(tuple(stat.active_fraction for stat in stats)) * 100.0
    mean_abs = _mean(tuple(stat.mean_abs for stat in stats))
    max_abs = max((stat.max_abs for stat in stats), default=0.0)
    return (
        f"dead {dead_count}/{len(stats)} · active {active_percent:5.1f}% · "
        f"mean|x| {_format_compact_float(mean_abs)} · max|x| {_format_compact_float(max_abs)}"
    )


def _weak_channel_summary(layer: CnnActivationLayer) -> str:
    stats = layer.channel_stats
    if not stats:
        return "weakest channels: n/a"

    weakest = sorted(stats, key=lambda stat: stat.max_abs)[: _STATS_LAYOUT.weakest_channel_count]
    channels = " ".join(f"{stat.index}:{_format_compact_float(stat.max_abs)}" for stat in weakest)
    return f"weakest max|x| ch: {channels}"


def _stats_legend() -> str:
    return "grid: brighter = active on more spatial cells · yellow border = dead"


def _draw_channel_stats_grid(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    x: int,
    y: int,
    width: int,
    stats: tuple[CnnActivationChannelStats, ...],
) -> int:
    columns = _stats_grid_columns(width=width, channel_count=len(stats))
    cell_size = _stats_cell_size(width=width, columns=columns)
    for index, stat in enumerate(stats):
        row, column = divmod(index, columns)
        cell_x = x + column * (cell_size + _STATS_LAYOUT.cell_gap)
        cell_y = y + row * (cell_size + _STATS_LAYOUT.cell_gap)
        rect = pygame.Rect(cell_x, cell_y, cell_size, cell_size)
        pygame.draw.rect(screen, _stats_cell_color(stat), rect, border_radius=2)
        border_color = PALETTE.text_warning if stat.dead else PALETTE.panel_border
        pygame.draw.rect(screen, border_color, rect, width=1, border_radius=2)

    rows = max(1, (len(stats) + columns - 1) // columns)
    return y + rows * cell_size + (rows - 1) * _STATS_LAYOUT.cell_gap


def _stats_grid_columns(*, width: int, channel_count: int) -> int:
    if channel_count <= 0:
        return 1
    max_columns = min(channel_count, _STATS_LAYOUT.max_columns)
    widest_columns = max(
        1,
        (width + _STATS_LAYOUT.cell_gap) // (_STATS_LAYOUT.min_cell_size + _STATS_LAYOUT.cell_gap),
    )
    return max(1, min(max_columns, widest_columns))


def _stats_cell_size(*, width: int, columns: int) -> int:
    available = width - ((columns - 1) * _STATS_LAYOUT.cell_gap)
    return max(_STATS_LAYOUT.min_cell_size, min(_STATS_LAYOUT.cell_size, available // columns))


def _stats_cell_color(stat: CnnActivationChannelStats) -> Color:
    if stat.dead:
        return (44, 31, 33)
    strength = max(0.0, min(1.0, stat.active_fraction)) ** 0.5
    return _blend_color((27, 34, 42), PALETTE.text_accent, strength)


def _blend_color(start: Color, end: Color, amount: float) -> Color:
    t = max(0.0, min(1.0, amount))
    return (
        round(start[0] + ((end[0] - start[0]) * t)),
        round(start[1] + ((end[1] - start[1]) * t)),
        round(start[2] + ((end[2] - start[2]) * t)),
    )


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _format_compact_float(value: float) -> str:
    absolute = abs(value)
    if absolute == 0.0:
        return "0"
    if absolute < 0.01 or absolute >= 1000.0:
        return f"{value:.1e}"
    if absolute < 1.0:
        return f"{value:.3f}"
    if absolute < 10.0:
        return f"{value:.2f}"
    return f"{value:.1f}"


def _draw_note(
    *,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    text: str,
) -> int:
    surface = fonts.small.render(text, True, PALETTE.text_muted)
    screen.blit(surface, (x, y))
    return y + surface.get_height()
