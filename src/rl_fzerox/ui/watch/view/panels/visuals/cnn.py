# src/rl_fzerox/ui/watch/view/panels/visuals/cnn.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.ui.watch.runtime.cnn import (
    CnnActivationChannelStats,
    CnnActivationLayer,
    CnnActivationSnapshot,
)
from rl_fzerox.ui.watch.view.panels.rendering.tab_bar import _draw_text_tabs
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import (
    MouseRect,
    PygameModule,
    PygameSurface,
    ViewerFonts,
)


@dataclass(frozen=True, slots=True)
class _CnnLayerLayoutConfig:
    multi_column_layer_threshold: int = 4
    multi_column_min_width: int = 380
    column_gap: int = 10
    layers_per_page: int = 2


_LAYER_LAYOUT = _CnnLayerLayoutConfig()


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


@dataclass(slots=True)
class _PlannedLayerDraw:
    layer: CnnActivationLayer
    label_surface: PygameSurface
    surface: PygameSurface
    target_size: tuple[int, int]
    column: int = 0
    x_offset: int = 0
    y_offset: int = 0


def _plan_layer_layout(
    *,
    pygame: PygameModule,
    fonts: ViewerFonts,
    width: int,
    available_height: int,
    layers: tuple[CnnActivationLayer, ...],
) -> tuple[_PlannedLayerDraw, ...]:
    columns = _layer_column_count(width=width, layer_count=len(layers))
    column_width = _layer_column_width(width=width, columns=columns)
    planned: list[_PlannedLayerDraw] = []
    for layer in layers:
        label_surface = fonts.small.render(_layer_label(layer), True, PALETTE.text_muted)
        surface = _rgb_surface(pygame, layer.image)
        target_size = _scaled_size(surface.get_size(), max_width=column_width)
        planned.append(
            _PlannedLayerDraw(
                layer=layer,
                label_surface=label_surface,
                surface=surface,
                target_size=target_size,
            )
        )

    if not planned:
        return tuple(planned)

    _place_layers(planned, columns=columns, column_width=column_width)
    scale = _height_fit_scale(planned, columns=columns, available_height=available_height)
    if scale >= 1.0:
        return tuple(planned)

    for item in planned:
        item.target_size = (
            max(1, round(item.target_size[0] * scale)),
            max(1, round(item.target_size[1] * scale)),
        )
    _place_layers(planned, columns=columns, column_width=column_width)
    return tuple(planned)


def _layer_label(layer: CnnActivationLayer) -> str:
    if layer.rendered_channel_count < layer.channel_count:
        channels = f"{layer.rendered_channel_count}/{layer.channel_count}"
    else:
        channels = str(layer.channel_count)
    return f"{layer.name}  {channels} x {layer.spatial_shape[0]} x {layer.spatial_shape[1]}"


def _layer_pages(
    layers: tuple[CnnActivationLayer, ...],
) -> tuple[tuple[CnnActivationLayer, ...], ...]:
    page_size = max(1, _LAYER_LAYOUT.layers_per_page)
    return tuple(layers[index : index + page_size] for index in range(0, len(layers), page_size))


def _layer_page_label(layers: tuple[CnnActivationLayer, ...]) -> str:
    if not layers:
        return "-"
    if len(layers) == 1:
        return layers[0].name
    return f"{layers[0].name}+{layers[-1].name}"


def _layer_column_count(*, width: int, layer_count: int) -> int:
    if layer_count < _LAYER_LAYOUT.multi_column_layer_threshold:
        return 1
    if width < _LAYER_LAYOUT.multi_column_min_width:
        return 1
    return 2


def _layer_column_width(*, width: int, columns: int) -> int:
    if columns <= 1:
        return max(1, width)
    return max(1, (width - ((columns - 1) * _LAYER_LAYOUT.column_gap)) // columns)


def _place_layers(
    planned: list[_PlannedLayerDraw],
    *,
    columns: int,
    column_width: int,
) -> None:
    column_heights = [0 for _ in range(columns)]
    for item in planned:
        column = min(range(columns), key=column_heights.__getitem__)
        item.column = column
        item.x_offset = column * (column_width + _LAYER_LAYOUT.column_gap)
        item.y_offset = column_heights[column]
        column_heights[column] += (
            item.label_surface.get_height()
            + LAYOUT.section_title_gap
            + item.target_size[1]
            + LAYOUT.section_gap
        )


def _height_fit_scale(
    planned: list[_PlannedLayerDraw],
    *,
    columns: int,
    available_height: int,
) -> float:
    column_fixed_heights = [0 for _ in range(columns)]
    column_image_heights = [0 for _ in range(columns)]
    for item in planned:
        column_fixed_heights[item.column] += (
            item.label_surface.get_height() + LAYOUT.section_title_gap + LAYOUT.section_gap
        )
        column_image_heights[item.column] += item.target_size[1]

    scale = 1.0
    for fixed_height, image_height in zip(
        column_fixed_heights,
        column_image_heights,
        strict=True,
    ):
        if image_height <= 0:
            continue
        remaining = max(1, available_height - fixed_height)
        scale = min(scale, remaining / float(image_height))
    return scale


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


def _rgb_surface(pygame: PygameModule, frame: RgbFrame) -> PygameSurface:
    rgb_frame = np.ascontiguousarray(frame)
    height, width, channels = rgb_frame.shape
    if channels != 3:
        raise ValueError(f"Expected an RGB activation grid, got shape {frame.shape!r}")
    return pygame.image.frombuffer(rgb_frame.tobytes(), (width, height), "RGB")


def _draw_grid_overlay(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    x: int,
    y: int,
    size: tuple[int, int],
    grid_shape: tuple[int, int],
    used_tiles: int,
) -> None:
    rows, columns = grid_shape
    width, height = size
    _draw_unused_tile_hatches(
        pygame=pygame,
        screen=screen,
        x=x,
        y=y,
        rects=_unused_tile_rects(size=size, grid_shape=grid_shape, used_tiles=used_tiles),
    )
    for line_x in _separator_positions(width, columns):
        absolute_x = x + line_x
        pygame.draw.line(
            screen,
            PALETTE.cnn_grid_separator,
            (absolute_x, y),
            (absolute_x, y + height - 1),
            width=1,
        )
    for line_y in _separator_positions(height, rows):
        absolute_y = y + line_y
        pygame.draw.line(
            screen,
            PALETTE.cnn_grid_separator,
            (x, absolute_y),
            (x + width - 1, absolute_y),
            width=1,
        )
    pygame.draw.rect(
        screen,
        PALETTE.cnn_grid_separator,
        (x, y, width, height),
        width=1,
    )


def _draw_unused_tile_hatches(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    x: int,
    y: int,
    rects: tuple[tuple[int, int, int, int], ...],
) -> None:
    previous_clip = screen.get_clip()
    try:
        for rel_x, rel_y, width, height in rects:
            tile_rect = pygame.Rect(x + rel_x, y + rel_y, width, height)
            screen.set_clip(tile_rect)
            for offset in range(-height, width, 6):
                pygame.draw.line(
                    screen,
                    PALETTE.cnn_grid_unused_hatch,
                    (tile_rect.left + offset, tile_rect.bottom - 1),
                    (tile_rect.left + offset + height - 1, tile_rect.top),
                    width=1,
                )
    finally:
        screen.set_clip(previous_clip)


def _unused_tile_rects(
    *,
    size: tuple[int, int],
    grid_shape: tuple[int, int],
    used_tiles: int,
) -> tuple[tuple[int, int, int, int], ...]:
    rows, columns = grid_shape
    width, height = size
    if rows <= 0 or columns <= 0:
        return ()
    first_unused = max(0, min(used_tiles, rows * columns))
    rects: list[tuple[int, int, int, int]] = []
    for index in range(first_unused, rows * columns):
        row, column = divmod(index, columns)
        left, right = _partition_bounds(width, columns, column)
        top, bottom = _partition_bounds(height, rows, row)
        if right > left and bottom > top:
            rects.append((left, top, right - left, bottom - top))
    return tuple(rects)


def _separator_positions(length: int, parts: int) -> tuple[int, ...]:
    if length <= 1 or parts <= 1:
        return ()
    return tuple(
        position
        for index in range(1, parts)
        if 0 < (position := round(index * length / parts)) < length
    )


def _partition_bounds(length: int, parts: int, index: int) -> tuple[int, int]:
    if length <= 0 or parts <= 0:
        return (0, 0)
    return (
        round(index * length / parts),
        round((index + 1) * length / parts),
    )


def _scaled_size(size: tuple[int, int], *, max_width: int) -> tuple[int, int]:
    width, height = size
    if width <= 0 or height <= 0:
        return (1, 1)
    if width == max_width:
        return size
    scale = max_width / float(width)
    return (max_width, max(1, round(height * scale)))
