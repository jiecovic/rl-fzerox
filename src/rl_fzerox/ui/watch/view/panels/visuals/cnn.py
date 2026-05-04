# src/rl_fzerox/ui/watch/view/panels/visuals/cnn.py
from __future__ import annotations

import numpy as np

from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.ui.watch.runtime.cnn import CnnActivationLayer, CnnActivationSnapshot
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PygameModule, PygameSurface, ViewerFonts


def _draw_cnn_tab(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    activations: CnnActivationSnapshot | None,
) -> int:
    y = _draw_heading(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=x,
        y=y,
        width=width,
        activations=activations,
    )
    if activations is None:
        return _draw_note(
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            text="Open this tab while a policy is running to capture CNN activations.",
        )
    if activations.error is not None:
        return _draw_note(
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            text=f"Activation capture failed: {activations.error}",
        )
    if not activations.layers:
        return _draw_note(
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            text="No F-Zero X CNN image extractor found for this policy.",
        )

    layer_layout = _plan_layer_layout(
        pygame=pygame,
        fonts=fonts,
        width=width,
        available_height=max(1, screen.get_height() - y - LAYOUT.panel_padding),
        layers=activations.layers,
    )
    current_y = y
    for planned in layer_layout:
        screen.blit(planned.label_surface, (x, current_y))
        current_y += planned.label_surface.get_height() + LAYOUT.section_title_gap

        surface = planned.surface
        if planned.target_size != surface.get_size():
            surface = pygame.transform.scale(surface, planned.target_size)
        screen.blit(surface, (x, current_y))
        _draw_grid_overlay(
            pygame=pygame,
            screen=screen,
            x=x,
            y=current_y,
            size=planned.target_size,
            grid_shape=planned.layer.grid_shape,
            used_tiles=planned.layer.rendered_channel_count,
        )
        current_y += planned.target_size[1] + LAYOUT.section_gap
    return current_y


class _PlannedLayerDraw:
    def __init__(
        self,
        *,
        layer: CnnActivationLayer,
        label_surface: PygameSurface,
        surface: PygameSurface,
        target_size: tuple[int, int],
    ) -> None:
        self.layer = layer
        self.label_surface = label_surface
        self.surface = surface
        self.target_size = target_size


def _plan_layer_layout(
    *,
    pygame: PygameModule,
    fonts: ViewerFonts,
    width: int,
    available_height: int,
    layers: tuple[CnnActivationLayer, ...],
) -> tuple[_PlannedLayerDraw, ...]:
    planned: list[_PlannedLayerDraw] = []
    fixed_height = 0
    image_height = 0
    for layer in layers:
        label = (
            f"{layer.name}  "
            f"{layer.channel_count} x {layer.spatial_shape[0]} x {layer.spatial_shape[1]}"
        )
        label_surface = fonts.small.render(label, True, PALETTE.text_muted)
        surface = _rgb_surface(pygame, layer.image)
        target_size = _scaled_size(surface.get_size(), max_width=width)
        planned.append(
            _PlannedLayerDraw(
                layer=layer,
                label_surface=label_surface,
                surface=surface,
                target_size=target_size,
            )
        )
        fixed_height += label_surface.get_height() + LAYOUT.section_title_gap + LAYOUT.section_gap
        image_height += target_size[1]

    if not planned or image_height <= 0:
        return tuple(planned)

    remaining_image_height = max(1, available_height - fixed_height)
    if image_height <= remaining_image_height:
        return tuple(planned)

    scale = remaining_image_height / float(image_height)
    for item in planned:
        item.target_size = (
            max(1, round(item.target_size[0] * scale)),
            max(1, round(item.target_size[1] * scale)),
        )
    return tuple(planned)


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
        return "C toggles norm"
    if activations.normalization == "layer_percentile":
        return "norm: layer p99"
    return "norm: channel"


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
