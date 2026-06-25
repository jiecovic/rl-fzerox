# src/rl_fzerox/ui/watch/view/panels/visuals/cnn_layout.py
"""Layout planning for CNN activation images in the Watch side panel.

The renderer asks this module to choose pages, columns, scaled image sizes, and
label positions. Keeping this planning separate from drawing makes the packing
rules easier to test without changing pygame blit order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.ui.watch.runtime.policy.cnn import CnnActivationLayer
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import (
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


def _rgb_surface(pygame: PygameModule, frame: RgbFrame) -> PygameSurface:
    rgb_frame = np.ascontiguousarray(frame)
    height, width, channels = rgb_frame.shape
    if channels != 3:
        raise ValueError(f"Expected an RGB activation grid, got shape {frame.shape!r}")
    return pygame.image.frombuffer(rgb_frame.tobytes(), (width, height), "RGB")


def _scaled_size(size: tuple[int, int], *, max_width: int) -> tuple[int, int]:
    width, height = size
    if width <= 0 or height <= 0:
        return (1, 1)
    if width == max_width:
        return size
    scale = max_width / float(width)
    return (max_width, max(1, round(height * scale)))
