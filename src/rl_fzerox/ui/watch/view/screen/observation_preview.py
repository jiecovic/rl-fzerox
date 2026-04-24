# src/rl_fzerox/ui/watch/view/screen/observation_preview.py
from __future__ import annotations

import numpy as np

from fzerox_emulator.arrays import ObservationFrame, RgbFrame
from rl_fzerox.ui.watch.view.panels.format import (
    _format_observation_summary,
    _observation_minimap_layer,
    _observation_preview_grid,
    _observation_stack_mode,
    _observation_stack_size,
    _preview_frame_shape,
)
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import FONT_SIZES, PALETTE
from rl_fzerox.ui.watch.view.screen.types import ViewerFonts


def _window_size(
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
) -> tuple[int, int]:
    return (
        game_display_size[0] + LAYOUT.preview_gap + LAYOUT.panel_width,
        max(game_display_size[1], LAYOUT.panel_min_height),
    )


def _preview_frame(
    observation: ObservationFrame,
    info: dict[str, object] | None = None,
) -> RgbFrame:
    if observation.ndim != 3:
        raise ValueError(f"Expected an HxWxC observation, got {observation.shape!r}")

    frames = _preview_frames(observation, info=info)
    if len(frames) == 1:
        return np.ascontiguousarray(frames[0])
    return _preview_frame_grid(frames)


def _preview_frames(
    observation: ObservationFrame,
    *,
    info: dict[str, object] | None,
) -> tuple[RgbFrame, ...]:
    stack_size = _observation_stack_size(observation.shape, info=info)
    stack_mode = _observation_stack_mode(info)
    minimap_layer = _observation_minimap_layer(info)
    base_observation = observation[:, :, :-1] if minimap_layer else observation

    if stack_mode == "gray":
        frames = _grayscale_preview_frames(base_observation, stack_size=stack_size)
        return (
            (*frames, _grayscale_preview_frame(observation[:, :, -1:])) if minimap_layer else frames
        )
    if stack_mode == "luma_chroma":
        frames = _luma_chroma_preview_frames(base_observation, stack_size=stack_size)
        return (
            (*frames, _grayscale_preview_frame(observation[:, :, -1:])) if minimap_layer else frames
        )

    base_channels = base_observation.shape[2]
    if base_channels == 3:
        frames = (np.ascontiguousarray(base_observation),)
    elif base_channels == 1:
        frames = (_grayscale_preview_frame(base_observation),)
    elif base_channels >= 3 and base_channels % 3 == 0:
        frames = [
            np.ascontiguousarray(base_observation[:, :, start : start + 3])
            for start in range(0, base_channels, 3)
        ]
        frames = tuple(frames)
    else:
        latest_channel = base_observation[:, :, -1:]
        frames = (_grayscale_preview_frame(latest_channel),)

    return (*frames, _grayscale_preview_frame(observation[:, :, -1:])) if minimap_layer else frames


def _grayscale_preview_frames(
    observation: ObservationFrame,
    *,
    stack_size: int,
) -> tuple[RgbFrame, ...]:
    frame_count = max(1, min(stack_size, observation.shape[2]))
    return tuple(
        _grayscale_preview_frame(observation[:, :, index : index + 1])
        for index in range(frame_count)
    )


def _luma_chroma_preview_frames(
    observation: ObservationFrame,
    *,
    stack_size: int,
) -> tuple[RgbFrame, ...]:
    frame_count = max(1, min(stack_size, observation.shape[2] // 2))
    return tuple(
        _luma_chroma_preview_frame(observation[:, :, start : start + 2])
        for start in range(0, frame_count * 2, 2)
    )


def _luma_chroma_preview_frame(channels: ObservationFrame) -> RgbFrame:
    luma = channels[:, :, 0].astype(np.int16)
    chroma = channels[:, :, 1].astype(np.int16) - 128
    yellow = np.clip(chroma, 0, 127)
    purple = np.clip(-chroma, 0, 127)
    red = np.clip(luma + yellow + (purple // 2), 0, 255)
    green = np.clip(luma + yellow, 0, 255)
    blue = np.clip(luma + purple, 0, 255)
    return np.ascontiguousarray(np.stack((red, green, blue), axis=2).astype(np.uint8))


def _grayscale_preview_frame(channel: ObservationFrame) -> RgbFrame:
    return np.repeat(channel, 3, axis=2)


def _preview_frame_grid(frames: tuple[RgbFrame, ...]) -> RgbFrame:
    tile_height, tile_width, channels = frames[0].shape
    columns, rows = _observation_preview_grid(len(frames))
    grid = np.zeros((rows * tile_height, columns * tile_width, channels), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        y = row * tile_height
        x = column * tile_width
        grid[y : y + tile_height, x : x + tile_width, :] = frame
    return np.ascontiguousarray(grid)


def _observation_preview_size(
    observation_shape: tuple[int, ...],
    info: dict[str, object] | None = None,
) -> tuple[int, int]:
    preview_shape = _preview_frame_shape(observation_shape, info=info)
    return (
        preview_shape[1] * LAYOUT.preview_scale,
        preview_shape[0] * LAYOUT.preview_scale,
    )


def _preview_panel_size(observation_shape: tuple[int, ...]) -> tuple[int, int]:
    preview_width, preview_height = _observation_preview_size(observation_shape)
    title_height = FONT_SIZES.section + LAYOUT.preview_title_gap + FONT_SIZES.small
    panel_height = (
        (2 * LAYOUT.preview_padding) + title_height + LAYOUT.section_rule_gap + preview_height
    )
    panel_width = preview_width + (2 * LAYOUT.preview_padding)
    return panel_width, panel_height


def _preview_block_height(
    observation_shape: tuple[int, ...],
    fonts: ViewerFonts,
    info: dict[str, object] | None = None,
) -> int:
    preview_height = _observation_preview_size(observation_shape, info=info)[1]
    title_height = fonts.section.render("Policy Obs", True, PALETTE.text_primary).get_height()
    subtitle_height = fonts.small.render(
        _format_observation_summary(observation_shape, info=info),
        True,
        PALETTE.text_muted,
    ).get_height()
    return (
        title_height
        + LAYOUT.preview_title_gap
        + subtitle_height
        + LAYOUT.section_rule_gap
        + preview_height
    )
