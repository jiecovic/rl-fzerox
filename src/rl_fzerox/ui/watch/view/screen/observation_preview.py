# src/rl_fzerox/ui/watch/view/screen/observation_preview.py
from __future__ import annotations

import numpy as np

from fzerox_emulator.arrays import ObservationFrame, RgbFrame
from rl_fzerox.ui.watch.view.panels.format import (
    _format_observation_summary,
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
    channels = observation.shape[2]
    stack_size = _observation_stack_size(observation.shape, info=info)
    stack_mode = _observation_stack_mode(info)

    if stack_mode == "rgb_gray":
        return _rgb_gray_preview_frames(observation, stack_size=stack_size)

    if channels == 3:
        return (np.ascontiguousarray(observation),)
    if channels == 1:
        return (np.repeat(observation, 3, axis=2),)
    if channels >= 3 and channels % 3 == 0:
        frames = [
            np.ascontiguousarray(observation[:, :, start : start + 3])
            for start in range(0, channels, 3)
        ]
        return tuple(frames)

    latest_channel = observation[:, :, -1:]
    return (np.repeat(latest_channel, 3, axis=2),)


def _rgb_gray_preview_frames(
    observation: ObservationFrame,
    *,
    stack_size: int,
) -> tuple[RgbFrame, ...]:
    channels = observation.shape[2]
    if stack_size <= 1 or channels <= 3:
        return (np.ascontiguousarray(observation[:, :, -3:]),)

    history_count = max(0, min(stack_size - 1, channels - 3))
    frames = [
        np.repeat(observation[:, :, index : index + 1], 3, axis=2) for index in range(history_count)
    ]
    frames.append(np.ascontiguousarray(observation[:, :, -3:]))
    return tuple(frames)


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
