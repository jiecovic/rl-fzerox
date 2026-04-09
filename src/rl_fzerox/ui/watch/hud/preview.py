# src/rl_fzerox/ui/watch/hud/preview.py
from __future__ import annotations

import numpy as np

from rl_fzerox.ui.watch.hud.format import (
    _format_observation_summary,
    _preview_frame_shape,
)
from rl_fzerox.ui.watch.layout import FONT_SIZES, LAYOUT, PALETTE, ViewerFonts


def _window_size(
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
) -> tuple[int, int]:
    return (
        game_display_size[0] + LAYOUT.preview_gap + LAYOUT.panel_width,
        max(game_display_size[1], LAYOUT.panel_min_height),
    )


def _preview_frame(observation: np.ndarray) -> np.ndarray:
    if observation.ndim != 3:
        raise ValueError(f"Expected an HxWxC observation, got {observation.shape!r}")

    channels = observation.shape[2]
    if channels == 3:
        return np.ascontiguousarray(observation)
    if channels == 1:
        return np.repeat(observation, 3, axis=2)
    if channels % 3 == 0:
        return np.ascontiguousarray(observation[:, :, -3:])

    latest_channel = observation[:, :, -1:]
    return np.repeat(latest_channel, 3, axis=2)


def _observation_preview_size(observation_shape: tuple[int, ...]) -> tuple[int, int]:
    preview_shape = _preview_frame_shape(observation_shape)
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
) -> int:
    preview_height = _observation_preview_size(observation_shape)[1]
    title_height = fonts.section.render("Policy Obs", True, PALETTE.text_primary).get_height()
    subtitle_height = fonts.small.render(
        _format_observation_summary(observation_shape),
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
