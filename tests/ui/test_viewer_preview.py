# tests/ui/test_viewer_preview.py
import numpy as np

from rl_fzerox.ui.watch.view.panels.format import _format_observation_summary
from rl_fzerox.ui.watch.view.panels.model import _observation_preview_size, _preview_frame


def test_preview_frame_shows_stacked_rgb_observations_as_grid() -> None:
    first = np.zeros((2, 3, 3), dtype=np.uint8)
    second = np.full((2, 3, 3), 255, dtype=np.uint8)
    stacked = np.concatenate((first, second), axis=2)

    preview = _preview_frame(stacked)

    assert preview.shape == (2, 6, 3)
    assert np.array_equal(preview[:, :3, :], second)
    assert np.array_equal(preview[:, 3:6, :], first)


def test_preview_frame_shows_grayscale_observations_as_grid() -> None:
    stacked = np.zeros((2, 3, 4), dtype=np.uint8)
    stacked[:, :, 0] = 32
    stacked[:, :, 1] = 96
    stacked[:, :, 2] = 160
    stacked[:, :, 3] = 224
    info = {"observation_stack": 4, "observation_stack_mode": "gray"}

    preview = _preview_frame(stacked, info=info)

    assert preview.shape == (2, 12, 3)
    assert np.array_equal(preview[:, :3, :], np.repeat(stacked[:, :, 3:4], 3, axis=2))
    assert np.array_equal(preview[:, 3:6, :], np.repeat(stacked[:, :, 2:3], 3, axis=2))
    assert np.array_equal(preview[:, 6:9, :], np.repeat(stacked[:, :, 1:2], 3, axis=2))
    assert np.array_equal(preview[:, 9:12, :], np.repeat(stacked[:, :, 0:1], 3, axis=2))
    assert _observation_preview_size(stacked.shape, info=info) == (12, 2)
    assert (
        _format_observation_summary(
            stacked.shape,
            info=info,
        )
        == "3x2 gray x4 stack"
    )


def test_preview_frame_shows_luma_chroma_observations_as_grid() -> None:
    stacked = np.zeros((2, 3, 8), dtype=np.uint8)
    stacked[:, :, 0::2] = 96
    stacked[:, :, 1::2] = 192
    info = {"observation_stack": 4, "observation_stack_mode": "luma_chroma"}

    preview = _preview_frame(stacked, info=info)

    assert preview.shape == (2, 12, 3)
    assert _observation_preview_size(stacked.shape, info=info) == (12, 2)
    assert (
        _format_observation_summary(
            stacked.shape,
            info=info,
        )
        == "3x2 y+c x4 stack"
    )
