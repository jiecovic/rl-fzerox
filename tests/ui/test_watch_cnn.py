# tests/ui/test_watch_cnn.py
from __future__ import annotations

import numpy as np

from fzerox_emulator.arrays import Float32Array
from rl_fzerox.core.training.inference import PolicyCnnActivation
from rl_fzerox.ui.watch.runtime.cnn import (
    CnnActivationSampler,
    _activation_grid,
    next_cnn_activation_normalization,
)
from rl_fzerox.ui.watch.view.panels.visuals.cnn import (
    _separator_positions,
    _unused_tile_rects,
)


class _PolicyRunner:
    def __init__(self) -> None:
        self.calls = 0

    def cnn_activations(self, observation: object) -> tuple[PolicyCnnActivation, ...]:
        del observation
        self.calls += 1
        return (
            PolicyCnnActivation(
                name="conv1",
                values=np.ones((4, 2, 3), dtype=np.float32),
            ),
        )


def test_activation_grid_tiles_channels_as_rgb() -> None:
    values: Float32Array = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)

    grid = _activation_grid(values)

    assert grid.dtype == np.uint8
    assert grid.shape == (4, 6, 3)
    assert grid[0, 0].tolist() == [int(grid[0, 0, 0])] * 3


def test_activation_grid_can_use_layer_percentile_normalization() -> None:
    values = np.array(
        [
            [[0.0, 1.0], [1.0, 1.0]],
            [[0.0, 100.0], [100.0, 100.0]],
        ],
        dtype=np.float32,
    )

    channel_grid = _activation_grid(values)
    layer_grid = _activation_grid(values, normalization="layer_percentile")

    assert int(channel_grid[:2, :2, 0].max()) == 255
    assert int(layer_grid[:2, :2, 0].max()) < 10
    assert int(layer_grid[:2, 2:, 0].max()) == 255


def test_cnn_activation_normalization_toggle_cycles_modes() -> None:
    assert next_cnn_activation_normalization("channel") == "layer_percentile"
    assert next_cnn_activation_normalization("layer_percentile") == "channel"


def test_separator_positions_are_drawn_after_scaling() -> None:
    assert _separator_positions(280, 4) == (70, 140, 210)
    assert _separator_positions(281, 4) == (70, 140, 211)
    assert _separator_positions(12, 1) == ()


def test_unused_tiles_are_hatched_after_scaling() -> None:
    assert _unused_tile_rects(size=(280, 140), grid_shape=(2, 3), used_tiles=5) == (
        (187, 70, 93, 70),
    )
    assert _unused_tile_rects(size=(280, 140), grid_shape=(2, 2), used_tiles=4) == ()


def test_cnn_activation_sampler_throttles_policy_captures() -> None:
    policy_runner = _PolicyRunner()
    sampler = CnnActivationSampler(refresh_interval_steps=3)
    observation = np.zeros((2, 3, 3), dtype=np.uint8)

    first = sampler.capture(enabled=True, policy_runner=policy_runner, observation=observation)
    second = sampler.capture(enabled=True, policy_runner=policy_runner, observation=observation)

    assert first is not None
    assert second is first
    assert first.layers[0].grid_shape == (2, 2)
    assert first.layers[0].rendered_channel_count == 4
    assert policy_runner.calls == 1


def test_cnn_activation_sampler_refreshes_when_normalization_changes() -> None:
    policy_runner = _PolicyRunner()
    sampler = CnnActivationSampler(refresh_interval_steps=3)
    observation = np.zeros((2, 3, 3), dtype=np.uint8)

    first = sampler.capture(enabled=True, policy_runner=policy_runner, observation=observation)
    second = sampler.capture(
        enabled=True,
        policy_runner=policy_runner,
        observation=observation,
        normalization="layer_percentile",
    )

    assert first is not None
    assert second is not None
    assert second is not first
    assert second.layers[0].normalization == "layer_percentile"
    assert policy_runner.calls == 2
