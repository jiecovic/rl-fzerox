# tests/ui/test_watch_cnn.py
from __future__ import annotations

import os

import numpy as np
import pygame

from fzerox_emulator.arrays import Float32Array
from rl_fzerox.core.training.inference import PolicyCnnActivation
from rl_fzerox.ui.watch.runtime.cnn import (
    CnnActivationLayer,
    CnnActivationSampler,
    CnnActivationSnapshot,
    _activation_grid,
    _activation_layer,
    next_cnn_activation_normalization,
)
from rl_fzerox.ui.watch.runtime.worker import _refresh_paused_cnn_activations
from rl_fzerox.ui.watch.view.panels.visuals.cnn import (
    _layer_page_label,
    _layer_pages,
    _mode_hint,
    _normalization_label,
    _plan_layer_layout,
    _separator_positions,
    _stats_summary,
    _unused_tile_rects,
    _weak_channel_summary,
)
from rl_fzerox.ui.watch.view.screen.frame import _create_fonts
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT


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


def test_activation_layer_renders_all_channels_up_to_watch_limit() -> None:
    activation = PolicyCnnActivation(
        name="conv3",
        values=np.zeros((128, 9, 12), dtype=np.float32),
    )

    layer = _activation_layer(activation, normalization="channel")

    assert layer.channel_count == 128
    assert layer.rendered_channel_count == 128
    assert layer.grid_shape == (11, 12)
    assert layer.image.shape == (99, 144, 3)


def test_activation_layer_records_channel_stats_for_dead_channel_checks() -> None:
    activation = PolicyCnnActivation(
        name="conv1",
        values=np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 2.0], [0.0, -2.0]],
            ],
            dtype=np.float32,
        ),
    )

    layer = _activation_layer(activation, normalization="stats")

    assert len(layer.channel_stats) == 2
    assert layer.channel_stats[0].dead is True
    assert layer.channel_stats[1].dead is False
    assert layer.channel_stats[1].active_fraction == 0.5
    assert layer.channel_stats[1].max_abs == 2.0
    assert _stats_summary(layer).startswith("dead 1/2")
    assert _weak_channel_summary(layer) == "weakest max|x| ch: 0:0 1:2.00"


def test_cnn_activation_normalization_toggle_cycles_modes() -> None:
    assert next_cnn_activation_normalization("channel") == "layer_percentile"
    assert next_cnn_activation_normalization("layer_percentile") == "stats"
    assert next_cnn_activation_normalization("stats") == "channel"


def test_cnn_activation_normalization_label_describes_stats_mode() -> None:
    assert _normalization_label(None) == "C cycles CNN view"
    assert (
        _normalization_label(CnnActivationSnapshot(layers=(), normalization="channel"))
        == "view: channel structure"
    )
    assert (
        _normalization_label(CnnActivationSnapshot(layers=(), normalization="layer_percentile"))
        == "view: layer strength"
    )
    assert (
        _normalization_label(CnnActivationSnapshot(layers=(), normalization="stats"))
        == "view: dead check"
    )
    assert _mode_hint(CnnActivationSnapshot(layers=(), normalization="layer_percentile")) == (
        "Shared 0..p99 scale per layer: brighter tile = stronger channel."
    )


def test_separator_positions_are_drawn_after_scaling() -> None:
    assert _separator_positions(280, 4) == (70, 140, 210)
    assert _separator_positions(281, 4) == (70, 140, 211)
    assert _separator_positions(12, 1) == ()


def test_unused_tiles_are_hatched_after_scaling() -> None:
    assert _unused_tile_rects(size=(280, 140), grid_shape=(2, 3), used_tiles=5) == (
        (187, 70, 93, 70),
    )
    assert _unused_tile_rects(size=(280, 140), grid_shape=(2, 2), used_tiles=4) == ()


def test_cnn_activation_layout_uses_columns_for_deep_extractors() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    try:
        fonts = _create_fonts(pygame)
        width = LAYOUT.panel_width - (2 * LAYOUT.panel_padding)
        layers = (
            _activation_layer_snapshot("conv1", image_size=(288, 216), channels=32),
            _activation_layer_snapshot("conv2", image_size=(192, 144), channels=64),
            _activation_layer_snapshot("conv3", image_size=(144, 99), channels=128),
            _activation_layer_snapshot("conv4", image_size=(120, 77), channels=128),
            _activation_layer_snapshot("conv5", image_size=(80, 56), channels=64),
        )

        planned = _plan_layer_layout(
            pygame=pygame,
            fonts=fonts,
            width=width,
            available_height=860,
            layers=layers,
        )

        assert {item.column for item in planned} == {0, 1}
        assert max(item.x_offset + item.target_size[0] for item in planned) <= width
        assert (
            max(
                item.y_offset
                + item.label_surface.get_height()
                + LAYOUT.section_title_gap
                + item.target_size[1]
                for item in planned
            )
            <= 860
        )
    finally:
        pygame.quit()


def test_cnn_activation_layers_are_split_into_readable_pages() -> None:
    layers = tuple(
        _activation_layer_snapshot(f"conv{index}", image_size=(80, 56), channels=64)
        for index in range(1, 6)
    )

    pages = _layer_pages(layers)

    assert tuple(tuple(layer.name for layer in page) for page in pages) == (
        ("conv1", "conv2"),
        ("conv3", "conv4"),
        ("conv5",),
    )
    assert tuple(_layer_page_label(page) for page in pages) == (
        "conv1+conv2",
        "conv3+conv4",
        "conv5",
    )


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


def test_cnn_activation_sampler_can_force_refresh_without_stepping() -> None:
    policy_runner = _PolicyRunner()
    sampler = CnnActivationSampler(refresh_interval_steps=3)
    observation = np.zeros((2, 3, 3), dtype=np.uint8)

    first = sampler.capture(enabled=True, policy_runner=policy_runner, observation=observation)
    second = sampler.capture(
        enabled=True,
        policy_runner=policy_runner,
        observation=observation,
        force_refresh=True,
    )

    assert first is not None
    assert second is not None
    assert second is not first
    assert policy_runner.calls == 2


def test_refresh_paused_cnn_activations_captures_when_tab_is_opened() -> None:
    policy_runner = _PolicyRunner()
    sampler = CnnActivationSampler(refresh_interval_steps=3)
    observation = np.zeros((2, 3, 3), dtype=np.uint8)

    activations, changed = _refresh_paused_cnn_activations(
        current_activations=None,
        cnn_sampler=sampler,
        cnn_visualization_enabled=True,
        previous_cnn_visualization_enabled=False,
        cnn_normalization="channel",
        previous_cnn_normalization="channel",
        policy_runner=policy_runner,
        observation=observation,
    )

    assert activations is not None
    assert changed is True
    assert policy_runner.calls == 1


def _activation_layer_snapshot(
    name: str,
    *,
    image_size: tuple[int, int],
    channels: int,
) -> CnnActivationLayer:
    width, height = image_size
    return CnnActivationLayer(
        name=name,
        image=np.zeros((height, width, 3), dtype=np.uint8),
        channel_count=channels,
        spatial_shape=(height, width),
        grid_shape=(1, 1),
        rendered_channel_count=channels,
        normalization="channel",
    )
