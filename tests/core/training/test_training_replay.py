from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fzerox_emulator.arrays import NumpyArray, UInt8Array
from rl_fzerox.core.training.session.model.replay import (
    LazyImageStateReplayBuffer,
    LazyMaskableReplayBuffer,
    LazyReplayStackMode,
)


def test_lazy_image_state_replay_buffer_requires_next_step_for_nonterminal_sample() -> None:
    buffer = LazyImageStateReplayBuffer(
        buffer_size=8,
        observation_space=_observation_space(image_channels=4),
        action_space=_action_space(),
        frame_stack=3,
        stack_mode="gray",
        minimap_layer=True,
    )
    observation = _image_state_observation(
        image=_stacked_image([_slice(11, 1)] * 3, minimap_value=61),
        state=[0.1, 0.2],
    )
    next_observation = _image_state_observation(
        image=_stacked_image([_slice(11, 1), _slice(11, 1), _slice(22, 1)], minimap_value=62),
        state=[0.3, 0.4],
    )
    buffer.add(
        observation,
        next_observation,
        np.zeros((1, 2), dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        np.array([False]),
        [{}],
    )

    try:
        buffer.sample(1)
    except ValueError as exc:
        assert "valid next observation" in str(exc)
    else:
        raise AssertionError("sample() should reject the latest non-terminal transition")


def test_lazy_image_state_replay_buffer_rebuilds_supported_stack_modes() -> None:
    cases: tuple[tuple[LazyReplayStackMode, int, int], ...] = (
        ("gray", 1, 4),
        ("rgb", 3, 10),
        ("luma_chroma", 2, 7),
    )

    for stack_mode, slice_channels, image_channels in cases:
        buffer = LazyImageStateReplayBuffer(
            buffer_size=8,
            observation_space=_observation_space(image_channels=image_channels),
            action_space=_action_space(),
            frame_stack=3,
            stack_mode=stack_mode,
            minimap_layer=True,
        )
        current_image = _stacked_image([_slice(10, slice_channels)] * 3, minimap_value=80)
        next_image = _stacked_image(
            [
                _slice(10, slice_channels),
                _slice(10, slice_channels),
                _slice(20, slice_channels),
            ],
            minimap_value=81,
        )
        later_image = _stacked_image(
            [
                _slice(10, slice_channels),
                _slice(20, slice_channels),
                _slice(30, slice_channels),
            ],
            minimap_value=82,
        )
        buffer.add(
            _image_state_observation(image=current_image, state=[0.1, 0.2]),
            _image_state_observation(image=next_image, state=[0.3, 0.4]),
            np.zeros((1, 2), dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([False]),
            [{}],
        )
        buffer.add(
            _image_state_observation(image=next_image, state=[0.3, 0.4]),
            _image_state_observation(image=later_image, state=[0.5, 0.6]),
            np.zeros((1, 2), dtype=np.float32),
            np.array([2.0], dtype=np.float32),
            np.array([False]),
            [{}],
        )

        samples = buffer._get_samples_for_pairs(
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            env=None,
        )
        np.testing.assert_array_equal(
            samples.observations["image"].cpu().numpy(),
            current_image[np.newaxis, ...],
        )
        np.testing.assert_allclose(
            samples.observations["state"].cpu().numpy(),
            np.asarray([[0.1, 0.2]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            samples.next_observations["image"].cpu().numpy(),
            next_image[np.newaxis, ...],
        )
        np.testing.assert_allclose(
            samples.next_observations["state"].cpu().numpy(),
            np.asarray([[0.3, 0.4]], dtype=np.float32),
        )


def test_lazy_image_state_replay_buffer_uses_terminal_next_image_override() -> None:
    buffer = LazyImageStateReplayBuffer(
        buffer_size=8,
        observation_space=_observation_space(image_channels=4),
        action_space=_action_space(),
        frame_stack=3,
        stack_mode="gray",
        minimap_layer=True,
    )
    current_image = _stacked_image([_slice(4, 1)] * 3, minimap_value=71)
    terminal_image = _stacked_image(
        [
            _slice(4, 1),
            _slice(5, 1),
            _slice(6, 1),
        ],
        minimap_value=72,
    )
    buffer.add(
        _image_state_observation(image=current_image, state=[0.1, 0.2]),
        _image_state_observation(image=terminal_image, state=[0.9, 1.0]),
        np.zeros((1, 2), dtype=np.float32),
        np.array([3.0], dtype=np.float32),
        np.array([True]),
        [{"terminal_observation": True}],
    )

    samples = buffer.sample(1)
    np.testing.assert_array_equal(
        samples.next_observations["image"].cpu().numpy(),
        terminal_image[np.newaxis, ...],
    )
    np.testing.assert_allclose(
        samples.next_observations["state"].cpu().numpy(),
        np.asarray([[0.9, 1.0]], dtype=np.float32),
    )


def test_lazy_maskable_hybrid_action_dict_replay_buffer_returns_masks() -> None:
    buffer = LazyMaskableReplayBuffer(
        buffer_size=8,
        observation_space=_observation_space(image_channels=4),
        action_space=_action_space(),
        frame_stack=3,
        stack_mode="gray",
        minimap_layer=True,
        mask_dims=5,
    )
    current_image = _stacked_image([_slice(8, 1)] * 3, minimap_value=91)
    terminal_image = _stacked_image(
        [_slice(8, 1), _slice(9, 1), _slice(10, 1)],
        minimap_value=92,
    )
    buffer.add(
        _image_state_observation(image=current_image, state=[0.1, 0.2]),
        _image_state_observation(image=terminal_image, state=[0.3, 0.4]),
        np.zeros((1, 2), dtype=np.float32),
        np.array([4.0], dtype=np.float32),
        np.array([True]),
        [{"terminal_observation": True}],
        action_masks=np.array([[1, 0, 1, 0, 1]], dtype=np.float32),
        next_action_masks=np.array([[0, 1, 0, 1, 0]], dtype=np.float32),
    )

    samples = buffer.sample(1)
    np.testing.assert_array_equal(
        samples.action_masks.cpu().numpy(),
        np.array([[1, 0, 1, 0, 1]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        samples.next_action_masks.cpu().numpy(),
        np.array([[0, 1, 0, 1, 0]], dtype=np.float32),
    )


def _observation_space(*, image_channels: int) -> spaces.Dict:
    return spaces.Dict(
        {
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(1, 1, image_channels),
                dtype=np.uint8,
            ),
            "state": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            ),
        }
    )


def _action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def _image_state_observation(
    *,
    image: UInt8Array,
    state: list[float],
) -> dict[str, NumpyArray]:
    return {
        "image": image[np.newaxis, ...],
        "state": np.asarray([state], dtype=np.float32),
    }


def _slice(start: int, channels: int) -> UInt8Array:
    return np.arange(start, start + channels, dtype=np.uint8).reshape(1, 1, channels)


def _stacked_image(slices: list[UInt8Array], *, minimap_value: int) -> UInt8Array:
    stacked = np.concatenate(slices, axis=2)
    minimap = np.asarray([[[minimap_value]]], dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate([stacked, minimap], axis=2))
