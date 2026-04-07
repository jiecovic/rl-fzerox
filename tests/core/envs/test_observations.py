# tests/core/envs/test_observations.py
import numpy as np

from rl_fzerox.core.config.schema import ObservationConfig
from rl_fzerox.core.envs.observations import FrameStackBuffer, ResizedObservationAdapter


def test_resized_observation_adapter_aspect_corrects_then_downscales() -> None:
    raw_frame = np.zeros((240, 640, 3), dtype=np.uint8)
    raw_frame[:, :320, 0] = 255

    adapter = ResizedObservationAdapter(
        ObservationConfig(width=160, height=120, frame_stack=4, rgb=True)
    )
    observation = adapter.transform(raw_frame, info={"display_aspect_ratio": 4.0 / 3.0})

    assert observation.shape == (120, 160, 3)
    assert observation.dtype == np.uint8
    assert observation[:, :80, 0].mean() > 200


def test_frame_stack_buffer_repeats_reset_frame_and_appends_new_frames() -> None:
    frame_space = ResizedObservationAdapter(
        ObservationConfig(width=4, height=2, frame_stack=4, rgb=True)
    ).observation_space
    stack = FrameStackBuffer(frame_space=frame_space, frame_stack=4)

    first = np.full((2, 4, 3), 10, dtype=np.uint8)
    second = np.full((2, 4, 3), 20, dtype=np.uint8)
    third = np.full((2, 4, 3), 30, dtype=np.uint8)

    reset_obs = stack.reset(first)
    next_obs = stack.append(second)
    later_obs = stack.append(third)

    assert reset_obs.shape == (2, 4, 12)
    assert np.array_equal(reset_obs[:, :, 0:3], first)
    assert np.array_equal(reset_obs[:, :, 9:12], first)
    assert np.array_equal(next_obs[:, :, 9:12], second)
    assert np.array_equal(later_obs[:, :, 6:9], second)
    assert np.array_equal(later_obs[:, :, 9:12], third)
