# tests/core/envs/test_observations.py
import numpy as np

from rl_fzerox.core.envs.observations import STATE_FEATURE_NAMES, telemetry_state_vector
from tests.support.fakes import SyntheticBackend
from tests.support.native_objects import make_telemetry


def test_native_observation_stack_repeats_the_first_frame() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(preset="native_crop_v1", frame_stack=4)

    assert observation.shape == (84, 116, 12)
    assert np.array_equal(observation[:, :, 0:3], observation[:, :, 3:6])
    assert np.array_equal(observation[:, :, 3:6], observation[:, :, 6:9])
    assert np.array_equal(observation[:, :, 6:9], observation[:, :, 9:12])


def test_native_observation_stack_shifts_forward_on_new_frames() -> None:
    class DistinctFrameBackend(SyntheticBackend):
        def _build_frame(self) -> np.ndarray:
            value = np.uint8((self.frame_index * 40) % 255)
            return np.full((240, 640, 3), value, dtype=np.uint8)

    backend = DistinctFrameBackend()
    backend.reset()

    initial = backend.render_observation(preset="native_crop_v1", frame_stack=4)
    backend.step_frames(1)
    next_observation = backend.render_observation(preset="native_crop_v1", frame_stack=4)
    backend.step_frames(1)
    later_observation = backend.render_observation(preset="native_crop_v1", frame_stack=4)

    assert not np.array_equal(initial, next_observation)
    assert np.array_equal(later_observation[:, :, 0:9], next_observation[:, :, 3:12])


def test_native_observation_v2_uses_larger_aspect_correct_shape() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(preset="native_crop_v2", frame_stack=4)

    assert observation.shape == (92, 124, 12)


def test_native_observation_v3_uses_largest_default_aspect_correct_shape() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(preset="native_crop_v3", frame_stack=4)

    assert observation.shape == (116, 164, 12)


def test_state_vector_treats_dash_pad_boost_as_boost_active() -> None:
    vector = telemetry_state_vector(
        make_telemetry(state_labels=("active", "dash_pad_boost"), boost_timer=0)
    )

    boost_active_index = STATE_FEATURE_NAMES.index("boost_active")
    assert vector[boost_active_index] == 1.0
