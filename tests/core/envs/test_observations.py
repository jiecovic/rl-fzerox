# tests/core/envs/test_observations.py
import numpy as np

from rl_fzerox.core.envs.observations import (
    STATE_FEATURE_NAMES,
    state_feature_names,
    telemetry_state_vector,
)
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


def test_steer_history_state_profile_appends_short_steer_memory() -> None:
    vector = telemetry_state_vector(
        make_telemetry(speed_kph=750.0),
        state_profile="steer_history",
        steer_left_held=1.0,
        steer_right_held=0.0,
        recent_steer_pressure=-0.5,
    )
    feature_names = state_feature_names("steer_history")

    assert vector.shape == (14,)
    assert feature_names[-3:] == (
        "steer_left_held",
        "steer_right_held",
        "recent_steer_pressure",
    )
    assert vector[feature_names.index("steer_left_held")] == 1.0
    assert vector[feature_names.index("recent_steer_pressure")] == -0.5


def test_race_core_state_profile_keeps_minimal_race_context() -> None:
    vector = telemetry_state_vector(
        make_telemetry(
            speed_kph=750.0,
            energy=89.0,
            reverse_timer=1,
            state_labels=("active", "airborne", "dash_pad_boost"),
        ),
        state_profile="race_core",
    )
    feature_names = state_feature_names("race_core")

    assert feature_names == (
        "speed_norm",
        "energy_frac",
        "reverse_active",
        "airborne",
        "boost_active",
    )
    assert vector.tolist() == [0.5, 0.5, 1.0, 1.0, 1.0]


def test_race_core_action_history_profile_appends_previous_actions() -> None:
    vector = telemetry_state_vector(
        make_telemetry(speed_kph=750.0),
        state_profile="race_core_action_history",
        action_history={
            "prev_steer_1": -0.75,
            "prev_steer_2": 0.25,
            "prev_gas_1": 1.0,
            "prev_air_brake_2": 1.0,
            "prev_boost_1": 1.0,
            "prev_lean_1": -1.0,
            "prev_lean_2": 1.0,
        },
    )
    feature_names = state_feature_names("race_core_action_history")

    assert vector.shape == (13,)
    assert feature_names[-8:] == (
        "prev_steer_1",
        "prev_steer_2",
        "prev_gas_1",
        "prev_gas_2",
        "prev_boost_1",
        "prev_boost_2",
        "prev_lean_1",
        "prev_lean_2",
    )
    values = {name: float(value) for name, value in zip(feature_names, vector, strict=True)}
    assert "prev_air_brake_1" not in values
    assert values["prev_steer_1"] == -0.75
    assert values["prev_steer_2"] == 0.25
    assert values["prev_gas_1"] == 1.0
    assert values["prev_boost_1"] == 1.0
    assert values["prev_lean_1"] == -1.0
    assert values["prev_lean_2"] == 1.0


def test_race_core_action_history_profile_uses_configured_history_length() -> None:
    vector = telemetry_state_vector(
        make_telemetry(speed_kph=750.0),
        state_profile="race_core_action_history",
        action_history_len=3,
        action_history_controls=("steer", "gas", "air_brake", "boost", "lean"),
        action_history={
            "prev_steer_3": -1.0,
            "prev_gas_3": 1.0,
            "prev_boost_2": 1.0,
        },
    )
    feature_names = state_feature_names(
        "race_core_action_history",
        action_history_len=3,
        action_history_controls=("steer", "gas", "air_brake", "boost", "lean"),
    )

    assert vector.shape == (20,)
    assert feature_names[-15:] == (
        "prev_steer_1",
        "prev_steer_2",
        "prev_steer_3",
        "prev_gas_1",
        "prev_gas_2",
        "prev_gas_3",
        "prev_air_brake_1",
        "prev_air_brake_2",
        "prev_air_brake_3",
        "prev_boost_1",
        "prev_boost_2",
        "prev_boost_3",
        "prev_lean_1",
        "prev_lean_2",
        "prev_lean_3",
    )
    values = {name: float(value) for name, value in zip(feature_names, vector, strict=True)}
    assert values["prev_steer_3"] == -1.0
    assert values["prev_gas_3"] == 1.0
    assert values["prev_boost_2"] == 1.0
