# tests/core/envs/test_observations.py
import numpy as np
import pytest
from gymnasium import spaces

from fzerox_emulator import stacked_observation_channels
from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.core.domain.observation_components import ObservationStateComponentSettings
from rl_fzerox.core.envs.course_effects import CourseEffect
from rl_fzerox.core.envs.observation_image import build_image_observation_space
from rl_fzerox.core.envs.observations import (
    STATE_FEATURE_NAMES,
    action_history_settings_for_observation,
    build_observation_space,
    state_feature_names,
    telemetry_state_vector,
)
from tests.support.fakes import SyntheticBackend
from tests.support.native_objects import encode_state_flags, make_telemetry


def test_native_observation_stack_repeats_the_first_frame() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(preset="crop_84x116", frame_stack=4)

    assert observation.shape == (84, 116, 12)
    assert np.array_equal(observation[:, :, 0:3], observation[:, :, 3:6])
    assert np.array_equal(observation[:, :, 3:6], observation[:, :, 6:9])
    assert np.array_equal(observation[:, :, 6:9], observation[:, :, 9:12])


def test_native_observation_stack_shifts_forward_on_new_frames() -> None:
    class DistinctFrameBackend(SyntheticBackend):
        def _build_frame(self) -> RgbFrame:
            value = np.uint8((self.frame_index * 40) % 255)
            return np.full((240, 640, 3), value, dtype=np.uint8)

    backend = DistinctFrameBackend()
    backend.reset()

    initial = backend.render_observation(preset="crop_84x116", frame_stack=4)
    backend.step_frames(1)
    next_observation = backend.render_observation(preset="crop_84x116", frame_stack=4)
    backend.step_frames(1)
    later_observation = backend.render_observation(preset="crop_84x116", frame_stack=4)

    assert not np.array_equal(initial, next_observation)
    assert np.array_equal(later_observation[:, :, 0:9], next_observation[:, :, 3:12])


def test_crop_92x124_uses_larger_aspect_correct_shape() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(preset="crop_92x124", frame_stack=4)

    assert observation.shape == (92, 124, 12)


def test_crop_116x164_uses_largest_default_aspect_correct_shape() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(preset="crop_116x164", frame_stack=4)

    assert observation.shape == (116, 164, 12)


def test_crop_98x130_uses_compact_deep_aspect_correct_shape() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(preset="crop_98x130", frame_stack=3)

    assert observation.shape == (98, 130, 9)


def test_crop_66x82_uses_small_aspect_correct_shape() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(
        preset="crop_66x82",
        frame_stack=4,
        stack_mode="rgb_gray",
    )

    assert observation.shape == (66, 82, 6)


def test_rgb_gray_observation_stack_keeps_current_frame_rgb() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(
        preset="crop_98x130",
        frame_stack=4,
        stack_mode="rgb_gray",
    )
    spec = backend.observation_spec("crop_98x130")
    image_space = build_image_observation_space(
        spec,
        frame_stack=4,
        stack_mode="rgb_gray",
    )
    current_frame = backend.render_observation(
        preset="crop_98x130",
        frame_stack=1,
        stack_mode="rgb",
    )

    assert observation.shape == (98, 130, 6)
    assert image_space.shape == (98, 130, 6)
    assert stacked_observation_channels(3, frame_stack=4, stack_mode="rgb_gray") == 6
    assert np.array_equal(observation[:, :, 0], observation[:, :, 1])
    assert np.array_equal(observation[:, :, 1], observation[:, :, 2])
    assert np.array_equal(observation[:, :, -3:], current_frame)


def test_minimap_layer_appends_single_channel_after_frame_stack() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(
        preset="crop_66x82",
        frame_stack=4,
        stack_mode="rgb_gray",
        minimap_layer=True,
    )
    spec = backend.observation_spec("crop_66x82")
    image_space = build_image_observation_space(
        spec,
        frame_stack=4,
        stack_mode="rgb_gray",
        minimap_layer=True,
    )

    assert observation.shape == (66, 82, 7)
    assert image_space.shape == (66, 82, 7)
    assert (
        stacked_observation_channels(
            3,
            frame_stack=4,
            stack_mode="rgb_gray",
            minimap_layer=True,
        )
        == 7
    )


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
        "can_boost",
        "boost_active",
    )
    assert vector.tolist() == [0.5, 0.5, 1.0, 1.0, 0.0, 1.0]


def test_builtin_course_context_adds_course_one_hot() -> None:
    vector = telemetry_state_vector(
        make_telemetry(course_index=3, speed_kph=750.0),
        state_profile="race_core",
        course_context="one_hot_builtin",
    )
    feature_names = state_feature_names(
        "race_core",
        course_context="one_hot_builtin",
    )

    assert vector.shape == (30,)
    assert feature_names[6:30] == tuple(f"course_builtin_{index:02d}" for index in range(24))
    assert vector[feature_names.index("course_builtin_03")] == 1.0
    assert float(sum(vector[6:30])) == 1.0


def test_builtin_course_context_ignores_non_builtin_course_index() -> None:
    vector = telemetry_state_vector(
        make_telemetry(course_index=48),
        state_profile="race_core",
        course_context="one_hot_builtin",
    )

    assert vector.shape == (30,)
    assert float(sum(vector[6:30])) == 0.0


def test_ground_effect_context_adds_course_effect_flags() -> None:
    vector = telemetry_state_vector(
        make_telemetry(state_flags=2),
        state_profile="race_core",
        ground_effect_context="effect_flags",
    )
    feature_names = state_feature_names(
        "race_core",
        ground_effect_context="effect_flags",
    )

    assert vector.shape == (10,)
    assert feature_names[6:10] == (
        "ground_pit",
        "ground_dash",
        "ground_dirt",
        "ground_ice",
    )
    assert vector[feature_names.index("ground_pit")] == 0.0
    assert vector[feature_names.index("ground_dash")] == 0.0
    assert vector[feature_names.index("ground_dirt")] == 1.0
    assert vector[feature_names.index("ground_ice")] == 0.0


def test_observation_contexts_precede_action_history() -> None:
    vector = telemetry_state_vector(
        make_telemetry(course_index=1, state_flags=4),
        state_profile="race_core",
        course_context="one_hot_builtin",
        ground_effect_context="effect_flags",
        action_history_len=1,
        action_history={"prev_steer_1": 0.5},
    )
    feature_names = state_feature_names(
        "race_core",
        course_context="one_hot_builtin",
        ground_effect_context="effect_flags",
        action_history_len=1,
    )

    assert vector.shape == (38,)
    assert feature_names[6:30] == tuple(f"course_builtin_{index:02d}" for index in range(24))
    assert feature_names[30:34] == (
        "ground_pit",
        "ground_dash",
        "ground_dirt",
        "ground_ice",
    )
    assert feature_names[34:] == (
        "prev_steer_1",
        "prev_gas_1",
        "prev_boost_1",
        "prev_lean_1",
    )
    assert vector[feature_names.index("course_builtin_01")] == 1.0
    assert vector[feature_names.index("ground_ice")] == 1.0
    assert vector[feature_names.index("prev_steer_1")] == 0.5


def test_action_history_len_none_disables_previous_actions() -> None:
    vector = telemetry_state_vector(
        make_telemetry(speed_kph=750.0),
        state_profile="race_core",
        action_history_len=None,
        action_history={"prev_steer_1": -1.0},
    )
    feature_names = state_feature_names("race_core", action_history_len=None)

    assert vector.shape == (6,)
    assert all(not name.startswith("prev_") for name in feature_names)


def test_action_history_len_zero_is_invalid() -> None:
    with pytest.raises(ValueError, match="positive or None"):
        state_feature_names("race_core", action_history_len=0)


def test_action_history_len_appends_previous_actions_to_race_core() -> None:
    vector = telemetry_state_vector(
        make_telemetry(speed_kph=750.0),
        state_profile="race_core",
        action_history_len=2,
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
    feature_names = state_feature_names("race_core", action_history_len=2)

    assert vector.shape == (14,)
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


def test_action_history_len_uses_configured_history_length() -> None:
    vector = telemetry_state_vector(
        make_telemetry(speed_kph=750.0),
        state_profile="race_core",
        action_history_len=3,
        action_history_controls=("steer", "gas", "air_brake", "boost", "lean"),
        action_history={
            "prev_steer_3": -1.0,
            "prev_gas_3": 1.0,
            "prev_boost_2": 1.0,
        },
    )
    feature_names = state_feature_names(
        "race_core",
        action_history_len=3,
        action_history_controls=("steer", "gas", "air_brake", "boost", "lean"),
    )

    assert vector.shape == (21,)
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


def test_state_components_build_clean_prefixed_state_vector() -> None:
    components = _clean_state_components()
    telemetry = make_telemetry(
        course_index=2,
        speed_kph=750.0,
        energy=89.0,
        max_energy=178.0,
        reverse_timer=1,
        state_flags=encode_state_flags(("active", "can_boost")) | int(CourseEffect.DIRT),
        local_lateral_velocity=16.0,
        signed_lateral_offset=-90.0,
        current_radius_left=120.0,
        current_radius_right=100.0,
        lap_distance=20_000.0,
        course_length=80_000.0,
        machine_body_stat=1,
        machine_boost_stat=2,
        machine_grip_stat=3,
        machine_weight=1560,
        engine_setting=0.7,
    )

    vector = telemetry_state_vector(
        telemetry,
        state_components=components,
        action_history={
            "prev_steer_1": -0.5,
            "prev_gas_1": 1.0,
            "prev_boost_2": 1.0,
            "prev_lean_1": -1.0,
        },
    )
    feature_names = state_feature_names("race_core", state_components=components)
    values = {name: float(value) for name, value in zip(feature_names, vector, strict=True)}

    assert vector.shape == (51,)
    assert feature_names[:16] == (
        "vehicle_state.speed_norm",
        "vehicle_state.energy_frac",
        "vehicle_state.reverse_active",
        "vehicle_state.airborne",
        "vehicle_state.boost_unlocked",
        "vehicle_state.boost_active",
        "vehicle_state.lateral_velocity_norm",
        "vehicle_state.sliding_active",
        "machine_context.body_stat",
        "machine_context.boost_stat",
        "machine_context.grip_stat",
        "machine_context.weight",
        "machine_context.engine",
        "track_position.lap_progress",
        "track_position.edge_ratio",
        "track_position.outside_track_bounds",
    )
    assert values["vehicle_state.speed_norm"] == 0.5
    assert values["vehicle_state.energy_frac"] == 0.5
    assert values["vehicle_state.reverse_active"] == 1.0
    assert values["vehicle_state.boost_unlocked"] == 1.0
    assert values["vehicle_state.lateral_velocity_norm"] == 0.5
    assert values["vehicle_state.sliding_active"] == 1.0
    assert values["machine_context.body_stat"] == 0.25
    assert values["machine_context.boost_stat"] == 0.5
    assert values["machine_context.grip_stat"] == 0.75
    assert values["machine_context.weight"] == 0.5
    assert values["machine_context.engine"] == pytest.approx(0.7)
    assert values["track_position.lap_progress"] == 0.25
    assert values["track_position.edge_ratio"] == pytest.approx(-0.9)
    assert values["track_position.outside_track_bounds"] == 0.0
    assert values["surface_state.on_dirt_surface"] == 1.0
    assert values["course_context.course_builtin_02"] == 1.0
    assert values["control_history.prev_steer_1"] == -0.5
    assert values["control_history.prev_thrust_1"] == 1.0
    assert values["control_history.prev_boost_2"] == 1.0
    assert values["control_history.prev_lean_1"] == -1.0


def test_state_components_can_zero_selected_components_without_changing_shape() -> None:
    components = _clean_state_components()
    telemetry = make_telemetry(
        course_index=2,
        speed_kph=750.0,
        signed_lateral_offset=-90.0,
        current_radius_right=100.0,
        lap_distance=20_000.0,
        course_length=80_000.0,
    )

    vector = telemetry_state_vector(
        telemetry,
        state_components=components,
        zeroed_state_components=frozenset({"track_position"}),
    )
    feature_names = state_feature_names("race_core", state_components=components)
    values = {name: float(value) for name, value in zip(feature_names, vector, strict=True)}

    assert vector.shape == (51,)
    assert values["vehicle_state.speed_norm"] == 0.5
    assert values["track_position.lap_progress"] == 0.0
    assert values["track_position.edge_ratio"] == 0.0
    assert values["track_position.outside_track_bounds"] == 0.0
    assert feature_names.index("track_position.lap_progress") == 13


def test_state_components_clamp_edge_ratio_and_mark_outside_bounds() -> None:
    components = _clean_state_components(control_history_enabled=False)
    telemetry = make_telemetry(
        signed_lateral_offset=150.0,
        current_radius_left=100.0,
    )

    vector = telemetry_state_vector(telemetry, state_components=components)
    feature_names = state_feature_names("race_core", state_components=components)
    values = {name: float(value) for name, value in zip(feature_names, vector, strict=True)}

    assert values["track_position.edge_ratio"] == 1.0
    assert values["track_position.outside_track_bounds"] == 1.0


def test_state_components_clamp_lap_progress() -> None:
    components = _clean_state_components(control_history_enabled=False)
    telemetry = make_telemetry(
        lap_distance=90_000.0,
        course_length=80_000.0,
    )

    vector = telemetry_state_vector(telemetry, state_components=components)
    feature_names = state_feature_names("race_core", state_components=components)
    values = {name: float(value) for name, value in zip(feature_names, vector, strict=True)}

    assert values["track_position.lap_progress"] == 1.0


def test_state_components_can_disable_control_history() -> None:
    components = _clean_state_components(control_history_enabled=False)
    feature_names = state_feature_names("race_core", state_components=components)

    assert all(not name.startswith("control_history.") for name in feature_names)
    assert action_history_settings_for_observation(
        state_components=components,
        fallback_len=2,
        fallback_controls=("steer", "gas"),
    ) == (None, ())


def test_state_components_define_observation_space_shape_and_bounds() -> None:
    components = _clean_state_components()
    backend = SyntheticBackend()
    spec = backend.observation_spec("crop_66x82")

    observation_space = build_observation_space(
        spec,
        frame_stack=4,
        stack_mode="rgb_gray",
        mode="image_state",
        state_components=components,
    )
    assert isinstance(observation_space, spaces.Dict)
    image_space = observation_space.spaces["image"]
    state_space = observation_space.spaces["state"]
    assert isinstance(image_space, spaces.Box)
    assert isinstance(state_space, spaces.Box)

    assert image_space.shape == (66, 82, 6)
    assert state_space.shape == (51,)
    feature_names = state_feature_names("race_core", state_components=components)
    lateral_index = feature_names.index("vehicle_state.lateral_velocity_norm")
    edge_index = feature_names.index("track_position.edge_ratio")
    assert state_space.low[lateral_index] == -1.0
    assert state_space.low[edge_index] == -1.0


def _clean_state_components(
    *,
    control_history_enabled: bool = True,
) -> tuple[ObservationStateComponentSettings, ...]:
    components: list[ObservationStateComponentSettings] = [
        ObservationStateComponentSettings(name="vehicle_state"),
        ObservationStateComponentSettings(name="machine_context"),
        ObservationStateComponentSettings(name="track_position"),
        ObservationStateComponentSettings(name="surface_state"),
        ObservationStateComponentSettings(
            name="course_context",
            encoding="one_hot_builtin",
        ),
    ]
    if control_history_enabled:
        components.append(
            ObservationStateComponentSettings(
                name="control_history",
                length=2,
                controls=("steer", "thrust", "boost", "lean"),
            )
        )
    return tuple(components)
