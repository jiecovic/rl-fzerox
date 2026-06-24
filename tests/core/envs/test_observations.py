# tests/core/envs/test_observations.py
import numpy as np
import pytest
from gymnasium import spaces

from fzerox_emulator import stacked_observation_channels
from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.core.domain.observations import ObservationStateComponentSettings
from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.envs.course_effects import CourseEffect
from rl_fzerox.core.envs.observations import (
    action_history_settings_for_observation,
    build_image_observation_space,
    build_observation_space,
    state_feature_names,
    telemetry_state_vector,
)
from tests.support.fakes import SyntheticBackend
from tests.support.native_objects import encode_state_flags, make_telemetry


def test_native_observation_stack_repeats_the_first_frame() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(preset="crop_84x84", frame_stack=4)

    assert observation.shape == (84, 84, 12)
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

    initial = backend.render_observation(preset="crop_84x84", frame_stack=4)
    backend.step_frames(1)
    next_observation = backend.render_observation(preset="crop_84x84", frame_stack=4)
    backend.step_frames(1)
    later_observation = backend.render_observation(preset="crop_84x84", frame_stack=4)

    assert not np.array_equal(initial, next_observation)
    assert np.array_equal(later_observation[:, :, 0:9], next_observation[:, :, 3:12])


def test_crop_84x84_uses_square_shape() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(
        preset="crop_84x84",
        frame_stack=4,
        stack_mode="rgb",
    )

    assert observation.shape == (84, 84, 12)


def test_custom_60x76_uses_requested_compact_shape() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(
        height=60,
        width=76,
        frame_stack=4,
        stack_mode="rgb",
    )

    assert observation.shape == (60, 76, 12)


def test_custom_68x108_uses_requested_wide_shape() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(
        height=68,
        width=108,
        frame_stack=4,
        stack_mode="rgb",
    )

    assert observation.shape == (68, 108, 12)


def test_crop_84x84_uses_square_grayscale_minimap_shape() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(
        preset="crop_84x84",
        frame_stack=4,
        stack_mode="gray",
        minimap_layer=True,
    )

    assert observation.shape == (84, 84, 5)


def test_rgb_observation_stack_keeps_all_frames_rgb() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(
        height=60,
        width=76,
        frame_stack=4,
        stack_mode="rgb",
    )
    spec = backend.observation_spec(height=60, width=76)
    image_space = build_image_observation_space(
        spec,
        frame_stack=4,
        stack_mode="rgb",
    )
    current_frame = backend.render_observation(
        height=60,
        width=76,
        frame_stack=1,
        stack_mode="rgb",
    )

    assert observation.shape == (60, 76, 12)
    assert image_space.shape == (60, 76, 12)
    assert stacked_observation_channels(3, frame_stack=4, stack_mode="rgb") == 12
    assert np.array_equal(observation[:, :, -3:], current_frame)


def test_gray_observation_stack_encodes_all_frames_as_luma() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(
        preset="crop_84x84",
        frame_stack=4,
        stack_mode="gray",
        minimap_layer=True,
    )
    spec = backend.observation_spec("crop_84x84")
    image_space = build_image_observation_space(
        spec,
        frame_stack=4,
        stack_mode="gray",
        minimap_layer=True,
    )

    assert observation.shape == (84, 84, 5)
    assert image_space.shape == (84, 84, 5)
    assert (
        stacked_observation_channels(
            3,
            frame_stack=4,
            stack_mode="gray",
            minimap_layer=True,
        )
        == 5
    )


def test_luma_chroma_observation_stack_uses_two_channels_per_frame() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(
        preset="crop_84x84",
        frame_stack=4,
        stack_mode="luma_chroma",
        minimap_layer=True,
    )
    spec = backend.observation_spec("crop_84x84")
    image_space = build_image_observation_space(
        spec,
        frame_stack=4,
        stack_mode="luma_chroma",
        minimap_layer=True,
    )

    assert observation.shape == (84, 84, 9)
    assert image_space.shape == (84, 84, 9)


def test_minimap_layer_appends_single_channel_after_frame_stack() -> None:
    backend = SyntheticBackend()
    backend.reset()

    observation = backend.render_observation(
        preset="crop_84x84",
        frame_stack=4,
        stack_mode="rgb",
        minimap_layer=True,
    )
    spec = backend.observation_spec("crop_84x84")
    image_space = build_image_observation_space(
        spec,
        frame_stack=4,
        stack_mode="rgb",
        minimap_layer=True,
    )

    assert observation.shape == (84, 84, 13)
    assert image_space.shape == (84, 84, 13)
    assert (
        stacked_observation_channels(
            3,
            frame_stack=4,
            stack_mode="rgb",
            minimap_layer=True,
        )
        == 13
    )


def test_state_vector_treats_dash_pad_boost_as_boost_active() -> None:
    components = (ObservationStateComponentSettings(name="vehicle_state"),)
    vector = telemetry_state_vector(
        make_telemetry(state_labels=("active", "dash_pad_boost"), boost_timer=0),
        state_components=components,
    )
    feature_names = state_feature_names(state_components=components)

    boost_active_index = feature_names.index("vehicle_state.boost_active")
    assert vector[boost_active_index] == 1.0


def test_component_course_context_adds_course_one_hot() -> None:
    components = (
        ObservationStateComponentSettings(name="vehicle_state"),
        ObservationStateComponentSettings(name="course_context", encoding="one_hot_builtin"),
    )
    vector = telemetry_state_vector(
        make_telemetry(course_index=3, speed_kph=750.0),
        state_components=components,
    )
    feature_names = state_feature_names(state_components=components)

    assert feature_names[-24:] == tuple(
        f"course_context.course_builtin_{index:02d}" for index in range(24)
    )
    assert vector[feature_names.index("course_context.course_builtin_03")] == 1.0
    assert float(np.sum(vector[-24:])) == 1.0


def test_component_course_context_ignores_non_builtin_course_index() -> None:
    components = (
        ObservationStateComponentSettings(name="vehicle_state"),
        ObservationStateComponentSettings(name="course_context", encoding="one_hot_builtin"),
    )
    vector = telemetry_state_vector(
        make_telemetry(course_index=X_CUP_COURSE.course_index),
        state_components=components,
    )
    assert float(np.sum(vector[-24:])) == 0.0


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
    feature_names = state_feature_names(state_components=components)
    values = {name: float(value) for name, value in zip(feature_names, vector, strict=True)}

    assert vector.shape == (51,)
    assert feature_names[:16] == (
        "vehicle_state.speed_norm",
        "vehicle_state.energy_frac",
        "vehicle_state.reverse_active",
        "vehicle_state.airborne",
        "vehicle_state.can_boost",
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
    assert values["vehicle_state.can_boost"] == 1.0
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


def test_vehicle_state_boost_availability_requires_energy() -> None:
    components = (ObservationStateComponentSettings(name="vehicle_state"),)

    low_energy = telemetry_state_vector(
        make_telemetry(
            state_labels=("active", "can_boost"),
            energy=29.0,
            max_energy=178.0,
        ),
        state_components=components,
    )
    available = telemetry_state_vector(
        make_telemetry(
            state_labels=("active", "can_boost"),
            energy=30.0,
            max_energy=178.0,
        ),
        state_components=components,
    )
    locked = telemetry_state_vector(
        make_telemetry(
            state_labels=("active",),
            energy=178.0,
            max_energy=178.0,
        ),
        state_components=components,
    )

    feature_names = state_feature_names(state_components=components)
    boost_index = feature_names.index("vehicle_state.can_boost")
    assert float(low_energy[boost_index]) == 0.0
    assert float(available[boost_index]) == 1.0
    assert float(locked[boost_index]) == 0.0


def test_surface_state_refill_surface_ignores_energy_fullness() -> None:
    components = (ObservationStateComponentSettings(name="surface_state"),)
    telemetry = make_telemetry(
        state_flags=1,
        energy=178.0,
        max_energy=178.0,
    )

    vector = telemetry_state_vector(telemetry, state_components=components)
    feature_names = state_feature_names(state_components=components)
    values = {name: float(value) for name, value in zip(feature_names, vector, strict=True)}

    assert telemetry.player.on_energy_refill is False
    assert values["surface_state.on_refill_surface"] == 1.0


def test_state_components_can_feed_segment_progress_through_progress_slot() -> None:
    components = (
        ObservationStateComponentSettings(name="vehicle_state"),
        ObservationStateComponentSettings(
            name="track_position",
            progress_source="segment_progress",
        ),
    )
    telemetry = make_telemetry(
        lap_distance=20_000.0,
        course_length=80_000.0,
        segment_index=16,
        course_segment_count=65,
    )

    vector = telemetry_state_vector(telemetry, state_components=components)
    feature_names = state_feature_names(state_components=components)
    values = {name: float(value) for name, value in zip(feature_names, vector, strict=True)}

    assert values["track_position.lap_progress"] == 0.25


def test_state_components_expand_independent_lean_history_channels() -> None:
    components = (
        ObservationStateComponentSettings(
            name="control_history",
            length=1,
            controls=("lean",),
        ),
    )

    feature_names = state_feature_names(
        state_components=components,
        split_lean_history=True,
    )
    vector = telemetry_state_vector(
        None,
        state_components=components,
        action_history={
            "prev_lean_left_1": 1.0,
            "prev_lean_right_1": 0.0,
        },
        split_lean_history=True,
    )
    values = {name: float(value) for name, value in zip(feature_names, vector, strict=True)}

    assert feature_names == (
        "control_history.prev_lean_left_1",
        "control_history.prev_lean_right_1",
    )
    assert values["control_history.prev_lean_left_1"] == 1.0
    assert values["control_history.prev_lean_right_1"] == 0.0


def test_state_components_can_disable_progress_slot_without_changing_shape() -> None:
    components = (
        ObservationStateComponentSettings(name="vehicle_state"),
        ObservationStateComponentSettings(
            name="track_position",
            progress_source="none",
        ),
    )
    telemetry = make_telemetry(
        lap_distance=20_000.0,
        course_length=80_000.0,
        segment_index=16,
        course_segment_count=65,
    )

    vector = telemetry_state_vector(telemetry, state_components=components)
    feature_names = state_feature_names(state_components=components)
    values = {name: float(value) for name, value in zip(feature_names, vector, strict=True)}

    assert values["track_position.lap_progress"] == 0.0


def test_state_components_can_opt_into_ground_height_without_affecting_defaults() -> None:
    default_components = _clean_state_components(control_history_enabled=False)
    components_with_height = (
        ObservationStateComponentSettings(name="vehicle_state"),
        ObservationStateComponentSettings(name="machine_context"),
        ObservationStateComponentSettings(
            name="track_position",
            included_features=(
                "track_position.lap_progress",
                "track_position.edge_ratio",
                "track_position.outside_track_bounds",
                "track_position.height_above_ground_norm",
            ),
        ),
        ObservationStateComponentSettings(name="surface_state"),
        ObservationStateComponentSettings(
            name="course_context",
            encoding="one_hot_builtin",
        ),
    )
    telemetry = make_telemetry(height_above_ground=500.0)

    default_feature_names = state_feature_names(state_components=default_components)
    feature_names = state_feature_names(state_components=components_with_height)
    values = {
        name: float(value)
        for name, value in zip(
            feature_names,
            telemetry_state_vector(telemetry, state_components=components_with_height),
            strict=True,
        )
    }

    assert "track_position.height_above_ground_norm" not in default_feature_names
    assert "track_position.height_above_ground_norm" in feature_names
    assert values["track_position.height_above_ground_norm"] == pytest.approx(0.5)


def test_state_components_can_include_individual_features() -> None:
    components = (
        ObservationStateComponentSettings(
            name="track_position",
            included_features=("track_position.edge_ratio",),
        ),
        ObservationStateComponentSettings(
            name="vehicle_state",
            included_features=("vehicle_state.airborne",),
        ),
    )
    telemetry = make_telemetry(signed_lateral_offset=25.0, current_radius_left=100.0)

    feature_names = state_feature_names(state_components=components)
    vector = telemetry_state_vector(telemetry, state_components=components)

    assert feature_names == ("track_position.edge_ratio", "vehicle_state.airborne")
    assert vector.tolist() == pytest.approx([0.25, 0.0])


def test_state_components_clamp_edge_ratio_and_mark_outside_bounds() -> None:
    components = _clean_state_components(control_history_enabled=False)
    telemetry = make_telemetry(
        signed_lateral_offset=150.0,
        current_radius_left=100.0,
    )

    vector = telemetry_state_vector(telemetry, state_components=components)
    feature_names = state_feature_names(state_components=components)
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
    feature_names = state_feature_names(state_components=components)
    values = {name: float(value) for name, value in zip(feature_names, vector, strict=True)}

    assert values["track_position.lap_progress"] == 1.0


def test_state_components_can_disable_control_history() -> None:
    components = _clean_state_components(control_history_enabled=False)
    feature_names = state_feature_names(state_components=components)

    assert all(not name.startswith("control_history.") for name in feature_names)
    assert action_history_settings_for_observation(state_components=components) == (None, ())


def test_state_components_define_observation_space_shape_and_bounds() -> None:
    components = _clean_state_components()
    backend = SyntheticBackend()
    spec = backend.observation_spec(height=60, width=76)

    observation_space = build_observation_space(
        spec,
        frame_stack=4,
        stack_mode="rgb",
        mode="image_state",
        state_components=components,
    )
    assert isinstance(observation_space, spaces.Dict)
    image_space = observation_space.spaces["image"]
    state_space = observation_space.spaces["state"]
    assert isinstance(image_space, spaces.Box)
    assert isinstance(state_space, spaces.Box)

    assert image_space.shape == (60, 76, 12)
    assert state_space.shape == (51,)
    feature_names = state_feature_names(state_components=components)
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
