# src/rl_fzerox/core/envs/state_observation/api.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fzerox_emulator import FZeroXTelemetry
from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.domain.observation_components import StateComponentsSettings
from rl_fzerox.core.envs.state_observation.components import (
    action_history_settings_for_observation,
    component_state_values,
    state_vector_spec_from_components,
)
from rl_fzerox.core.envs.state_observation.contexts import (
    course_context_features,
    course_context_values,
    ground_effect_context_features,
    ground_effect_context_values,
)
from rl_fzerox.core.envs.state_observation.history import (
    action_history_feature_names,
    action_history_features,
    action_history_values,
)
from rl_fzerox.core.envs.state_observation.legacy import (
    DEFAULT_STATE_VECTOR_SPEC,
    STATE_VECTOR_SPECS,
    legacy_state_profile_values,
)
from rl_fzerox.core.envs.state_observation.types import (
    OBSERVATION_STATE_DEFAULTS,
    ActionHistoryControl,
    ObservationCourseContext,
    ObservationGroundEffectContext,
    ObservationStateProfile,
    StateVectorSpec,
)


def telemetry_state_vector(
    telemetry: FZeroXTelemetry | None,
    *,
    state_profile: ObservationStateProfile = OBSERVATION_STATE_DEFAULTS.state_profile,
    course_context: ObservationCourseContext = OBSERVATION_STATE_DEFAULTS.course_context,
    ground_effect_context: ObservationGroundEffectContext = (
        OBSERVATION_STATE_DEFAULTS.ground_effect_context
    ),
    left_lean_held: float = 0.0,
    right_lean_held: float = 0.0,
    left_press_age_norm: float = 1.0,
    right_press_age_norm: float = 1.0,
    recent_boost_pressure: float = 0.0,
    steer_left_held: float = 0.0,
    steer_right_held: float = 0.0,
    recent_steer_pressure: float = 0.0,
    action_history_len: int | None = OBSERVATION_STATE_DEFAULTS.action_history_len,
    action_history_controls: tuple[
        ActionHistoryControl, ...
    ] = OBSERVATION_STATE_DEFAULTS.action_history_controls,
    action_history: Mapping[str, float] | None = None,
    state_components: StateComponentsSettings | None = None,
) -> StateVector:
    """Build the normalized scalar policy-state vector from live game telemetry."""

    if state_components is not None:
        values = component_state_values(
            telemetry,
            state_components=state_components,
            action_history=action_history or {},
            legacy_fields={
                "left_lean_held": left_lean_held,
                "right_lean_held": right_lean_held,
                "left_press_age_norm": left_press_age_norm,
                "right_press_age_norm": right_press_age_norm,
                "recent_boost_pressure": recent_boost_pressure,
                "steer_left_held": steer_left_held,
                "steer_right_held": steer_right_held,
                "recent_steer_pressure": recent_steer_pressure,
            },
        )
        expected_count = state_vector_spec_from_components(state_components).count
        if len(values) != expected_count:
            raise ValueError(
                "Observation state component value count does not match feature spec: "
                f"{len(values)} != {expected_count}"
            )
        return np.array(values, dtype=np.float32)

    values = legacy_state_profile_values(
        telemetry,
        profile=state_profile,
        legacy_fields={
            "left_lean_held": left_lean_held,
            "right_lean_held": right_lean_held,
            "left_press_age_norm": left_press_age_norm,
            "right_press_age_norm": right_press_age_norm,
            "recent_boost_pressure": recent_boost_pressure,
            "steer_left_held": steer_left_held,
            "steer_right_held": steer_right_held,
            "recent_steer_pressure": recent_steer_pressure,
        },
    )
    values.extend(course_context_values(telemetry, course_context=course_context))
    values.extend(
        ground_effect_context_values(
            telemetry,
            ground_effect_context=ground_effect_context,
        )
    )
    if action_history_len is not None:
        values.extend(
            action_history_values(
                action_history or {},
                action_history_len=action_history_len,
                action_history_controls=action_history_controls,
            )
        )
    return np.array(values, dtype=np.float32)


def state_vector_spec(
    state_profile: ObservationStateProfile,
    *,
    course_context: ObservationCourseContext = OBSERVATION_STATE_DEFAULTS.course_context,
    ground_effect_context: ObservationGroundEffectContext = (
        OBSERVATION_STATE_DEFAULTS.ground_effect_context
    ),
    action_history_len: int | None = OBSERVATION_STATE_DEFAULTS.action_history_len,
    action_history_controls: tuple[
        ActionHistoryControl, ...
    ] = OBSERVATION_STATE_DEFAULTS.action_history_controls,
    state_components: StateComponentsSettings | None = None,
) -> StateVectorSpec:
    """Return the scalar-state schema selected by config."""

    if state_components is not None:
        return state_vector_spec_from_components(state_components)

    try:
        base_spec = STATE_VECTOR_SPECS[state_profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported observation state profile: {state_profile!r}") from exc
    base_spec = _state_vector_spec_with_course_context(base_spec, course_context=course_context)
    base_spec = _state_vector_spec_with_ground_effect_context(
        base_spec,
        ground_effect_context=ground_effect_context,
    )
    if action_history_len is None:
        return base_spec
    return _state_vector_spec_with_action_history(
        base_spec,
        action_history_len,
        action_history_controls=action_history_controls,
    )


def state_feature_names(
    state_profile: ObservationStateProfile,
    *,
    course_context: ObservationCourseContext = OBSERVATION_STATE_DEFAULTS.course_context,
    ground_effect_context: ObservationGroundEffectContext = (
        OBSERVATION_STATE_DEFAULTS.ground_effect_context
    ),
    action_history_len: int | None = OBSERVATION_STATE_DEFAULTS.action_history_len,
    action_history_controls: tuple[
        ActionHistoryControl, ...
    ] = OBSERVATION_STATE_DEFAULTS.action_history_controls,
    state_components: StateComponentsSettings | None = None,
) -> tuple[str, ...]:
    """Return ordered scalar-state feature names for one profile."""

    return state_vector_spec(
        state_profile,
        course_context=course_context,
        ground_effect_context=ground_effect_context,
        action_history_len=action_history_len,
        action_history_controls=action_history_controls,
        state_components=state_components,
    ).names


def state_feature_count(
    state_profile: ObservationStateProfile,
    *,
    course_context: ObservationCourseContext = OBSERVATION_STATE_DEFAULTS.course_context,
    ground_effect_context: ObservationGroundEffectContext = (
        OBSERVATION_STATE_DEFAULTS.ground_effect_context
    ),
    action_history_len: int | None = OBSERVATION_STATE_DEFAULTS.action_history_len,
    action_history_controls: tuple[
        ActionHistoryControl, ...
    ] = OBSERVATION_STATE_DEFAULTS.action_history_controls,
    state_components: StateComponentsSettings | None = None,
) -> int:
    """Return scalar-state width for one profile."""

    return state_vector_spec(
        state_profile,
        course_context=course_context,
        ground_effect_context=ground_effect_context,
        action_history_len=action_history_len,
        action_history_controls=action_history_controls,
        state_components=state_components,
    ).count


def _state_vector_spec_with_action_history(
    base_spec: StateVectorSpec,
    action_history_len: int,
    *,
    action_history_controls: tuple[ActionHistoryControl, ...],
) -> StateVectorSpec:
    return StateVectorSpec(
        features=(
            *base_spec.features,
            *action_history_features(
                action_history_len,
                action_history_controls=action_history_controls,
            ),
        ),
        speed_normalizer_kph=base_spec.speed_normalizer_kph,
        lean_tap_guard_frames=base_spec.lean_tap_guard_frames,
        recent_boost_window_frames=base_spec.recent_boost_window_frames,
        recent_steer_window_frames=base_spec.recent_steer_window_frames,
    )


def _state_vector_spec_with_course_context(
    base_spec: StateVectorSpec,
    *,
    course_context: ObservationCourseContext,
) -> StateVectorSpec:
    return StateVectorSpec(
        features=(
            *base_spec.features,
            *course_context_features(course_context),
        ),
        speed_normalizer_kph=base_spec.speed_normalizer_kph,
        lean_tap_guard_frames=base_spec.lean_tap_guard_frames,
        recent_boost_window_frames=base_spec.recent_boost_window_frames,
        recent_steer_window_frames=base_spec.recent_steer_window_frames,
    )


def _state_vector_spec_with_ground_effect_context(
    base_spec: StateVectorSpec,
    *,
    ground_effect_context: ObservationGroundEffectContext,
) -> StateVectorSpec:
    return StateVectorSpec(
        features=(
            *base_spec.features,
            *ground_effect_context_features(ground_effect_context),
        ),
        speed_normalizer_kph=base_spec.speed_normalizer_kph,
        lean_tap_guard_frames=base_spec.lean_tap_guard_frames,
        recent_boost_window_frames=base_spec.recent_boost_window_frames,
        recent_steer_window_frames=base_spec.recent_steer_window_frames,
    )


__all__ = [
    "DEFAULT_STATE_VECTOR_SPEC",
    "action_history_feature_names",
    "action_history_settings_for_observation",
    "state_feature_count",
    "state_feature_names",
    "state_vector_spec",
    "telemetry_state_vector",
]
