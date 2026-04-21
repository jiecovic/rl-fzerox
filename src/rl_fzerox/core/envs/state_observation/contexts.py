# src/rl_fzerox/core/envs/state_observation/contexts.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.envs.course_effects import GROUND_EFFECT_FEATURES, ground_effect_flags
from rl_fzerox.core.envs.state_observation.types import (
    OBSERVATION_STATE_DEFAULTS,
    ObservationCourseContext,
    ObservationGroundEffectContext,
    StateFeature,
)


def course_context_features(
    course_context: ObservationCourseContext,
) -> tuple[StateFeature, ...]:
    if course_context == "none":
        return ()
    if course_context == "one_hot_builtin":
        return _course_one_hot_features(prefix="course_builtin_")
    raise ValueError(f"Unsupported observation course context: {course_context!r}")


def course_context_values(
    telemetry: FZeroXTelemetry | None,
    *,
    course_context: ObservationCourseContext,
) -> list[float]:
    if course_context == "none":
        return []
    if course_context != "one_hot_builtin":
        raise ValueError(f"Unsupported observation course context: {course_context!r}")
    return course_one_hot_values(telemetry)


def course_component_features(encoding: str) -> tuple[StateFeature, ...]:
    if encoding == "none":
        return ()
    if encoding == "one_hot_builtin":
        return _course_one_hot_features(prefix="course_context.course_builtin_")
    raise ValueError(f"Unsupported course-context encoding: {encoding!r}")


def course_component_values(
    telemetry: FZeroXTelemetry | None,
    *,
    encoding: str,
) -> list[float]:
    if encoding == "none":
        return []
    if encoding != "one_hot_builtin":
        raise ValueError(f"Unsupported course-context encoding: {encoding!r}")
    return course_one_hot_values(telemetry)


def course_one_hot_values(telemetry: FZeroXTelemetry | None) -> list[float]:
    values = [0.0] * OBSERVATION_STATE_DEFAULTS.builtin_course_count
    if telemetry is None:
        return values
    course_index = int(telemetry.course_index)
    if 0 <= course_index < OBSERVATION_STATE_DEFAULTS.builtin_course_count:
        values[course_index] = 1.0
    return values


def _course_one_hot_features(*, prefix: str) -> tuple[StateFeature, ...]:
    return tuple(
        StateFeature(f"{prefix}{index:02d}", 1.0)
        for index in range(OBSERVATION_STATE_DEFAULTS.builtin_course_count)
    )


def ground_effect_context_features(
    ground_effect_context: ObservationGroundEffectContext,
) -> tuple[StateFeature, ...]:
    if ground_effect_context == "none":
        return ()
    if ground_effect_context == "effect_flags":
        return tuple(StateFeature(name, 1.0) for name in GROUND_EFFECT_FEATURES)
    raise ValueError(f"Unsupported observation ground-effect context: {ground_effect_context!r}")


def ground_effect_context_values(
    telemetry: FZeroXTelemetry | None,
    *,
    ground_effect_context: ObservationGroundEffectContext,
) -> list[float]:
    if ground_effect_context == "none":
        return []
    if ground_effect_context != "effect_flags":
        raise ValueError(
            f"Unsupported observation ground-effect context: {ground_effect_context!r}"
        )
    return list(ground_effect_flags(telemetry))
