# src/rl_fzerox/core/envs/observations/state/contexts.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry

from .types import OBSERVATION_STATE_DEFAULTS, StateFeature


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
