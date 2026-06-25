# src/rl_fzerox/core/envs/observations/state/components/course.py
from __future__ import annotations

from collections.abc import Mapping

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.observations import ObservationStateComponentSettings
from rl_fzerox.core.envs.observations.state.components.core import component_str
from rl_fzerox.core.envs.observations.state.contexts import (
    course_component_features,
    course_component_values,
)
from rl_fzerox.core.envs.observations.state.types import StateFeature


def course_context_component_features(
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    encoding = component_str(component, "encoding", default="one_hot_builtin")
    return course_component_features(encoding)


def course_context_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    _action_history: Mapping[str, float],
) -> list[float]:
    encoding = component_str(component, "encoding", default="one_hot_builtin")
    return course_component_values(telemetry, encoding=encoding)
