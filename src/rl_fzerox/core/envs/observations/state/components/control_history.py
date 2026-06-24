# src/rl_fzerox/core/envs/observations/state/components/control_history.py
from __future__ import annotations

from collections.abc import Mapping

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.observations import ObservationStateComponentSettings
from rl_fzerox.core.envs.observations.state.components.core import (
    component_controls,
    component_int,
)
from rl_fzerox.core.envs.observations.state.history import (
    component_action_history_features,
    component_action_history_values,
)
from rl_fzerox.core.envs.observations.state.types import StateFeature


def control_history_component_features(
    component: ObservationStateComponentSettings,
    *,
    split_lean_history: bool = False,
) -> tuple[StateFeature, ...]:
    length = component_int(component, "length", default=2)
    controls = component_controls(
        component,
        default=("steer", "thrust", "boost", "lean"),
    )
    return component_action_history_features(
        length,
        controls=controls,
        split_lean_history=split_lean_history,
    )


def control_history_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    split_lean_history: bool = False,
) -> list[float]:
    del telemetry
    length = component_int(component, "length", default=2)
    controls = component_controls(
        component,
        default=("steer", "thrust", "boost", "lean"),
    )
    return component_action_history_values(
        action_history,
        action_history_len=length,
        controls=controls,
        split_lean_history=split_lean_history,
    )
