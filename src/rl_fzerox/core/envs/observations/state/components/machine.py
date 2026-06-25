# src/rl_fzerox/core/envs/observations/state/components/machine.py
"""State-vector features for machine orientation and dynamics.

These features are direct telemetry-derived vehicle dynamics such as velocity,
heading, drift, and angular motion. They are normalized here before being
appended to the policy state vector.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.observations import ObservationStateComponentSettings
from rl_fzerox.core.envs.observations.state.types import StateFeature
from rl_fzerox.core.envs.observations.state.utils import clamp


@dataclass(frozen=True, slots=True)
class MachineContextNormalization:
    """Fixed stock-machine bounds used to scale vehicle setup into [0, 1]."""

    stat_max: float = 4.0
    weight_min: float = 780.0
    weight_max: float = 2340.0

    @property
    def weight_range(self) -> float:
        return self.weight_max - self.weight_min


MACHINE_CONTEXT_NORMALIZATION = MachineContextNormalization()


def machine_context_component_features(
    _component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    return machine_context_features()


def machine_context_component_values(
    telemetry: FZeroXTelemetry | None,
    _component: ObservationStateComponentSettings,
    _action_history: Mapping[str, float],
) -> list[float]:
    return machine_context_values(telemetry)


def machine_context_features() -> tuple[StateFeature, ...]:
    return (
        StateFeature("machine_context.body_stat", 1.0),
        StateFeature("machine_context.boost_stat", 1.0),
        StateFeature("machine_context.grip_stat", 1.0),
        StateFeature("machine_context.weight", 1.0),
        StateFeature("machine_context.engine", 1.0),
    )


def machine_context_values(telemetry: FZeroXTelemetry | None) -> list[float]:
    if telemetry is None:
        return [0.0] * len(machine_context_features())
    player = telemetry.player
    return [
        machine_stat_value(float(player.machine_body_stat)),
        machine_stat_value(float(player.machine_boost_stat)),
        machine_stat_value(float(player.machine_grip_stat)),
        machine_weight_value(float(player.machine_weight)),
        clamp(float(player.engine_setting), 0.0, 1.0),
    ]


def machine_stat_value(raw_value: float) -> float:
    return clamp(raw_value / MACHINE_CONTEXT_NORMALIZATION.stat_max, 0.0, 1.0)


def machine_weight_value(raw_value: float) -> float:
    return clamp(
        (raw_value - MACHINE_CONTEXT_NORMALIZATION.weight_min)
        / MACHINE_CONTEXT_NORMALIZATION.weight_range,
        0.0,
        1.0,
    )
