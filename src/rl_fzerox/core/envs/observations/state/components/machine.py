# src/rl_fzerox/core/envs/observations/state/components/machine.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.observation_components import ObservationStateComponentSettings
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
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    del component
    return machine_context_features()


def machine_context_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    profile_fields: Mapping[str, float],
) -> list[float]:
    del component, action_history, profile_fields
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
