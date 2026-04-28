# src/rl_fzerox/core/envs/observations/state/components/surface.py
from __future__ import annotations

from collections.abc import Mapping

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.observation_components import ObservationStateComponentSettings
from rl_fzerox.core.envs.course_effects import CourseEffect, course_effect_raw
from rl_fzerox.core.envs.observations.state.types import StateFeature


def surface_state_component_features(
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    del component
    return surface_state_features()


def surface_state_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    profile_fields: Mapping[str, float],
) -> list[float]:
    del component, action_history, profile_fields
    return surface_state_values(telemetry)


def surface_state_features() -> tuple[StateFeature, ...]:
    return (
        StateFeature("surface_state.on_refill_surface", 1.0),
        StateFeature("surface_state.on_dirt_surface", 1.0),
        StateFeature("surface_state.on_ice_surface", 1.0),
    )


def surface_state_values(telemetry: FZeroXTelemetry | None) -> list[float]:
    raw_effect = course_effect_raw(telemetry)
    return [
        1.0 if telemetry is not None and telemetry.player.on_energy_refill else 0.0,
        1.0 if raw_effect == CourseEffect.DIRT else 0.0,
        1.0 if raw_effect == CourseEffect.ICE else 0.0,
    ]
