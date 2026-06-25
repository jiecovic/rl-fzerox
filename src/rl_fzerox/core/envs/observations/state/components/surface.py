# src/rl_fzerox/core/envs/observations/state/components/surface.py
"""State-vector features for course surface effects.

Surface features expose whether the machine is on refill, dirt, ice, dash, jump,
or related course effects. The raw effect decoding lives in course-effect
helpers; this module only packages it for observation state.
"""
from __future__ import annotations

from collections.abc import Mapping

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.observations import ObservationStateComponentSettings
from rl_fzerox.core.envs.course_effects import CourseEffect, course_effect_raw, on_refill_surface
from rl_fzerox.core.envs.observations.state.types import StateFeature


def surface_state_component_features(
    _component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    return surface_state_features()


def surface_state_component_values(
    telemetry: FZeroXTelemetry | None,
    _component: ObservationStateComponentSettings,
    _action_history: Mapping[str, float],
) -> list[float]:
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
        1.0 if on_refill_surface(telemetry) else 0.0,
        1.0 if raw_effect == CourseEffect.DIRT else 0.0,
        1.0 if raw_effect == CourseEffect.ICE else 0.0,
    ]
