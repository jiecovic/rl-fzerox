# src/rl_fzerox/core/envs/observations/state/components/track.py
from __future__ import annotations

from collections.abc import Mapping

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.observation_components import ObservationStateComponentSettings
from rl_fzerox.core.envs.observations.state.types import StateFeature
from rl_fzerox.core.envs.observations.state.utils import clamp
from rl_fzerox.core.envs.track_bounds import track_edge_state


def track_position_component_features(
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    del component
    return track_position_features()


def track_position_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    profile_fields: Mapping[str, float],
) -> list[float]:
    del component, action_history, profile_fields
    return track_position_values(telemetry)


def track_position_features() -> tuple[StateFeature, ...]:
    return (
        StateFeature("track_position.lap_progress", 1.0),
        StateFeature("track_position.edge_ratio", 1.0, low=-1.0),
        StateFeature("track_position.outside_track_bounds", 1.0),
    )


def track_position_values(telemetry: FZeroXTelemetry | None) -> list[float]:
    lap_progress = lap_progress_fraction(telemetry)
    if telemetry is None:
        return [lap_progress, 0.0, 0.0]
    edge_state = track_edge_state(telemetry.player)
    edge_ratio = edge_state.ratio
    if edge_ratio is None:
        return [lap_progress, 0.0, 0.0]
    return [
        lap_progress,
        clamp(edge_ratio, -1.0, 1.0),
        1.0 if edge_state.outside_bounds else 0.0,
    ]


def lap_progress_fraction(telemetry: FZeroXTelemetry | None) -> float:
    if telemetry is None or telemetry.course_length <= 0.0:
        return 0.0
    return clamp(
        float(telemetry.player.lap_distance) / float(telemetry.course_length),
        0.0,
        1.0,
    )
