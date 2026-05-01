# src/rl_fzerox/core/envs/observations/state/components/track.py
from __future__ import annotations

from collections.abc import Mapping

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.observation_components import (
    ObservationStateComponentSettings,
    TrackPositionProgressSourceName,
)
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
    del action_history, profile_fields
    return track_position_values(
        telemetry,
        progress_source=component.progress_source or "lap_progress",
    )


def track_position_features() -> tuple[StateFeature, ...]:
    return (
        StateFeature("track_position.lap_progress", 1.0),
        StateFeature("track_position.edge_ratio", 1.0, low=-1.0),
        StateFeature("track_position.outside_track_bounds", 1.0),
    )


def track_position_values(
    telemetry: FZeroXTelemetry | None,
    *,
    progress_source: TrackPositionProgressSourceName = "lap_progress",
) -> list[float]:
    progress = track_position_progress(telemetry, progress_source=progress_source)
    if telemetry is None:
        return [progress, 0.0, 0.0]
    edge_state = track_edge_state(telemetry.player)
    edge_ratio = edge_state.ratio
    if edge_ratio is None:
        return [progress, 0.0, 0.0]
    return [
        progress,
        clamp(edge_ratio, -1.0, 1.0),
        1.0 if edge_state.outside_bounds else 0.0,
    ]


def track_position_progress(
    telemetry: FZeroXTelemetry | None,
    *,
    progress_source: TrackPositionProgressSourceName,
) -> float:
    match progress_source:
        case "lap_progress":
            return lap_progress_fraction(telemetry)
        case "segment_progress":
            return segment_progress_fraction(telemetry)
        case "none":
            return 0.0
        case _:
            raise ValueError(f"Unsupported track position progress source: {progress_source!r}")


def lap_progress_fraction(telemetry: FZeroXTelemetry | None) -> float:
    if telemetry is None or telemetry.course_length <= 0.0:
        return 0.0
    return clamp(
        float(telemetry.player.lap_distance) / float(telemetry.course_length),
        0.0,
        1.0,
    )


def segment_progress_fraction(telemetry: FZeroXTelemetry | None) -> float:
    if telemetry is None:
        return 0.0
    segment_index = telemetry.player.segment_index
    segment_count = int(telemetry.course_segment_count)
    if segment_index is None or segment_count <= 1:
        return 0.0
    return clamp(float(segment_index) / float(segment_count - 1), 0.0, 1.0)
