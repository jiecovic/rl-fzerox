# src/rl_fzerox/core/envs/track_bounds.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry, PlayerTelemetry


@dataclass(frozen=True, slots=True)
class TrackEdgeState:
    """Signed lateral track-edge state derived from native spline geometry."""

    side: str
    ratio: float | None
    near_side: str | None
    outside_bounds: bool


def telemetry_outside_track_bounds(telemetry: FZeroXTelemetry | None) -> bool:
    if telemetry is None:
        return False
    return track_edge_state(telemetry.player).outside_bounds


def track_edge_state(
    player: PlayerTelemetry,
    *,
    near_edge_ratio_threshold: float = 0.8,
) -> TrackEdgeState:
    offset = float(player.signed_lateral_offset)
    if offset >= 0.0:
        side = "left"
        ratio = _safe_ratio(offset, float(player.current_radius_left))
    else:
        side = "right"
        ratio = _safe_ratio(offset, float(player.current_radius_right))

    near_side = _near_edge_side(ratio, near_edge_ratio_threshold=near_edge_ratio_threshold)
    return TrackEdgeState(
        side=side,
        ratio=ratio,
        near_side=near_side,
        outside_bounds=ratio is not None and abs(ratio) > 1.0,
    )


def _safe_ratio(value: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return value / denominator


def _near_edge_side(
    edge_ratio: float | None,
    *,
    near_edge_ratio_threshold: float,
) -> str | None:
    if edge_ratio is None:
        return None
    if edge_ratio >= near_edge_ratio_threshold:
        return "left"
    if edge_ratio <= -near_edge_ratio_threshold:
        return "right"
    return None
