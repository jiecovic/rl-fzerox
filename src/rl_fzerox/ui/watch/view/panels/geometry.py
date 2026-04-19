# src/rl_fzerox/ui/watch/view/panels/geometry.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, PlayerTelemetry
from rl_fzerox.ui.watch.view.panels.lines import panel_line
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import PanelSection


def track_geometry_sections(
    telemetry: FZeroXTelemetry | None,
) -> tuple[PanelSection, ...]:
    if telemetry is None:
        return ()
    player = telemetry.player
    sliding = _sliding_active(player)
    edge_side, edge_ratio = _lateral_edge_ratio(player)
    near_edge = _near_edge_side(edge_ratio)
    edge_color = _edge_warning_color(edge_ratio)
    return (
        PanelSection(
            title="Track Geometry",
            lines=[
                panel_line(
                    "Segment",
                    _format_optional_int(player.segment_index),
                    PALETTE.text_primary
                    if player.segment_index is not None
                    else PALETTE.text_muted,
                ),
                panel_line("Spline t", f"{player.segment_t:.3f}", PALETTE.text_primary),
                panel_line(
                    "Spline len",
                    f"{player.segment_length_proportion:.3f}",
                    PALETTE.text_primary,
                ),
                panel_line(
                    "Center dist",
                    f"{player.lateral_distance:.1f}",
                    _center_distance_warning_color(player),
                ),
                panel_line(
                    "Lat vel",
                    f"{player.local_lateral_velocity:+.2f}",
                    PALETTE.text_warning if sliding else PALETTE.text_primary,
                ),
                panel_line(
                    "Sliding",
                    "yes" if sliding else "no",
                    PALETTE.text_warning if sliding else PALETTE.text_muted,
                ),
                panel_line(
                    "Lat offset",
                    f"{player.signed_lateral_offset:+.1f}",
                    edge_color,
                ),
                panel_line(
                    "Edge ratio",
                    _format_edge_ratio(edge_side, edge_ratio),
                    edge_color,
                ),
                panel_line(
                    "Near edge",
                    near_edge if near_edge is not None else "no",
                    PALETTE.text_warning if near_edge is not None else PALETTE.text_muted,
                ),
                panel_line(
                    "Disp mag",
                    f"{player.lateral_displacement_magnitude:.1f}",
                    PALETTE.text_muted,
                ),
                panel_line(
                    "Radius L/R",
                    f"{player.current_radius_left:.0f} / {player.current_radius_right:.0f}",
                    PALETTE.text_primary,
                ),
                panel_line(
                    "Ground height",
                    f"{player.height_above_ground:.1f}",
                    PALETTE.text_primary,
                ),
                panel_line(
                    "Vel / acc",
                    f"{player.velocity_magnitude:.2f} / {player.acceleration_magnitude:.2f}",
                    PALETTE.text_primary,
                ),
                panel_line(
                    "Force / slide",
                    f"{player.acceleration_force:.2f} / {player.drift_attack_force:.2f}",
                    PALETTE.text_primary,
                ),
                panel_line(
                    "Impact",
                    _format_impact_debug(player),
                    PALETTE.text_warning if _impact_active(player) else PALETTE.text_muted,
                ),
            ],
        ),
    )


def _format_optional_int(value: int | None) -> str:
    return "--" if value is None else str(value)


def _center_distance_warning_color(player: PlayerTelemetry) -> Color:
    track_half_width = max(player.current_radius_left, player.current_radius_right)
    if track_half_width <= 0.0:
        return PALETTE.text_primary
    if player.lateral_distance > track_half_width * 0.9:
        return PALETTE.text_warning
    return PALETTE.text_primary


def _sliding_active(
    player: PlayerTelemetry,
    *,
    lateral_velocity_threshold: float = 8.0,
) -> bool:
    return not player.airborne and abs(player.local_lateral_velocity) > lateral_velocity_threshold


def _lateral_edge_ratio(player: PlayerTelemetry) -> tuple[str, float | None]:
    offset = player.signed_lateral_offset
    if offset >= 0.0:
        return "left", _safe_ratio(offset, player.current_radius_left)
    return "right", _safe_ratio(offset, player.current_radius_right)


def _safe_ratio(value: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return value / denominator


def _format_edge_ratio(side: str, ratio: float | None) -> str:
    if ratio is None:
        return "--"
    return f"{side} {ratio:+.2f}"


def _near_edge_side(
    edge_ratio: float | None,
    *,
    near_edge_ratio_threshold: float = 0.8,
) -> str | None:
    if edge_ratio is None:
        return None
    if edge_ratio >= near_edge_ratio_threshold:
        return "left"
    if edge_ratio <= -near_edge_ratio_threshold:
        return "right"
    return None


def _edge_warning_color(edge_ratio: float | None) -> Color:
    return (
        PALETTE.text_warning
        if _near_edge_side(edge_ratio) is not None
        else PALETTE.text_primary
    )


def _format_impact_debug(player: PlayerTelemetry) -> str:
    return f"rumble {player.damage_rumble_counter} / recoil {player.recoil_tilt_magnitude:.3f}"


def _impact_active(player: PlayerTelemetry) -> bool:
    return (
        player.collision_recoil
        or player.damage_rumble_counter > 0
        or abs(player.recoil_tilt_magnitude) > 0.001
    )
