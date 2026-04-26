# src/rl_fzerox/ui/watch/view/panels/game.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.envs.course_effects import CourseEffect, course_effect_raw
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.core.envs.track_bounds import track_edge_state
from rl_fzerox.ui.watch.view.panels.format import (
    _float_info,
    _format_distance,
    _format_mode_name,
    _format_race_time_ms,
    _int_info,
)
from rl_fzerox.ui.watch.view.panels.lines import panel_line
from rl_fzerox.ui.watch.view.panels.viz import _flag_viz
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelLine, PanelSection


def game_overview_section(
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
    *,
    stuck_min_speed_kph: float,
) -> PanelSection:
    if telemetry is None:
        return PanelSection(
            title="Race",
            lines=[
                panel_line("Status", "unavailable", PALETTE.text_warning),
            ],
        )

    raw_course_effect = course_effect_raw(telemetry)
    edge_state = track_edge_state(telemetry.player)
    lines = [
        panel_line("Mode", _format_mode_name(telemetry.game_mode_name), PALETTE.text_primary),
        panel_line("Course", _format_course_name(info, telemetry), PALETTE.text_primary),
        panel_line(
            "Time",
            _format_race_time_ms(telemetry.player.race_time_ms),
            PALETTE.text_primary,
        ),
        panel_line("Speed", f"{telemetry.player.speed_kph:.1f} km/h", PALETTE.text_primary),
        panel_line(
            "Energy",
            f"{telemetry.player.energy:.1f} / {telemetry.player.max_energy:.1f}",
            PALETTE.text_primary,
        ),
        panel_line("Lap", str(telemetry.player.lap), PALETTE.text_primary),
        panel_line("Position", _format_position(telemetry), PALETTE.text_primary),
        panel_line(
            "Total progress",
            _format_total_progress(telemetry),
            PALETTE.text_primary,
        ),
        panel_line(
            "Lap progress",
            _format_lap_progress(telemetry),
            PALETTE.text_primary,
        ),
    ]
    if difficulty_line := _difficulty_line(telemetry):
        lines.insert(1, difficulty_line)

    return PanelSection(
        title="Race",
        lines=lines,
        flag_viz=_flag_viz(
            telemetry.player.state_labels,
            boost_active=telemetry_boost_active(telemetry),
            reverse_detected=telemetry.player.reverse_timer > 0,
            low_speed_detected=telemetry.player.speed_kph < stuck_min_speed_kph,
            energy_depleted=info.get("termination_reason") == "energy_depleted",
            energy_refill_detected=telemetry.player.on_energy_refill,
            dirt_detected=raw_course_effect == CourseEffect.DIRT,
            ice_detected=raw_course_effect == CourseEffect.ICE,
            track_edge_detected=edge_state.near_side is not None,
            outside_track_bounds=edge_state.outside_bounds,
            energy_loss_detected=_float_info(info, "energy_loss_total") > 0.0,
            damage_taken_detected=_int_info(info, "damage_taken_frames") > 0,
        ),
    )


def game_detail_section(
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> PanelSection:
    if telemetry is None:
        return PanelSection(
            title="Game Details",
            lines=[
                panel_line("Status", "unavailable", PALETTE.text_warning),
            ],
        )

    recoil_magnitude = (
        telemetry.player.recoil_tilt_magnitude if telemetry.player.collision_recoil else 0.0
    )
    return PanelSection(
        title="Game Details",
        lines=[
            panel_line("Camera", _format_camera_setting(telemetry), PALETTE.text_primary),
            panel_line("Vehicle", _format_vehicle_setup(info), PALETTE.text_primary),
            panel_line(
                "Recoil",
                f"{recoil_magnitude:.3f}",
                PALETTE.text_warning if recoil_magnitude > 0.001 else PALETTE.text_muted,
            ),
        ],
    )


def _difficulty_line(telemetry: FZeroXTelemetry) -> PanelLine | None:
    if telemetry.game_mode_name != "gp_race":
        return None
    return panel_line("Difficulty", _format_difficulty(telemetry), PALETTE.text_primary)


def _format_difficulty(telemetry: FZeroXTelemetry) -> str:
    difficulty_name = telemetry.difficulty_name
    if difficulty_name != "unknown":
        return difficulty_name
    return f"unknown ({telemetry.difficulty_raw})"


def _format_camera_setting(telemetry: FZeroXTelemetry) -> str:
    camera_setting_name = telemetry.camera_setting_name
    if camera_setting_name != "unknown":
        return _format_mode_name(camera_setting_name)
    return f"unknown ({telemetry.camera_setting_raw})"


def _format_course_name(info: dict[str, object], telemetry: FZeroXTelemetry) -> str:
    course_name = info.get("track_course_name")
    if isinstance(course_name, str) and course_name:
        return course_name

    display_name = info.get("track_display_name")
    if isinstance(display_name, str) and display_name:
        return _short_track_name(display_name)

    course_id = info.get("track_course_id")
    if isinstance(course_id, str) and course_id:
        return _format_mode_name(course_id)

    course_index = info.get("track_course_index", telemetry.course_index)
    if isinstance(course_index, int | float):
        return f"course {int(course_index)}"
    return "unknown"


def _format_vehicle_setup(info: dict[str, object]) -> str:
    vehicle = info.get("track_vehicle_name", info.get("track_vehicle"))
    engine_setting = info.get("track_engine_setting")
    parts = [
        _format_mode_name(value)
        for value in (vehicle, engine_setting)
        if isinstance(value, str) and value
    ]
    return " / ".join(parts) if parts else "unknown"


def _short_track_name(value: str) -> str:
    suffixes = (
        " Time Attack - Blue Falcon Balanced",
        " time attack blue falcon balanced",
    )
    for suffix in suffixes:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _format_position(telemetry: FZeroXTelemetry) -> str:
    total_racers = telemetry.total_racers
    if total_racers > 0:
        return f"{telemetry.player.position} / {total_racers}"
    return str(telemetry.player.position)


def _format_total_progress(telemetry: FZeroXTelemetry) -> str:
    current = _format_distance(float(telemetry.player.race_distance))
    maximum = _race_progress_limit(telemetry)
    if maximum is None:
        return f"{current} / --"
    return f"{current} / {_format_distance(maximum)}"


def _format_lap_progress(telemetry: FZeroXTelemetry) -> str:
    current = _format_distance(float(telemetry.player.lap_distance))
    maximum = _lap_progress_limit(telemetry)
    if maximum is None:
        return f"{current} / --"
    return f"{current} / {_format_distance(maximum)}"


def _race_progress_limit(telemetry: FZeroXTelemetry) -> float | None:
    course_length = _lap_progress_limit(telemetry)
    total_laps = int(telemetry.total_lap_count)
    if course_length is None or total_laps <= 0:
        return None
    return course_length * total_laps


def _lap_progress_limit(telemetry: FZeroXTelemetry) -> float | None:
    course_length = float(telemetry.course_length)
    if course_length <= 0.0:
        return None
    return course_length
