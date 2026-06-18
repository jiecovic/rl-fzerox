# src/rl_fzerox/ui/watch/view/panels/content/game.py
from __future__ import annotations

import math
from typing import Protocol

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.engine_setting import (
    ENGINE_SLIDER,
    engine_percent_to_slider_step,
    engine_value_to_slider_step,
)
from rl_fzerox.core.envs.course_effects import CourseEffect, course_effect_raw, on_refill_surface
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.core.envs.track_bounds import track_edge_state
from rl_fzerox.core.runtime_spec.vehicle_catalog import (
    VehicleInfo,
    engine_setting_display_name_for_raw,
    vehicle_by_character_index,
)
from rl_fzerox.ui.watch.view.panels.core.format import (
    _float_info,
    _format_distance,
    _format_mode_name,
    _format_race_time_ms,
    _int_info,
)
from rl_fzerox.ui.watch.view.panels.core.lines import panel_line
from rl_fzerox.ui.watch.view.panels.visuals.viz import _flag_viz
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelLine, PanelSection


class _VehicleSetupTelemetry(Protocol):
    @property
    def player(self) -> object: ...


def race_state_section(
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
    *,
    stuck_min_speed_kph: float,
) -> PanelSection:
    if telemetry is None:
        return PanelSection(
            title="Race State",
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
            "KO stars",
            str(max(int(telemetry.player.ko_star_count), 0)),
            PALETTE.text_primary,
        ),
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
        title="Race State",
        lines=lines,
        flag_viz=_flag_viz(
            telemetry.player.state_labels,
            boost_active=telemetry_boost_active(telemetry),
            reverse_detected=telemetry.player.reverse_timer > 0,
            low_speed_detected=telemetry.player.speed_kph < stuck_min_speed_kph,
            energy_depleted=info.get("termination_reason") == "energy_depleted",
            refill_surface_detected=on_refill_surface(telemetry),
            dirt_detected=raw_course_effect == CourseEffect.DIRT,
            ice_detected=raw_course_effect == CourseEffect.ICE,
            track_edge_detected=edge_state.near_side is not None,
            outside_track_bounds=edge_state.outside_bounds,
            energy_loss_detected=_float_info(info, "energy_loss_total") > 0.0,
            damage_taken_detected=_int_info(info, "damage_taken_frames") > 0,
        ),
    )


def race_setup_section(
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> PanelSection:
    if telemetry is None:
        return PanelSection(
            title="Race Setup",
            lines=[
                panel_line("Status", "unavailable", PALETTE.text_warning),
            ],
        )

    return PanelSection(
        title="Race Setup",
        lines=[
            panel_line("Vehicle", _format_vehicle_setup(info, telemetry), PALETTE.text_primary),
            panel_line("Camera", _format_camera_setting(telemetry), PALETTE.text_primary),
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


def _format_vehicle_setup(info: dict[str, object], telemetry: FZeroXTelemetry) -> str:
    if live_setup := _format_telemetry_vehicle_setup(telemetry):
        return live_setup
    if live_setup := _format_live_vehicle_setup(info):
        return live_setup
    return "unknown"


def _format_telemetry_vehicle_setup(telemetry: _VehicleSetupTelemetry) -> str | None:
    character_index = _int_setup_value(getattr(telemetry.player, "machine_character_index", None))
    vehicle = None if character_index is None else vehicle_by_character_index(character_index)
    if vehicle is None:
        return None

    engine_setting = getattr(telemetry.player, "engine_setting", None)
    engine_raw = None if engine_setting is None else engine_value_to_slider_step(engine_setting)
    parts = [vehicle.display_name]
    if engine_raw is not None and 0 <= engine_raw <= ENGINE_SLIDER.max_step:
        parts.append(_format_engine_setting_raw(engine_raw))
    return " / ".join(parts)


def _format_live_vehicle_setup(info: dict[str, object]) -> str | None:
    character_index = _first_known_character_index(
        (
            info.get("racer_character_index"),
            info.get("racer_character_index_ram"),
            info.get("player_character_index"),
            info.get("player_character_index_ram"),
        )
    )
    engine_raw = _live_engine_setting_raw(info)
    if engine_raw is not None and not 0 <= engine_raw <= ENGINE_SLIDER.max_step:
        engine_raw = None
    parts: list[str] = []
    if character_index is not None:
        parts.append(character_index.display_name)
    if engine_raw is not None:
        parts.append(_format_engine_setting_raw(engine_raw))
    return " / ".join(parts) if parts else None


def _format_engine_setting_raw(raw_value: int) -> str:
    return engine_setting_display_name_for_raw(raw_value)


def _live_engine_setting_raw(info: dict[str, object]) -> int | None:
    raw = _int_setup_value(
        info.get(
            "engine_setting_raw_value_ram",
            info.get(
                "engine_setting_raw_value",
                info.get("track_engine_setting_raw_value"),
            ),
        )
    )
    if raw is not None:
        return raw
    percent = info.get("engine_setting_percent_ram")
    if isinstance(percent, bool) or not isinstance(percent, int | float):
        return None
    if not math.isfinite(float(percent)) or not 0.0 <= float(percent) <= 100.0:
        return None
    return engine_percent_to_slider_step(percent)


def _first_known_character_index(values: tuple[object, ...]) -> VehicleInfo | None:
    for value in values:
        character_index = _int_setup_value(value)
        if character_index is None:
            continue
        vehicle = vehicle_by_character_index(character_index)
        if vehicle is not None:
            return vehicle
    return None


def _int_setup_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        if not math.isfinite(float(value)):
            return None
        return int(round(value))
    return None


def _short_track_name(value: str) -> str:
    suffixes = tuple(
        f" {mode} - Blue Falcon {engine}"
        for mode in ("Time Attack", "GP Race")
        for engine in ("Engine 50",)
    ) + tuple(
        f" {mode} - Blue Falcon {engine}".lower()
        for mode in ("Time Attack", "GP Race")
        for engine in ("Engine 50",)
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
