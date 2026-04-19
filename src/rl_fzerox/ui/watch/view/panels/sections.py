# src/rl_fzerox/ui/watch/view/panels/sections.py
from __future__ import annotations

import numpy as np

from fzerox_emulator import ControllerState, FZeroXTelemetry, PlayerTelemetry
from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.course_effects import CourseEffect, course_effect_raw
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.ui.watch.view.panels.format import (
    _float_info,
    _format_control_rate,
    _format_distance,
    _format_env_step,
    _format_episode_frames,
    _format_game_rate,
    _format_mode_name,
    _format_observation_shape,
    _format_policy_action,
    _format_progress_frontier_counter,
    _format_race_time_ms,
    _format_reload_age,
    _format_reload_error,
    _format_render_rate,
    _format_reverse_counter,
    _format_stuck_counter,
    _int_info,
)
from rl_fzerox.ui.watch.view.panels.viz import (
    _control_viz_height,
    _flag_viz,
    _flag_viz_height,
    _wrap_text,
)
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import (
    PanelColumns,
    PanelLine,
    PanelSection,
    StatusIcon,
    ViewerFonts,
)


def _build_panel_columns(
    *,
    episode: int,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode_reward: float,
    paused: bool,
    control_state: ControllerState,
    policy_label: str | None = None,
    policy_curriculum_stage: str | None,
    policy_action: ActionValue | None,
    policy_reload_age_seconds: float | None,
    policy_reload_error: str | None,
    gas_level: float = 0.0,
    thrust_warning_threshold: float | None = None,
    boost_active: bool = False,
    boost_lamp_level: float = 0.0,
    best_finish_position: int | None = None,
    best_finish_times: dict[str, int] | None = None,
    track_pool_records: tuple[dict[str, object], ...] = (),
    continuous_drive_deadzone: float = 0.2,
    continuous_air_brake_mode: str = "always",
    continuous_air_brake_disabled: bool = False,
    action_repeat: int,
    stuck_step_limit: int | None,
    stuck_min_speed_kph: float,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
    telemetry: FZeroXTelemetry | None,
    policy_deterministic: bool | None = None,
    max_episode_steps: int = 50_000,
    wrong_way_timer_limit: int | None = 300,
    progress_frontier_stall_limit_frames: int | None = 900,
    observation_state: StateVector | None = None,
    observation_state_feature_names: tuple[str, ...] = (),
) -> PanelColumns:
    return PanelColumns(
        left=[
            PanelSection(
                title="Session",
                lines=[
                    _panel_line(
                        "State",
                        "paused" if paused else "running",
                        PALETTE.text_warning if paused else PALETTE.text_accent,
                    ),
                    _panel_line(
                        "Policy",
                        policy_label if policy_label is not None else "manual",
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Checkpoint stage",
                        policy_curriculum_stage if policy_curriculum_stage is not None else "-",
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Deterministic",
                        _format_policy_deterministic(policy_deterministic),
                        PALETTE.text_primary
                        if policy_deterministic is not None
                        else PALETTE.text_muted,
                    ),
                    _panel_line(
                        "Action",
                        _format_policy_action(policy_action),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Reload",
                        _format_reload_age(policy_reload_age_seconds),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Last reload err",
                        _format_reload_error(policy_reload_error),
                        PALETTE.text_warning
                        if policy_reload_error is not None
                        else PALETTE.text_muted,
                        wrap=True,
                        min_value_lines=2,
                    ),
                    _panel_line("Episode", str(episode), PALETTE.text_primary),
                    _panel_line(
                        "Best position",
                        _format_best_position(best_finish_position),
                        PALETTE.text_primary
                        if best_finish_position is not None
                        else PALETTE.text_muted,
                    ),
                    _panel_line(
                        "Episode frame",
                        _format_episode_frames(info, max_episode_steps=max_episode_steps),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Env step",
                        _format_env_step(
                            info,
                            action_repeat=action_repeat,
                            max_episode_steps=max_episode_steps,
                        ),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Stuck frames",
                        _format_stuck_counter(info, stuck_step_limit=stuck_step_limit),
                        PALETTE.text_warning
                        if _int_info(info, "stalled_steps") > 0
                        else PALETTE.text_muted,
                    ),
                    _panel_line(
                        "Reverse frames",
                        _format_reverse_counter(
                            info,
                            wrong_way_timer_limit=wrong_way_timer_limit,
                        ),
                        PALETTE.text_warning
                        if _int_info(info, "reverse_timer") > 0
                        else PALETTE.text_muted,
                    ),
                    _panel_line(
                        "Frontier frames",
                        _format_progress_frontier_counter(
                            info,
                            progress_frontier_stall_limit_frames=(
                                progress_frontier_stall_limit_frames
                            ),
                        ),
                        PALETTE.text_warning
                        if _int_info(info, "progress_frontier_stalled_frames") > 0
                        else PALETTE.text_muted,
                    ),
                    _panel_line(
                        "Step",
                        _format_reward_value(_float_info(info, "step_reward")),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Return",
                        _format_reward_value(episode_reward),
                        PALETTE.text_primary,
                    ),
                ],
            ),
            *_track_record_sections(
                current_info=info,
                track_pool_records=track_pool_records,
                best_finish_times=best_finish_times or {},
            ),
        ],
        middle=[
            _game_section(
                info,
                telemetry,
                stuck_min_speed_kph=stuck_min_speed_kph,
            ),
            PanelSection(
                title="Display",
                lines=[
                    _panel_line(
                        "Game",
                        f"{game_display_size[0]}x{game_display_size[1]}",
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Obs",
                        _format_observation_shape(observation_shape),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Action repeat",
                        str(action_repeat),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Control FPS",
                        _format_control_rate(info),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Game FPS",
                        _format_game_rate(info),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Render FPS",
                        _format_render_rate(info),
                        PALETTE.text_primary,
                    ),
                ],
            ),
            *_track_geometry_sections(telemetry),
        ],
        stats=[
            *_policy_state_sections(
                observation_state=observation_state,
                feature_names=observation_state_feature_names,
            ),
        ],
    )


def _column_content_height(
    fonts: ViewerFonts,
    sections: list[PanelSection],
    *,
    width: int,
) -> int:
    y = 0
    for section_index, section in enumerate(sections):
        section_title = fonts.section.render(section.title, True, PALETTE.text_primary)
        y += section_title.get_height() + LAYOUT.section_title_gap
        y += LAYOUT.section_rule_gap
        for line in section.lines:
            if line.divider:
                y += LAYOUT.line_gap + 1
                continue
            if line.label and line.wrap:
                label_surface = fonts.small.render(line.label, True, PALETTE.text_muted)
                y += label_surface.get_height() + LAYOUT.line_gap
                wrapped_lines = _wrap_text(
                    fonts.small,
                    line.value,
                    width - LAYOUT.wrapped_value_indent,
                )
                for wrapped_line in wrapped_lines:
                    value_surface = fonts.small.render(wrapped_line, True, line.color)
                    y += value_surface.get_height() + LAYOUT.line_gap
                if len(wrapped_lines) < line.min_value_lines:
                    blank_height = fonts.small.render("Ag", True, PALETTE.text_primary).get_height()
                    y += (line.min_value_lines - len(wrapped_lines)) * (
                        blank_height + LAYOUT.line_gap
                    )
                continue
            if line.label:
                label_font = fonts.record_header if line.heading else fonts.small
                value_font = fonts.small if line.heading else fonts.body
                label_surface = label_font.render(line.label, True, PALETTE.text_muted)
                value_surface = value_font.render(line.value, True, line.color)
                status_text_surface = fonts.small.render(line.status_text, True, line.color)
                value_width = (
                    value_surface.get_width()
                    if line.status_icon is None
                    else value_surface.get_height() + status_text_surface.get_width()
                )
                inline_value_space = width - label_surface.get_width() - LAYOUT.inline_value_gap
                if value_width <= inline_value_space:
                    y += (
                        max(label_surface.get_height(), value_surface.get_height())
                        + LAYOUT.line_gap
                    )
                else:
                    y += label_surface.get_height() + LAYOUT.line_gap
                    y += value_surface.get_height() + LAYOUT.line_gap
            else:
                value_surface = fonts.small.render(line.value, True, line.color)
                y += value_surface.get_height() + LAYOUT.line_gap
        if section.control_viz is not None:
            y += LAYOUT.control_viz_gap + _control_viz_height(fonts)
        if section.flag_viz is not None:
            y += LAYOUT.control_viz_gap + _flag_viz_height(fonts, section.flag_viz)
        if section_index < len(sections) - 1:
            y += LAYOUT.section_gap
    return y


def _format_reward_value(value: float) -> str:
    return f"{value:.4f}"


def _format_policy_deterministic(value: bool | None) -> str:
    if value is None:
        return "-"
    return "true" if value else "false"


def _format_best_position(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _panel_line(
    label: str,
    value: str,
    color: Color,
    *,
    wrap: bool = False,
    min_value_lines: int = 1,
    divider: bool = False,
    heading: bool = False,
    status_icon: StatusIcon | None = None,
    status_text: str = "",
) -> PanelLine:
    return PanelLine(
        label=label,
        value=value,
        color=color,
        wrap=wrap,
        min_value_lines=min_value_lines,
        divider=divider,
        heading=heading,
        status_icon=status_icon,
        status_text=status_text,
    )


def _panel_divider() -> PanelLine:
    return _panel_line("", "", PALETTE.panel_border, divider=True)


def _panel_heading(label: str) -> PanelLine:
    return _panel_line(label, "", PALETTE.text_primary, heading=True)


def _game_section(
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
    *,
    stuck_min_speed_kph: float,
) -> PanelSection:
    if telemetry is None:
        return PanelSection(
            title="Game",
            lines=[
                _panel_line("Status", "unavailable", PALETTE.text_warning),
            ],
        )

    recoil_magnitude = (
        telemetry.player.recoil_tilt_magnitude if telemetry.player.collision_recoil else 0.0
    )
    raw_course_effect = course_effect_raw(telemetry)

    return PanelSection(
        title="Game",
        lines=[
            _panel_line("Mode", _format_mode_name(telemetry.game_mode_name), PALETTE.text_primary),
            _panel_line("Difficulty", _format_difficulty(telemetry), PALETTE.text_primary),
            _panel_line("Camera", _format_camera_setting(telemetry), PALETTE.text_primary),
            _panel_line("Course", _format_course_name(info, telemetry), PALETTE.text_primary),
            _panel_line("Vehicle", _format_vehicle_setup(info), PALETTE.text_primary),
            _panel_line(
                "Time",
                _format_race_time_ms(telemetry.player.race_time_ms),
                PALETTE.text_primary,
            ),
            _panel_line("Speed", f"{telemetry.player.speed_kph:.1f} km/h", PALETTE.text_primary),
            _panel_line(
                "Recoil",
                f"{recoil_magnitude:.3f}",
                PALETTE.text_warning if recoil_magnitude > 0.001 else PALETTE.text_muted,
            ),
            _panel_line(
                "Energy",
                f"{telemetry.player.energy:.1f} / {telemetry.player.max_energy:.1f}",
                PALETTE.text_primary,
            ),
            _panel_line("Lap", str(telemetry.player.lap), PALETTE.text_primary),
            _panel_line("Position", _format_position(telemetry), PALETTE.text_primary),
            _panel_line(
                "Progress",
                _format_distance(telemetry.player.race_distance),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Lap prog",
                _format_distance(telemetry.player.lap_distance),
                PALETTE.text_primary,
            ),
        ],
        flag_viz=_flag_viz(
            telemetry.player.state_labels,
            boost_active=telemetry_boost_active(telemetry),
            reverse_detected=telemetry.player.reverse_timer > 0,
            low_speed_detected=telemetry.player.speed_kph < stuck_min_speed_kph,
            energy_depleted=info.get("termination_reason") == "energy_depleted",
            energy_refill_detected=telemetry.player.on_energy_refill,
            dirt_detected=raw_course_effect == CourseEffect.DIRT,
            ice_detected=raw_course_effect == CourseEffect.ICE,
            energy_loss_detected=_float_info(info, "energy_loss_total") > 0.0,
            damage_taken_detected=_int_info(info, "damage_taken_frames") > 0,
        ),
    )


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


def _track_record_sections(
    *,
    current_info: dict[str, object],
    track_pool_records: tuple[dict[str, object], ...],
    best_finish_times: dict[str, int],
) -> tuple[PanelSection, ...]:
    records = track_pool_records or _current_track_record_pool(current_info)
    if not records and not best_finish_times:
        return ()
    lines: list[PanelLine] = []
    for index, record in enumerate(records):
        if index > 0:
            lines.append(_panel_divider())
        lines.extend(_track_record_pool_lines(record, best_finish_times=best_finish_times))
    if not lines:
        return ()
    return (
        PanelSection(
            title="Records",
            lines=lines,
        ),
    )


def _current_track_record_pool(info: dict[str, object]) -> tuple[dict[str, object], ...]:
    if (
        _track_best_key(info) is None
        and _optional_int_info(info, "track_non_agg_best_time_ms") is None
    ):
        return ()
    return (dict(info),)


def _track_record_pool_lines(
    record: dict[str, object],
    *,
    best_finish_times: dict[str, int],
) -> tuple[PanelLine, ...]:
    watch_best = _watch_best_time_ms(record, best_finish_times)
    best_time = _optional_int_info(record, "track_non_agg_best_time_ms")
    worst_time = _optional_int_info(record, "track_non_agg_worst_time_ms")
    status_icon, status_color = _track_record_status(
        watch_best_ms=watch_best,
        best_time_ms=best_time,
        worst_time_ms=worst_time,
    )
    status_text = _track_record_gap_text(
        watch_best_ms=watch_best,
        best_time_ms=best_time,
        worst_time_ms=worst_time,
    )
    return (
        _panel_line(
            _format_track_record_label(record),
            "",
            status_color,
            heading=True,
            status_icon=status_icon,
            status_text=status_text,
        ),
        _panel_line(
            "PB",
            _format_optional_compact_time(watch_best),
            status_color if watch_best is not None else PALETTE.text_muted,
        ),
        _panel_line(
            "WR",
            _format_record_range(best_time, worst_time),
            PALETTE.text_primary
            if best_time is not None or worst_time is not None
            else PALETTE.text_muted,
        ),
    )


def _format_track_record_label(record: dict[str, object]) -> str:
    course_name = record.get("track_course_name")
    if isinstance(course_name, str) and course_name:
        return course_name

    display_name = record.get("track_display_name")
    if isinstance(display_name, str) and display_name:
        return _short_track_name(display_name)

    course_id = record.get("track_course_id")
    if isinstance(course_id, str) and course_id:
        return _format_mode_name(course_id)

    course_index = record.get("track_course_index", record.get("course_index"))
    if isinstance(course_index, bool):
        return "Track"
    if isinstance(course_index, int):
        return f"course {course_index}"
    return "Track"


def _short_track_name(value: str) -> str:
    suffixes = (
        " Time Attack - Blue Falcon Balanced",
        " time attack blue falcon balanced",
    )
    for suffix in suffixes:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _track_record_status(
    *,
    watch_best_ms: int | None,
    best_time_ms: int | None,
    worst_time_ms: int | None,
) -> tuple[StatusIcon, Color]:
    if watch_best_ms is None:
        return "none", PALETTE.text_muted

    range_cutoff_ms = worst_time_ms if worst_time_ms is not None else best_time_ms
    if range_cutoff_ms is None:
        return "outside", PALETTE.text_warning
    if watch_best_ms <= range_cutoff_ms:
        return "in_range", PALETTE.text_accent
    return "outside", PALETTE.text_warning


def _track_record_gap_text(
    *,
    watch_best_ms: int | None,
    best_time_ms: int | None,
    worst_time_ms: int | None,
) -> str:
    if watch_best_ms is None:
        return ""
    range_cutoff_ms = worst_time_ms if worst_time_ms is not None else best_time_ms
    if range_cutoff_ms is not None and watch_best_ms > range_cutoff_ms:
        return f"+{_format_compact_duration_ms(watch_best_ms - range_cutoff_ms)}"
    if best_time_ms is None:
        return ""
    if watch_best_ms > best_time_ms:
        return f"+{_format_compact_duration_ms(watch_best_ms - best_time_ms)}"
    return f"-{_format_compact_duration_ms(best_time_ms - watch_best_ms)}"


def _format_optional_compact_time(time_ms: int | None) -> str:
    if time_ms is None:
        return "--"
    return _format_compact_race_time_ms(time_ms)


def _format_record_range(best_time_ms: int | None, worst_time_ms: int | None) -> str:
    best = _format_optional_compact_time(best_time_ms)
    worst = _format_optional_compact_time(worst_time_ms)
    return f"{best} - {worst}"


def _track_geometry_sections(
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
                _panel_line(
                    "Segment",
                    _format_optional_int(player.segment_index),
                    PALETTE.text_primary
                    if player.segment_index is not None
                    else PALETTE.text_muted,
                ),
                _panel_line("Spline t", f"{player.segment_t:.3f}", PALETTE.text_primary),
                _panel_line(
                    "Spline len",
                    f"{player.segment_length_proportion:.3f}",
                    PALETTE.text_primary,
                ),
                _panel_line(
                    "Center dist",
                    f"{player.lateral_distance:.1f}",
                    _center_distance_warning_color(player),
                ),
                _panel_line(
                    "Lat vel",
                    f"{player.local_lateral_velocity:+.2f}",
                    PALETTE.text_warning if sliding else PALETTE.text_primary,
                ),
                _panel_line(
                    "Sliding",
                    "yes" if sliding else "no",
                    PALETTE.text_warning if sliding else PALETTE.text_muted,
                ),
                _panel_line(
                    "Lat offset",
                    f"{player.signed_lateral_offset:+.1f}",
                    edge_color,
                ),
                _panel_line(
                    "Edge ratio",
                    _format_edge_ratio(edge_side, edge_ratio),
                    edge_color,
                ),
                _panel_line(
                    "Near edge",
                    near_edge if near_edge is not None else "no",
                    PALETTE.text_warning if near_edge is not None else PALETTE.text_muted,
                ),
                _panel_line(
                    "Disp mag",
                    f"{player.lateral_displacement_magnitude:.1f}",
                    PALETTE.text_muted,
                ),
                _panel_line(
                    "Radius L/R",
                    f"{player.current_radius_left:.0f} / {player.current_radius_right:.0f}",
                    PALETTE.text_primary,
                ),
                _panel_line(
                    "Ground height",
                    f"{player.height_above_ground:.1f}",
                    PALETTE.text_primary,
                ),
                _panel_line(
                    "Vel / acc",
                    f"{player.velocity_magnitude:.2f} / {player.acceleration_magnitude:.2f}",
                    PALETTE.text_primary,
                ),
                _panel_line(
                    "Force / slide",
                    f"{player.acceleration_force:.2f} / {player.drift_attack_force:.2f}",
                    PALETTE.text_primary,
                ),
                _panel_line(
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
    return (
        not player.airborne
        and abs(player.local_lateral_velocity) > lateral_velocity_threshold
    )


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


def _format_compact_race_time_ms(race_time_ms: int) -> str:
    minutes, remainder = divmod(max(0, race_time_ms), 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    if minutes == 0:
        return f"{seconds}.{milliseconds:03d}"
    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


def _format_compact_duration_ms(duration_ms: int) -> str:
    tenths = round(max(0, duration_ms) / 100)
    if tenths < 600:
        return f"{tenths / 10:.1f}s"
    minutes, remaining_tenths = divmod(tenths, 600)
    return f"{minutes}min {remaining_tenths / 10:.1f}s"


def _watch_best_time_ms(
    info: dict[str, object],
    best_finish_times: dict[str, int],
) -> int | None:
    track_key = _track_best_key(info)
    if track_key is None:
        return None
    return best_finish_times.get(track_key)


def _track_best_key(info: dict[str, object]) -> str | None:
    for key in ("track_id", "track_display_name"):
        value = info.get(key)
        if isinstance(value, str) and value:
            return value
    value = info.get("track_course_index", info.get("course_index"))
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"course:{value}"
    return None


def _optional_int_info(info: dict[str, object], key: str) -> int | None:
    value = info.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _format_position(telemetry: FZeroXTelemetry) -> str:
    total_racers = telemetry.total_racers
    if total_racers > 0:
        return f"{telemetry.player.position} / {total_racers}"
    return str(telemetry.player.position)


def _policy_state_sections(
    *,
    observation_state: StateVector | None,
    feature_names: tuple[str, ...],
) -> list[PanelSection]:
    if observation_state is None:
        return []

    values: StateVector = np.asarray(observation_state, dtype=np.float32).reshape(-1)
    names = (
        feature_names
        if len(feature_names) == values.size
        else tuple(f"state_{index}" for index in range(values.size))
    )
    section_lines: list[PanelLine] = []
    for group_title, group_prefix in _state_vector_groups(names):
        group_lines = _state_vector_group_lines(
            names=names,
            values=values,
            group_prefix=group_prefix,
        )
        if group_lines:
            if section_lines:
                section_lines.append(_panel_divider())
            section_lines.append(_panel_heading(group_title))
            section_lines.extend(group_lines)
    if not section_lines:
        return []
    return [PanelSection(title="State Vector", lines=section_lines)]


def _state_vector_groups(names: tuple[str, ...]) -> tuple[tuple[str, str | None], ...]:
    component_groups = (
        ("Vehicle", "vehicle_state."),
        ("Track Position", "track_position."),
        ("Surface", "surface_state."),
        ("Course", "course_context."),
    )
    used_component_names = {
        name
        for _, prefix in component_groups
        for name in names
        if prefix is not None and name.startswith(prefix)
    }
    groups: tuple[tuple[str, str | None], ...] = tuple(
        (title, prefix)
        for title, prefix in component_groups
        if any(name.startswith(prefix) for name in names)
    )
    if any(name.startswith("control_history.") or name.startswith("prev_") for name in names):
        groups = (*groups, ("Control History", "control_history."))
    legacy_names = tuple(
        name
        for name in names
        if name not in used_component_names and not name.startswith("control_history.")
        and not name.startswith("prev_")
    )
    if legacy_names:
        return (
            *groups,
            ("Legacy", None),
        )
    return groups


def _state_vector_group_lines(
    *,
    names: tuple[str, ...],
    values: StateVector,
    group_prefix: str | None,
) -> list[PanelLine]:
    if group_prefix == "course_context.":
        return _course_context_state_lines(names=names, values=values)
    if group_prefix == "control_history.":
        return _control_history_state_lines(names=names, values=values)
    return [
        _panel_line(
            _state_vector_label(name, group_prefix=group_prefix),
            f"{float(value):.3f}",
            PALETTE.text_primary,
        )
        for name, value in zip(names, values, strict=True)
        if _state_vector_name_matches_group(name, group_prefix)
    ]


def _control_history_state_lines(
    *,
    names: tuple[str, ...],
    values: StateVector,
) -> list[PanelLine]:
    return [
        _panel_line(
            _state_vector_label(name, group_prefix="control_history."),
            f"{float(value):.3f}",
            PALETTE.text_primary,
        )
        for name, value in zip(names, values, strict=True)
        if name.startswith("control_history.") or name.startswith("prev_")
    ]


def _course_context_state_lines(
    *,
    names: tuple[str, ...],
    values: StateVector,
) -> list[PanelLine]:
    course_bits = [
        float(value)
        for name, value in zip(names, values, strict=True)
        if name.startswith("course_context.course_builtin_")
    ]
    if not course_bits:
        return []

    active_index = _one_hot_active_index(course_bits)
    return [
        _panel_line(
            "categorical",
            "--" if active_index is None else str(active_index),
            PALETTE.text_primary if active_index is not None else PALETTE.text_muted,
        ),
        _panel_line(
            "one hot",
            "".join("1" if value >= 0.5 else "0" for value in course_bits),
            PALETTE.text_primary,
        ),
    ]


def _one_hot_active_index(values: list[float]) -> int | None:
    active_indices = [index for index, value in enumerate(values) if value >= 0.5]
    if len(active_indices) == 1:
        return active_indices[0]
    return None


def _state_vector_name_matches_group(name: str, group_prefix: str | None) -> bool:
    if group_prefix is None:
        return "." not in name and not name.startswith("prev_")
    return name.startswith(group_prefix)


def _state_vector_label(name: str, *, group_prefix: str | None) -> str:
    if group_prefix is None:
        return name
    if group_prefix == "control_history." and name.startswith("prev_"):
        return name
    return name.removeprefix(group_prefix)
