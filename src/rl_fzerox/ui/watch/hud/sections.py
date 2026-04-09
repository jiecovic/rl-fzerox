# src/rl_fzerox/ui/watch/hud/sections.py
from __future__ import annotations

import numpy as np

from fzerox_emulator import ControllerState, FZeroXTelemetry
from rl_fzerox.ui.watch.hud.format import (
    _float_info,
    _format_control_game_rate,
    _format_distance,
    _format_mode_name,
    _format_observation_shape,
    _format_policy_action,
    _format_race_time_ms,
    _format_reload_age,
    _format_reload_error,
    _format_stuck_counter,
    _int_info,
)
from rl_fzerox.ui.watch.hud.viz import (
    _control_viz,
    _control_viz_height,
    _flag_viz,
    _flag_viz_height,
    _wrap_text,
)
from rl_fzerox.ui.watch.layout import (
    LAYOUT,
    PALETTE,
    Color,
    PanelColumns,
    PanelLine,
    PanelSection,
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
    policy_label: str | None,
    policy_curriculum_stage: str | None,
    policy_action: np.ndarray | None,
    policy_reload_age_seconds: float | None,
    policy_reload_error: str | None,
    action_repeat: int,
    stuck_step_limit: int,
    stuck_min_speed_kph: float,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
    telemetry: FZeroXTelemetry | None,
    observation_state: np.ndarray | None = None,
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
                    _panel_line("Frame", str(info.get("frame_index", 0)), PALETTE.text_primary),
                    _panel_line(
                        "Stuck",
                        _format_stuck_counter(info, stuck_step_limit=stuck_step_limit),
                        PALETTE.text_warning
                        if _int_info(info, "stalled_steps") > 0
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
            PanelSection(
                title="Reset",
                lines=[
                    _panel_line(
                        "Mode",
                        str(reset_info.get("reset_mode", "baseline")),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Baseline",
                        str(reset_info.get("baseline_kind", "unknown")),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Boot",
                        str(reset_info.get("boot_state", "-")),
                        PALETTE.text_primary,
                    ),
                ],
            ),
            PanelSection(
                title="Input",
                lines=[],
                control_viz=_control_viz(control_state),
            ),
        ],
        right=[
            _game_section(info, telemetry, stuck_min_speed_kph=stuck_min_speed_kph),
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
                        "Frame skip",
                        str(action_repeat),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Control/Game",
                        _format_control_game_rate(info),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Milestones",
                        _format_milestone_status(info),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Next milestone",
                        _format_next_milestone(info),
                        PALETTE.text_primary,
                    ),
                ],
            ),
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
                label_surface = fonts.small.render(line.label, True, PALETTE.text_muted)
                value_surface = fonts.body.render(line.value, True, line.color)
                y += max(label_surface.get_height(), value_surface.get_height()) + LAYOUT.line_gap
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


def _format_milestone_status(info: dict[str, object]) -> str:
    completed = _int_info(info, "milestones_completed")
    next_index = _int_info(info, "next_milestone_index")
    if info.get("bootstrap_progress_active"):
        return f"{completed} done / next {next_index} / bootstrap"
    return f"{completed} done / next {next_index}"


def _format_next_milestone(info: dict[str, object]) -> str:
    remaining = info.get("distance_to_next_milestone")
    next_distance = info.get("next_milestone_distance")
    if isinstance(remaining, int | float) and isinstance(next_distance, int | float):
        return f"{remaining:,.1f} to {next_distance:,.0f}"
    return "-"


def _format_reward_value(value: float) -> str:
    return f"{value:.4f}"


def _panel_line(
    label: str,
    value: str,
    color: Color,
    *,
    wrap: bool = False,
    min_value_lines: int = 1,
) -> PanelLine:
    return PanelLine(
        label=label,
        value=value,
        color=color,
        wrap=wrap,
        min_value_lines=min_value_lines,
    )


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

    return PanelSection(
        title="Game",
        lines=[
            _panel_line("Mode", _format_mode_name(telemetry.game_mode_name), PALETTE.text_primary),
            _panel_line("Course", str(telemetry.course_index), PALETTE.text_primary),
            _panel_line(
                "Time",
                _format_race_time_ms(telemetry.player.race_time_ms),
                PALETTE.text_primary,
            ),
            _panel_line("Speed", f"{telemetry.player.speed_kph:.1f} km/h", PALETTE.text_primary),
            _panel_line("Boost", str(telemetry.player.boost_timer), PALETTE.text_primary),
            _panel_line(
                "Energy",
                f"{telemetry.player.energy:.1f} / {telemetry.player.max_energy:.1f}",
                PALETTE.text_primary,
            ),
            _panel_line("Lap", str(telemetry.player.lap), PALETTE.text_primary),
            _panel_line("Pos", str(telemetry.player.position), PALETTE.text_primary),
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
            reverse_detected=telemetry.player.reverse_timer > 0,
            low_speed_detected=telemetry.player.speed_kph < stuck_min_speed_kph,
            energy_depleted=info.get("termination_reason") == "energy_depleted",
        ),
    )


def _policy_state_sections(
    *,
    observation_state: np.ndarray | None,
    feature_names: tuple[str, ...],
) -> list[PanelSection]:
    if observation_state is None:
        return []

    values = np.asarray(observation_state, dtype=np.float32).reshape(-1)
    names = (
        feature_names
        if len(feature_names) == values.size
        else tuple(f"state_{index}" for index in range(values.size))
    )
    return [
        PanelSection(
            title="Obs Vector",
            lines=[
                _panel_line(name, f"{float(value):.3f}", PALETTE.text_primary)
                for name, value in zip(names, values, strict=True)
            ],
        )
    ]
