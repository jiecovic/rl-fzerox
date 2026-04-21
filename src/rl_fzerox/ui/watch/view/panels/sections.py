# src/rl_fzerox/ui/watch/view/panels/sections.py
from __future__ import annotations

from fzerox_emulator import ControllerState, FZeroXTelemetry
from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.masks import ActionMaskBranches
from rl_fzerox.ui.watch.view.panels.format import (
    _float_info,
    _format_control_rate,
    _format_env_step,
    _format_episode_frames,
    _format_game_rate,
    _format_observation_shape,
    _format_policy_action,
    _format_progress_frontier_counter,
    _format_reload_age,
    _format_reload_error,
    _format_render_rate,
    _format_reverse_counter,
    _format_stuck_counter,
    _int_info,
)
from rl_fzerox.ui.watch.view.panels.game import game_section
from rl_fzerox.ui.watch.view.panels.geometry import track_geometry_sections
from rl_fzerox.ui.watch.view.panels.lines import panel_line as _panel_line
from rl_fzerox.ui.watch.view.panels.records import track_record_sections
from rl_fzerox.ui.watch.view.panels.state_vector import policy_state_sections
from rl_fzerox.ui.watch.view.panels.viz import (
    _control_viz_height,
    _flag_viz_height,
    _wrap_text,
)
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import (
    PanelColumns,
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
    policy_label: str | None = None,
    policy_curriculum_stage: str | None,
    policy_action: ActionValue | None,
    policy_reload_age_seconds: float | None,
    policy_reload_error: str | None,
    gas_level: float = 0.0,
    thrust_warning_threshold: float | None = None,
    boost_active: bool = False,
    boost_lamp_level: float = 0.0,
    action_mask_branches: ActionMaskBranches | None = None,
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
                        "Env stage",
                        _format_env_curriculum_stage(info),
                        PALETTE.text_primary
                        if _format_env_curriculum_stage(info) != "-"
                        else PALETTE.text_muted,
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
            *track_record_sections(
                current_info=info,
                track_pool_records=track_pool_records,
                best_finish_times=best_finish_times or {},
            ),
        ],
        middle=[
            game_section(
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
            *track_geometry_sections(telemetry),
        ],
        stats=[
            *policy_state_sections(
                observation_state=observation_state,
                feature_names=observation_state_feature_names,
                zeroed_components=_zeroed_state_components(info),
            ),
        ],
    )


def _zeroed_state_components(info: dict[str, object]) -> frozenset[str]:
    raw_components = info.get("observation_zeroed_state_components")
    if not isinstance(raw_components, tuple | list):
        return frozenset()
    return frozenset(
        component for component in raw_components if isinstance(component, str) and component
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


def _format_env_curriculum_stage(info: dict[str, object]) -> str:
    stage_name = info.get("curriculum_stage_name")
    if isinstance(stage_name, str) and stage_name:
        return stage_name
    stage_index = info.get("curriculum_stage")
    if isinstance(stage_index, int):
        return str(stage_index)
    return "-"


def _format_best_position(value: int | None) -> str:
    return "n/a" if value is None else str(value)
