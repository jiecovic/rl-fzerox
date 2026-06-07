# src/rl_fzerox/ui/watch/view/panels/content/sections.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, RaceControlState
from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches
from rl_fzerox.core.runtime_spec.schema import PolicyConfig, TrainConfig
from rl_fzerox.ui.watch.view.auxiliary_metrics import AuxiliaryEpisodeMetricsSnapshot
from rl_fzerox.ui.watch.view.panels.content.auxiliary import (
    auxiliary_episode_sections,
)
from rl_fzerox.ui.watch.view.panels.content.game import (
    race_setup_section,
    race_state_section,
)
from rl_fzerox.ui.watch.view.panels.content.geometry import track_geometry_sections
from rl_fzerox.ui.watch.view.panels.content.hparams import training_hparam_sections
from rl_fzerox.ui.watch.view.panels.content.records import track_record_sections
from rl_fzerox.ui.watch.view.panels.content.state_vector import policy_state_sections
from rl_fzerox.ui.watch.view.panels.core.format import (
    _float_info,
    _format_checkpoint_experience,
    _format_env_step,
    _format_episode_frames,
    _format_game_speed,
    _format_height_width,
    _format_mode_name,
    _format_observation_shape,
    _format_policy_action,
    _format_progress_frontier_counter,
    _format_reload_age,
    _int_info,
)
from rl_fzerox.ui.watch.view.panels.core.lines import panel_line as _panel_line
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import (
    PanelColumns,
    PanelSection,
)


def _build_panel_columns(
    *,
    episode: int,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode_reward: float,
    paused: bool,
    control_state: RaceControlState,
    policy_label: str | None = None,
    policy_curriculum_stage: str | None,
    policy_action: ActionValue | None,
    policy_reload_age_seconds: float | None,
    policy_reload_error: str | None,
    policy_num_timesteps: int | None = None,
    policy_experience_frames: int | None = None,
    gas_level: float = 0.0,
    thrust_warning_threshold: float | None = None,
    boost_active: bool = False,
    boost_lamp_level: float = 0.0,
    action_mask_branches: ActionMaskBranches | None = None,
    best_finish_position: int | None = None,
    best_finish_ranks: dict[str, int] | None = None,
    best_finish_times: dict[str, int] | None = None,
    latest_finish_times: dict[str, int] | None = None,
    latest_finish_deltas_ms: dict[str, int] | None = None,
    failed_track_attempts: frozenset[str] = frozenset(),
    track_pool_records: tuple[dict[str, object], ...] = (),
    continuous_drive_deadzone: float = 0.2,
    continuous_air_brake_axis_index: int | None = 2,
    continuous_air_brake_deadzone: float = 0.05,
    continuous_air_brake_full_threshold: float = 0.85,
    continuous_air_brake_min_duty: float = 0.0,
    continuous_air_brake_mode: str = "always",
    continuous_air_brake_disabled: bool = False,
    action_repeat: int,
    stuck_min_speed_kph: float,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...] | None,
    telemetry: FZeroXTelemetry | None,
    policy_deterministic: bool | None = None,
    manual_control_enabled: bool = False,
    max_episode_steps: int = 50_000,
    progress_frontier_stall_limit_frames: int | None = 900,
    observation_state: StateVector | None = None,
    observation_state_reference: StateVector | None = None,
    observation_state_feature_names: tuple[str, ...] = (),
    policy_auxiliary_state_predictions: dict[str, object] | None = None,
    policy_auxiliary_state_targets: dict[str, object] | None = None,
    auxiliary_episode_metrics: AuxiliaryEpisodeMetricsSnapshot | None = None,
    emulator_renderer: str = "unknown",
    watch_device: str = "unknown",
    train_config: TrainConfig | None = None,
    policy_config: PolicyConfig | None = None,
) -> PanelColumns:
    curriculum_stage = _format_curriculum_stage(
        checkpoint_stage=policy_curriculum_stage,
        info=info,
    )
    return PanelColumns(
        left=[
            PanelSection(
                title="Run State",
                lines=[
                    _panel_line(
                        "State",
                        "paused" if paused else "running",
                        PALETTE.text_warning if paused else PALETTE.text_accent,
                    ),
                    _panel_line(
                        "Driver",
                        _format_driver_mode(
                            policy_label=policy_label,
                            manual_control_enabled=manual_control_enabled,
                        ),
                        _driver_mode_color(
                            policy_label=policy_label,
                            manual_control_enabled=manual_control_enabled,
                        ),
                    ),
                    _panel_line(
                        "Stage",
                        curriculum_stage,
                        PALETTE.text_primary if curriculum_stage != "-" else PALETTE.text_muted,
                    ),
                    _panel_line(
                        "Experience",
                        _format_checkpoint_experience(policy_experience_frames),
                        PALETTE.text_primary
                        if policy_experience_frames is not None
                        else PALETTE.text_muted,
                    ),
                    _panel_line(
                        "Reload",
                        _format_reload_age(policy_reload_age_seconds),
                        PALETTE.text_primary,
                    ),
                ],
            ),
            PanelSection(
                title="Episode Progress",
                lines=[
                    _panel_line("Episode", str(episode), PALETTE.text_primary),
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
                        "Step reward",
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
                title="Policy Output",
                lines=[
                    _panel_line(
                        "Mode",
                        _format_policy_deterministic(policy_deterministic),
                        PALETTE.text_primary
                        if policy_deterministic is not None
                        else PALETTE.text_muted,
                    ),
                    _panel_line("Device", watch_device, PALETTE.text_primary),
                    _panel_line(
                        "Action",
                        _format_policy_action(policy_action),
                        PALETTE.text_primary,
                    ),
                ],
            ),
            race_state_section(
                info,
                telemetry,
                stuck_min_speed_kph=stuck_min_speed_kph,
            ),
            race_setup_section(info, telemetry),
            PanelSection(
                title="Runtime",
                lines=[
                    _panel_line("Renderer", emulator_renderer, PALETTE.text_primary),
                    _panel_line(
                        "Repeat",
                        f"x{action_repeat}",
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Control FPS",
                        _format_fps_value(info, "control_fps"),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Game FPS",
                        _format_fps_value(info, "game_fps"),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Render FPS",
                        _format_fps_value(info, "render_fps"),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Speed multiplier",
                        _format_game_speed(info, action_repeat=action_repeat),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Game size",
                        _format_height_width(game_display_size[1], game_display_size[0]),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Obs size",
                        "-"
                        if observation_shape is None
                        else _format_observation_shape(observation_shape),
                        PALETTE.text_muted if observation_shape is None else PALETTE.text_primary,
                    ),
                ],
            ),
        ],
        middle=[
            *track_geometry_sections(telemetry),
        ],
        stats=[
            *policy_state_sections(
                observation_state=observation_state,
                observation_state_reference=observation_state_reference,
                feature_names=observation_state_feature_names,
                policy_config=policy_config,
                auxiliary_predictions=policy_auxiliary_state_predictions,
                auxiliary_targets=policy_auxiliary_state_targets,
                zeroed_features=_zeroed_state_features(info),
                watch_zeroed_features=_watch_zeroed_state_features(info),
            ),
        ],
        aux=auxiliary_episode_sections(auxiliary_episode_metrics),
        records=[
            *track_record_sections(
                current_info=info,
                track_pool_records=track_pool_records,
                best_finish_ranks=best_finish_ranks or {},
                best_finish_times=best_finish_times or {},
                latest_finish_times=latest_finish_times or {},
                latest_finish_deltas_ms=latest_finish_deltas_ms or {},
                failed_track_attempts=failed_track_attempts,
            ),
        ],
        career=_career_mode_sections(info),
        train=training_hparam_sections(
            train_config=train_config,
            policy_config=policy_config,
        ),
    )


def _career_mode_sections(info: dict[str, object]) -> list[PanelSection]:
    target = _non_empty_text(info.get("career_mode_target_label"))
    phase = _non_empty_text(info.get("career_mode_phase"))
    save_attempt_id = _non_empty_text(info.get("career_mode_attempt_id"))
    completed_targets = _int_info(info, "career_mode_completed_targets")
    total_targets = _int_info(info, "career_mode_total_targets")
    inspection_status = _non_empty_text(info.get("career_mode_inspection_status"))
    policy_run_name = _non_empty_text(info.get("career_mode_policy_run_name"))
    policy_run_id = _non_empty_text(info.get("career_mode_policy_run_id"))
    policy_artifact = _non_empty_text(info.get("career_mode_policy_artifact"))
    policy_course_id = _non_empty_text(info.get("career_mode_policy_course_id"))
    policy_active = info.get("career_mode_policy_active") is True
    progress_lines = [
        _panel_line(
            "Progress",
            _format_career_progress(completed_targets, total_targets),
            PALETTE.text_primary if total_targets else PALETTE.text_muted,
        ),
        _panel_line(
            "Save state",
            _format_mode_name(inspection_status) if inspection_status else "-",
            PALETTE.text_primary if inspection_status else PALETTE.text_muted,
        ),
        _panel_line(
            "Current target",
            target or "-",
            PALETTE.text_primary if target else PALETTE.text_muted,
            wrap=True,
            min_value_lines=2,
        ),
    ]
    controller_lines = [
        _panel_line(
            "Phase",
            _format_mode_name(phase) if phase else "-",
            PALETTE.text_primary if phase else PALETTE.text_muted,
        ),
        _panel_line(
            "Last input",
            _career_last_input(info),
            PALETTE.text_primary,
            wrap=True,
            min_value_lines=2,
        ),
        _panel_line(
            "Game facts",
            _career_game_facts(info),
            PALETTE.text_primary,
            wrap=True,
            min_value_lines=2,
        ),
        _panel_line(
            "Race boundary",
            _career_boundary_facts(info),
            PALETTE.text_primary,
            wrap=True,
            min_value_lines=2,
        ),
        _panel_line(
            "Race progress",
            _career_race_progress_facts(info),
            PALETTE.text_primary,
            wrap=True,
            min_value_lines=2,
        ),
        _panel_line(
            "Camera",
            _career_camera_facts(info),
            PALETTE.text_primary,
            wrap=True,
            min_value_lines=2,
        ),
    ]
    policy_lines = [
        _panel_line(
            "Policy control",
            "active" if policy_active else "inactive",
            PALETTE.text_accent if policy_active else PALETTE.text_muted,
        ),
        _panel_line(
            "Policy",
            policy_run_name or "-",
            PALETTE.text_primary if policy_run_name else PALETTE.text_muted,
            wrap=True,
            min_value_lines=2,
        ),
        _panel_line(
            "Artifact",
            policy_artifact or "-",
            PALETTE.text_primary if policy_artifact else PALETTE.text_muted,
        ),
        _panel_line(
            "Course",
            policy_course_id or "-",
            PALETTE.text_primary if policy_course_id else PALETTE.text_muted,
        ),
        _panel_line(
            "Attempt",
            save_attempt_id or "-",
            PALETTE.text_primary if save_attempt_id else PALETTE.text_muted,
            wrap=True,
            min_value_lines=2,
        ),
        _panel_line(
            "Policy run id",
            policy_run_id or "-",
            PALETTE.text_primary if policy_run_id else PALETTE.text_muted,
            wrap=True,
            min_value_lines=2,
        ),
    ]
    return [
        PanelSection(
            title="Career Progress",
            lines=progress_lines,
        ),
        PanelSection(
            title="Career Controller",
            lines=controller_lines,
        ),
        PanelSection(
            title="Career Policy",
            lines=policy_lines,
        ),
    ]


def _format_career_progress(completed: int | None, total: int | None) -> str:
    if completed is None or total is None:
        return "-"
    return f"{completed} / {total} targets"


def _career_game_facts(info: dict[str, object]) -> str:
    fields = [
        ("screen", _non_empty_text(info.get("career_mode_fsm_observed_screen"))),
        ("mode", _non_empty_text(info.get("career_mode_fsm_game_mode"))),
        ("course", _format_optional_int(info, "career_mode_fsm_course_index")),
        ("selected", _format_optional_int(info, "career_mode_fsm_selected_mode_raw")),
        ("diff_state", _format_optional_int(info, "career_mode_fsm_difficulty_state_raw")),
        ("diff_cursor", _format_optional_int(info, "career_mode_fsm_difficulty_cursor_raw")),
        ("transition", _format_optional_int(info, "career_mode_fsm_transition_raw")),
        ("popup", _non_empty_text(info.get("career_mode_fsm_popup_state"))),
    ]
    return _join_named_values(fields)


def _career_last_input(info: dict[str, object]) -> str:
    fields = [
        ("input", _non_empty_text(info.get("career_mode_last_input"))),
        ("step", _non_empty_text(info.get("career_mode_last_step"))),
        ("frames", _format_optional_int(info, "career_mode_last_step_frames")),
    ]
    return _join_named_values(fields)


def _career_boundary_facts(info: dict[str, object]) -> str:
    fields = [
        ("terminal", _format_optional_bool(info, "career_mode_fsm_terminal_result")),
        ("result", _format_optional_bool(info, "career_mode_fsm_completed_result_screen")),
        ("fresh", _format_optional_bool(info, "career_mode_fsm_fresh_race_ready")),
        ("reason", _non_empty_text(info.get("career_mode_fsm_terminal_reason"))),
    ]
    return _join_named_values(fields)


def _career_race_progress_facts(info: dict[str, object]) -> str:
    completed_laps = _format_optional_int(info, "career_mode_fsm_completed_laps")
    total_laps = _format_optional_int(info, "career_mode_fsm_total_laps")
    lap_text = None
    if completed_laps is not None or total_laps is not None:
        lap_text = f"{completed_laps or '-'} / {total_laps or '-'}"
    fields = [
        ("laps", lap_text),
        ("intro", _format_optional_int(info, "career_mode_fsm_intro_timer")),
        ("time", _format_optional_float(info, "career_mode_fsm_race_time_ms", suffix="ms")),
        ("comp", _format_optional_percent(info, "career_mode_fsm_completion_fraction")),
    ]
    return _join_named_values(fields)


def _career_camera_facts(info: dict[str, object]) -> str:
    fields = [
        ("target", _non_empty_text(info.get("career_mode_fsm_camera_target"))),
        ("synced", _format_optional_bool(info, "career_mode_fsm_camera_synced")),
    ]
    return _join_named_values(fields)


def _join_named_values(fields: list[tuple[str, str | None]]) -> str:
    parts = [f"{name}={value}" for name, value in fields if value is not None]
    return " · ".join(parts) if parts else "-"


def _format_optional_bool(info: dict[str, object], key: str) -> str | None:
    value = info.get(key)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return None


def _format_optional_int(info: dict[str, object], key: str) -> str | None:
    if key not in info:
        return None
    value = _int_info(info, key)
    return str(value) if value is not None else None


def _format_optional_float(
    info: dict[str, object],
    key: str,
    *,
    suffix: str = "",
) -> str | None:
    if key not in info:
        return None
    value = _float_info(info, key)
    return f"{value:.0f}{suffix}"


def _format_optional_percent(info: dict[str, object], key: str) -> str | None:
    if key not in info:
        return None
    value = _float_info(info, key)
    return f"{value * 100.0:.1f}%"


def _non_empty_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _zeroed_state_features(info: dict[str, object]) -> frozenset[str]:
    raw_features = info.get("observation_zeroed_state_features")
    if isinstance(raw_features, tuple | list):
        return frozenset(str(feature) for feature in raw_features)
    return frozenset()


def _watch_zeroed_state_features(info: dict[str, object]) -> frozenset[str]:
    raw_features = info.get("watch_zeroed_state_features")
    if isinstance(raw_features, tuple | list):
        return frozenset(str(feature) for feature in raw_features)
    return frozenset()


def _format_reward_value(value: float) -> str:
    return f"{value:.4f}"


def _format_fps_value(info: dict[str, object], key: str) -> str:
    return f"{_float_info(info, key):.1f}"


def _format_policy_deterministic(value: bool | None) -> str:
    if value is None:
        return "-"
    return "deterministic" if value else "stochastic"


def _format_driver_mode(*, policy_label: str | None, manual_control_enabled: bool) -> str:
    if policy_label is None or manual_control_enabled:
        return "manual"
    return "policy"


def _driver_mode_color(
    *,
    policy_label: str | None,
    manual_control_enabled: bool,
) -> tuple[int, int, int]:
    if manual_control_enabled:
        return PALETTE.text_warning
    if policy_label is not None:
        return PALETTE.text_accent
    return PALETTE.text_primary


def _format_env_curriculum_stage(info: dict[str, object]) -> str:
    stage_name = info.get("curriculum_stage_name")
    if isinstance(stage_name, str) and stage_name:
        return stage_name
    stage_index = info.get("curriculum_stage")
    if isinstance(stage_index, int):
        return str(stage_index)
    return "-"


def _format_curriculum_stage(*, checkpoint_stage: str | None, info: dict[str, object]) -> str:
    if checkpoint_stage is not None and checkpoint_stage:
        return checkpoint_stage
    return _format_env_curriculum_stage(info)
