# src/rl_fzerox/ui/watch/view/panels/content/sections.py
from __future__ import annotations

from fzerox_emulator import ControllerState, FZeroXTelemetry
from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.config.schema import PolicyConfig, TrainConfig
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches
from rl_fzerox.ui.watch.view.panels.content.game import (
    game_detail_section,
    game_overview_section,
)
from rl_fzerox.ui.watch.view.panels.content.geometry import track_geometry_sections
from rl_fzerox.ui.watch.view.panels.content.hparams import training_hparam_sections
from rl_fzerox.ui.watch.view.panels.content.records import track_record_sections
from rl_fzerox.ui.watch.view.panels.content.state_vector import policy_state_sections
from rl_fzerox.ui.watch.view.panels.core.format import (
    _float_info,
    _format_checkpoint_experience,
    _format_control_rate,
    _format_env_step,
    _format_episode_frames,
    _format_game_rate,
    _format_game_speed,
    _format_observation_shape,
    _format_policy_action,
    _format_progress_frontier_counter,
    _format_reload_age,
    _format_render_rate,
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
    control_state: ControllerState,
    policy_label: str | None = None,
    policy_curriculum_stage: str | None,
    policy_action: ActionValue | None,
    policy_reload_age_seconds: float | None,
    policy_reload_error: str | None,
    policy_num_timesteps: int | None = None,
    gas_level: float = 0.0,
    thrust_warning_threshold: float | None = None,
    boost_active: bool = False,
    boost_lamp_level: float = 0.0,
    action_mask_branches: ActionMaskBranches | None = None,
    best_finish_position: int | None = None,
    best_finish_times: dict[str, int] | None = None,
    latest_finish_times: dict[str, int] | None = None,
    latest_finish_deltas_ms: dict[str, int] | None = None,
    failed_track_attempts: frozenset[str] = frozenset(),
    track_pool_records: tuple[dict[str, object], ...] = (),
    continuous_drive_deadzone: float = 0.2,
    continuous_air_brake_mode: str = "always",
    continuous_air_brake_disabled: bool = False,
    action_repeat: int,
    stuck_min_speed_kph: float,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
    telemetry: FZeroXTelemetry | None,
    policy_deterministic: bool | None = None,
    manual_control_enabled: bool = False,
    max_episode_steps: int = 50_000,
    progress_frontier_stall_limit_frames: int | None = 900,
    observation_state: StateVector | None = None,
    observation_state_feature_names: tuple[str, ...] = (),
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
                title="Run",
                lines=[
                    _panel_line(
                        "State",
                        "paused" if paused else "running",
                        PALETTE.text_warning if paused else PALETTE.text_accent,
                    ),
                    _panel_line(
                        "Stage",
                        curriculum_stage,
                        PALETTE.text_primary if curriculum_stage != "-" else PALETTE.text_muted,
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
                        "Experience",
                        _format_checkpoint_experience(
                            policy_num_timesteps,
                            action_repeat=action_repeat,
                        ),
                        PALETTE.text_primary
                        if policy_num_timesteps is not None
                        else PALETTE.text_muted,
                    ),
                    _panel_line(
                        "Reload",
                        _format_reload_age(policy_reload_age_seconds),
                        PALETTE.text_primary,
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
            PanelSection(
                title="Policy Details",
                lines=[
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
                ],
            ),
            game_overview_section(
                info,
                telemetry,
                stuck_min_speed_kph=stuck_min_speed_kph,
            ),
            game_detail_section(info, telemetry),
            PanelSection(
                title="Timing",
                lines=[
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
                        "Game speed",
                        _format_game_speed(info, action_repeat=action_repeat),
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
                ],
            ),
        ],
        middle=[
            *track_geometry_sections(telemetry),
        ],
        stats=[
            *policy_state_sections(
                observation_state=observation_state,
                feature_names=observation_state_feature_names,
                zeroed_components=_zeroed_state_components(info),
                zeroed_features=_zeroed_state_features(info),
            ),
        ],
        records=[
            *track_record_sections(
                current_info=info,
                track_pool_records=track_pool_records,
                best_finish_times=best_finish_times or {},
                latest_finish_times=latest_finish_times or {},
                latest_finish_deltas_ms=latest_finish_deltas_ms or {},
                failed_track_attempts=failed_track_attempts,
            ),
        ],
        train=training_hparam_sections(
            train_config=train_config,
            policy_config=policy_config,
        ),
    )


def _zeroed_state_components(info: dict[str, object]) -> frozenset[str]:
    raw_components = info.get("observation_zeroed_state_components")
    if not isinstance(raw_components, tuple | list):
        return frozenset()
    return frozenset(
        component for component in raw_components if isinstance(component, str) and component
    )


def _zeroed_state_features(info: dict[str, object]) -> frozenset[str]:
    raw_features = info.get("observation_zeroed_state_features")
    if isinstance(raw_features, tuple | list):
        return frozenset(str(feature) for feature in raw_features)
    return frozenset()


def _format_reward_value(value: float) -> str:
    return f"{value:.4f}"


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


def _format_best_position(value: int | None) -> str:
    return "n/a" if value is None else str(value)
