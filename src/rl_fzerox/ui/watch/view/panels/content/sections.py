# src/rl_fzerox/ui/watch/view/panels/content/sections.py
"""Compose side-panel content sections for the Watch UI.

This module wires high-level panel columns from focused content builders.
Individual tabs own their own labels, formatting, and section-specific helper
logic in sibling modules.
"""

from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, RaceControlState
from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches
from rl_fzerox.core.runtime_spec.schema import PolicyConfig, TrainConfig
from rl_fzerox.ui.watch.records import TrackRecordBook
from rl_fzerox.ui.watch.view.auxiliary_metrics import AuxiliaryEpisodeMetricsSnapshot
from rl_fzerox.ui.watch.view.panels.content.auxiliary import auxiliary_episode_sections
from rl_fzerox.ui.watch.view.panels.content.career import career_mode_sections
from rl_fzerox.ui.watch.view.panels.content.game import (
    race_setup_section,
    race_state_section,
)
from rl_fzerox.ui.watch.view.panels.content.geometry import track_geometry_sections
from rl_fzerox.ui.watch.view.panels.content.hparams import training_hparam_sections
from rl_fzerox.ui.watch.view.panels.content.policy import policy_output_section
from rl_fzerox.ui.watch.view.panels.content.records import track_record_sections
from rl_fzerox.ui.watch.view.panels.content.runtime import (
    episode_progress_section,
    run_state_section,
    runtime_section,
)
from rl_fzerox.ui.watch.view.panels.content.state_vector import (
    policy_state_sections,
    watch_zeroed_state_features,
    zeroed_state_features,
)
from rl_fzerox.ui.watch.view.screen.types import PanelColumns


def _build_panel_columns(
    *,
    episode: int,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode_reward: float,
    paused: bool,
    control_state: RaceControlState,
    policy_label: str | None = None,
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
    track_record_book: TrackRecordBook | None = None,
    track_pool_records: tuple[dict[str, object], ...] = (),
    allow_record_course_jumps: bool = True,
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
    return PanelColumns(
        left=[
            run_state_section(
                paused=paused,
                policy_label=policy_label,
                manual_control_enabled=manual_control_enabled,
                policy_experience_frames=policy_experience_frames,
                policy_reload_age_seconds=policy_reload_age_seconds,
            ),
            episode_progress_section(
                episode=episode,
                info=info,
                episode_reward=episode_reward,
                action_repeat=action_repeat,
                max_episode_steps=max_episode_steps,
                progress_frontier_stall_limit_frames=progress_frontier_stall_limit_frames,
            ),
            policy_output_section(
                policy_action=policy_action,
                policy_deterministic=policy_deterministic,
                watch_device=watch_device,
            ),
            race_state_section(
                info,
                telemetry,
                stuck_min_speed_kph=stuck_min_speed_kph,
            ),
            race_setup_section(info, telemetry),
            runtime_section(
                info=info,
                emulator_renderer=emulator_renderer,
                action_repeat=action_repeat,
                game_display_size=game_display_size,
                observation_shape=observation_shape,
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
                zeroed_features=zeroed_state_features(info),
                watch_zeroed_features=watch_zeroed_state_features(info),
            ),
        ],
        aux=auxiliary_episode_sections(auxiliary_episode_metrics),
        records=[
            *track_record_sections(
                current_info=info,
                track_pool_records=track_pool_records,
                track_record_book=track_record_book or TrackRecordBook(),
                allow_course_jumps=allow_record_course_jumps,
            ),
        ],
        career=career_mode_sections(info),
        train=training_hparam_sections(
            train_config=train_config,
            policy_config=policy_config,
        ),
    )
