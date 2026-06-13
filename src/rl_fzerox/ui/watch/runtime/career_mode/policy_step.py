# src/rl_fzerox/ui/watch/runtime/career_mode/policy_step.py
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing.queues import Queue as ProcessQueue

from fzerox_emulator import FZeroXTelemetry, RaceControlState, SpinRequest
from rl_fzerox.core.career_mode.runner.controller import CareerModeController
from rl_fzerox.core.career_mode.runner.policy import CareerModePolicyControl
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import action_mask_violations
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.core.policy.auxiliary_state import AuxiliaryStateTargetName
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.core.training.inference import PolicyRunner
from rl_fzerox.ui.watch.live_series import (
    LIVE_SERIES_PUBLISH_POLICY,
    EpisodeLiveSeriesTracker,
)
from rl_fzerox.ui.watch.records import TrackRecordBook
from rl_fzerox.ui.watch.runtime.career_mode.recording import FrameRecorder
from rl_fzerox.ui.watch.runtime.career_mode.session import CareerModeRuntimeSession
from rl_fzerox.ui.watch.runtime.career_mode.timing import (
    measured_game_fps,
    target_game_fps,
    with_measured_game_fps,
)
from rl_fzerox.ui.watch.runtime.cnn import (
    CnnActivationNormalizationMode,
    CnnActivationSampler,
    CnnActivationSnapshot,
)
from rl_fzerox.ui.watch.runtime.observation import apply_watch_state_feature_zeroing
from rl_fzerox.ui.watch.runtime.policy import _policy_reload_error
from rl_fzerox.ui.watch.runtime.snapshots import (
    _next_boost_lamp_level,
    _publish_step_snapshots,
)
from rl_fzerox.ui.watch.runtime.telemetry import _read_live_telemetry, _telemetry_to_data
from rl_fzerox.ui.watch.runtime.timing import RateMeter
from rl_fzerox.ui.watch.runtime.visualization import (
    current_auxiliary_predictions as _current_auxiliary_predictions,
)
from rl_fzerox.ui.watch.runtime.visualization import (
    current_auxiliary_targets as _current_auxiliary_targets,
)


@dataclass(frozen=True, slots=True)
class CareerPolicyStepResult:
    raw_observation: ObservationValue
    raw_info: dict[str, object]
    observation: ObservationValue
    info: dict[str, object]
    episode_reward: float
    control_state: RaceControlState
    gas_level: float
    boost_lamp_level: float
    policy_action: ActionValue | None
    cnn_activations: CnnActivationSnapshot | None
    telemetry: FZeroXTelemetry | None
    auxiliary_predictions: dict[str, object] | None
    auxiliary_targets: dict[str, object] | None
    last_live_series_publish_time: float


def step_policy_or_manual(
    *,
    config: WatchAppConfig,
    session: CareerModeRuntimeSession,
    controller: CareerModeController,
    snapshot_queue: ProcessQueue,
    active_policy_control: CareerModePolicyControl,
    policy_runner: PolicyRunner,
    observation: ObservationValue,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode: int,
    episode_reward: float,
    control_rate: RateMeter,
    target_policy_control_fps: float | None,
    target_control_seconds: float | None,
    deterministic_policy: bool,
    manual_control_enabled: bool,
    current_control_state: RaceControlState,
    spin_request: SpinRequest,
    boost_lamp_level: float,
    cnn_visualization_enabled: bool,
    cnn_normalization: CnnActivationNormalizationMode,
    cnn_sampler: CnnActivationSampler,
    auxiliary_visualization_enabled: bool,
    auxiliary_target_names: tuple[AuxiliaryStateTargetName, ...],
    watch_zeroed_state_features: frozenset[str],
    live_visualization_enabled: bool,
    live_series: EpisodeLiveSeriesTracker,
    last_live_series_publish_time: float,
    frame_recorder: FrameRecorder | None = None,
) -> CareerPolicyStepResult:
    previous_observation = observation
    previous_info = controller.viewer_info(
        info=dict(info),
        active_policy_control=active_policy_control,
    )
    previous_episode_reward = episode_reward
    previous_telemetry = _read_live_telemetry(session.emulator)
    previous_control_state = current_control_state
    previous_gas_level = session.last_gas_level
    previous_action_mask_branches = session.action_mask_branches()
    previous_auxiliary_predictions = _current_auxiliary_predictions(
        policy_runner=policy_runner,
        enabled=auxiliary_visualization_enabled,
        observation=observation,
        target_names=auxiliary_target_names,
    )
    previous_auxiliary_targets = _current_auxiliary_targets(
        telemetry=previous_telemetry,
        enabled=auxiliary_visualization_enabled,
        target_names=auxiliary_target_names,
    )
    if manual_control_enabled:
        race_step = session.step_manual_race(
            current_control_state,
            spin_request=spin_request,
        )
        raw_observation = race_step.observation
        raw_info = race_step.info
        current_policy_action = None
        cnn_activations = None
    else:
        session.sync_policy_curriculum_stage(policy_runner.checkpoint_curriculum_stage_index)
        decision_action_mask = session.action_mask_snapshot()
        policy_action_mask = decision_action_mask if policy_runner.supports_action_masks else None
        action = policy_runner.predict(
            observation,
            deterministic=deterministic_policy,
            action_masks=policy_action_mask.flat if policy_action_mask is not None else None,
            refresh=False,
        )
        if policy_action_mask is not None:
            violations = action_mask_violations(policy_action_mask.branches, action)
            if violations:
                details = ", ".join(violations)
                raise RuntimeError(f"Policy selected masked action values: {details}")
        current_policy_action = action
        cnn_activations = cnn_sampler.capture(
            enabled=cnn_visualization_enabled,
            policy_runner=policy_runner,
            observation=observation,
            normalization=cnn_normalization,
        )
        race_step = session.step_policy(action)
        raw_observation = race_step.observation
        raw_info = race_step.info

    observation, info = apply_watch_state_feature_zeroing(
        raw_observation,
        raw_info,
        watch_zeroed_features=watch_zeroed_state_features,
    )
    info = controller.viewer_info(
        info=info,
        active_policy_control=active_policy_control,
    )
    current_control_state = session.last_requested_control_state
    current_gas_level = session.last_gas_level
    final_action_mask_branches = session.action_mask_branches()
    live_telemetry = _read_live_telemetry(session.emulator)
    final_auxiliary_predictions = _current_auxiliary_predictions(
        policy_runner=policy_runner,
        enabled=auxiliary_visualization_enabled,
        observation=observation,
        target_names=auxiliary_target_names,
    )
    final_auxiliary_targets = _current_auxiliary_targets(
        telemetry=live_telemetry,
        enabled=auxiliary_visualization_enabled,
        target_names=auxiliary_target_names,
    )
    snapshot_config = session.snapshot_config(config)
    boost_lamp_level = _next_boost_lamp_level(
        previous=boost_lamp_level,
        control_state=current_control_state,
        boost_active=telemetry_boost_active(live_telemetry),
        action_repeat=snapshot_config.env.action_repeat,
    )
    control_rate.tick()
    episode_reward = required_episode_return(info)
    measured_game_fps_value = measured_game_fps(
        control_fps=control_rate.rate_hz(),
        action_repeat=snapshot_config.env.action_repeat,
    )
    previous_info = _with_policy_game_fps(
        previous_info,
        measured_game_fps_value=measured_game_fps_value,
        target_policy_control_fps=target_policy_control_fps,
        action_repeat=snapshot_config.env.action_repeat,
    )
    info = _with_policy_game_fps(
        info,
        measured_game_fps_value=measured_game_fps_value,
        target_policy_control_fps=target_policy_control_fps,
        action_repeat=snapshot_config.env.action_repeat,
    )
    live_series.observe_decision(
        episode=episode,
        info=info,
        episode_reward=episode_reward,
        telemetry_data=_telemetry_to_data(live_telemetry),
        action_repeat=snapshot_config.env.action_repeat,
    )
    live_episode_series = None
    if live_visualization_enabled:
        current_time = time.perf_counter()
        if (
            current_time - last_live_series_publish_time
            >= LIVE_SERIES_PUBLISH_POLICY.interval_seconds
        ):
            live_episode_series = live_series.snapshot()
            last_live_series_publish_time = current_time
    _publish_step_snapshots(
        config=snapshot_config,
        env=session,
        emulator=session.emulator,
        snapshot_queue=snapshot_queue,
        display_frames=race_step.display_frames,
        display_controller_masks=race_step.display_controller_masks,
        previous_observation=previous_observation,
        previous_info=previous_info,
        previous_episode_reward=previous_episode_reward,
        previous_telemetry=previous_telemetry,
        final_observation=observation,
        final_info=info,
        final_episode_reward=episode_reward,
        final_telemetry=live_telemetry,
        previous_control_state=previous_control_state,
        previous_gas_level=previous_gas_level,
        previous_action_mask_branches=previous_action_mask_branches,
        previous_policy_action=None,
        final_control_state=current_control_state,
        final_gas_level=current_gas_level,
        final_action_mask_branches=final_action_mask_branches,
        final_policy_action=current_policy_action,
        previous_auxiliary_predictions=previous_auxiliary_predictions,
        previous_auxiliary_targets=previous_auxiliary_targets,
        final_auxiliary_predictions=final_auxiliary_predictions,
        final_auxiliary_targets=final_auxiliary_targets,
        reset_info=reset_info,
        episode=episode,
        control_fps=control_rate.rate_hz(),
        target_control_fps=target_policy_control_fps,
        target_control_seconds=target_control_seconds,
        boost_lamp_level=boost_lamp_level,
        policy_runner=policy_runner,
        deterministic_policy=deterministic_policy,
        policy_reload_error=_policy_reload_error(policy_runner),
        cnn_activations=cnn_activations,
        active_track_sampling=None,
        track_record_book=TrackRecordBook(),
        manual_control_enabled=manual_control_enabled,
        live_episode_series=live_episode_series,
        frame_recorder=frame_recorder,
    )
    return CareerPolicyStepResult(
        raw_observation=raw_observation,
        raw_info=raw_info,
        observation=observation,
        info=info,
        episode_reward=episode_reward,
        control_state=current_control_state,
        gas_level=current_gas_level,
        boost_lamp_level=boost_lamp_level,
        policy_action=current_policy_action,
        cnn_activations=cnn_activations,
        telemetry=live_telemetry,
        auxiliary_predictions=final_auxiliary_predictions,
        auxiliary_targets=final_auxiliary_targets,
        last_live_series_publish_time=last_live_series_publish_time,
    )


def required_episode_return(info: Mapping[str, object]) -> float:
    value = info.get("episode_return")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise RuntimeError("Career Mode policy step did not publish episode_return")


def _with_policy_game_fps(
    info: dict[str, object],
    *,
    measured_game_fps_value: float,
    target_policy_control_fps: float | None,
    action_repeat: int,
) -> dict[str, object]:
    return with_measured_game_fps(
        info,
        game_fps=measured_game_fps_value,
        game_fps_target=target_game_fps(
            target_control_fps=target_policy_control_fps,
            action_repeat=action_repeat,
        ),
    )
