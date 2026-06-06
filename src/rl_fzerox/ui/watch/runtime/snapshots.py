# src/rl_fzerox/ui/watch/runtime/snapshots.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from fzerox_emulator import FZeroXTelemetry, RaceControlState, stacked_observation_channels
from fzerox_emulator.arrays import ControllerMaskBatch, DisplayFrames, RgbFrame, StateVector
from rl_fzerox.core.envs import observations as observation_access
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.policy.auxiliary_state import (
    AuxiliaryStateTargetName,
    auxiliary_state_target_values,
)
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, WatchAppConfig
from rl_fzerox.ui.watch.live_series import EpisodeLiveSeriesSnapshot
from rl_fzerox.ui.watch.runtime.cnn import CnnActivationSnapshot
from rl_fzerox.ui.watch.runtime.episode import _update_best_finish_position
from rl_fzerox.ui.watch.runtime.ipc import (
    PolicyObservationSnapshot,
    WatchSnapshot,
    WorkerMessageQueue,
    publish_worker_message,
)
from rl_fzerox.ui.watch.runtime.policy import (
    _policy_curriculum_stage,
    _policy_deterministic,
    _policy_experience_frames,
    _policy_label,
    _policy_num_timesteps,
    _policy_reload_age_seconds,
)
from rl_fzerox.ui.watch.runtime.telemetry import (
    TelemetryReader,
    _read_live_telemetry,
    _telemetry_to_data,
)

if TYPE_CHECKING:
    from rl_fzerox.core.training.inference import PolicyRunner


class _SnapshotBackend(Protocol):
    @property
    def native_fps(self) -> float: ...


class _SnapshotEnv(Protocol):
    @property
    def backend(self) -> _SnapshotBackend: ...

    def render(self) -> RgbFrame: ...

    def action_mask_branches(self) -> ActionMaskBranches: ...


@dataclass(frozen=True, slots=True)
class _BoostLampConfig:
    active_level: float = 0.55
    manual_level: float = 1.0
    fade_frames: int = 18


BOOST_LAMP_CONFIG = _BoostLampConfig()


def _publish_step_snapshots(
    *,
    config: WatchAppConfig,
    env: _SnapshotEnv,
    emulator: TelemetryReader,
    snapshot_queue: WorkerMessageQueue,
    display_frames: DisplayFrames,
    display_controller_masks: ControllerMaskBatch = (),
    previous_observation: ObservationValue,
    previous_info: dict[str, object],
    previous_episode_reward: float,
    previous_telemetry: FZeroXTelemetry | None,
    final_observation: ObservationValue,
    final_info: dict[str, object],
    final_episode_reward: float,
    final_telemetry: FZeroXTelemetry | None,
    reset_info: dict[str, object],
    episode: int,
    control_fps: float,
    target_control_fps: float | None,
    target_control_seconds: float | None,
    boost_lamp_level: float,
    policy_runner: PolicyRunner | None,
    deterministic_policy: bool,
    manual_control_enabled: bool,
    policy_reload_error: str | None,
    cnn_activations: CnnActivationSnapshot | None,
    active_track_sampling: TrackSamplingConfig | None,
    best_finish_position: int | None,
    best_finish_ranks: dict[str, int],
    best_finish_times: dict[str, int],
    latest_finish_times: dict[str, int],
    latest_finish_deltas_ms: dict[str, int],
    failed_track_attempts: frozenset[str],
    previous_control_state: RaceControlState | None = None,
    previous_gas_level: float | None = None,
    previous_action_mask_branches: ActionMaskBranches | None = None,
    previous_policy_action: ActionValue | None = None,
    final_control_state: RaceControlState | None = None,
    final_gas_level: float | None = None,
    final_action_mask_branches: ActionMaskBranches | None = None,
    final_policy_action: ActionValue | None = None,
    control_state: RaceControlState | None = None,
    gas_level: float | None = None,
    action_mask_branches: ActionMaskBranches | None = None,
    policy_action: ActionValue | None = None,
    previous_auxiliary_predictions: dict[str, object] | None = None,
    previous_auxiliary_targets: dict[str, object] | None = None,
    final_auxiliary_predictions: dict[str, object] | None = None,
    final_auxiliary_targets: dict[str, object] | None = None,
    live_episode_series: EpisodeLiveSeriesSnapshot | None = None,
    frame_interval_seconds: float | None = None,
) -> None:
    resolved_control_state = control_state
    if previous_control_state is None:
        previous_control_state = resolved_control_state
    if final_control_state is None:
        final_control_state = (
            resolved_control_state if resolved_control_state is not None else previous_control_state
        )
    if previous_control_state is None or final_control_state is None:
        raise TypeError(
            "Step snapshot publishing requires control state for hold and final frames."
        )

    resolved_gas_level = gas_level
    if previous_gas_level is None:
        previous_gas_level = resolved_gas_level
    if final_gas_level is None:
        final_gas_level = (
            resolved_gas_level if resolved_gas_level is not None else previous_gas_level
        )
    if previous_gas_level is None or final_gas_level is None:
        raise TypeError("Step snapshot publishing requires gas level for hold and final frames.")

    default_action_mask_branches = (
        env.action_mask_branches() if action_mask_branches is None else action_mask_branches
    )
    if previous_action_mask_branches is None:
        previous_action_mask_branches = default_action_mask_branches
    if final_action_mask_branches is None:
        final_action_mask_branches = default_action_mask_branches

    if previous_policy_action is None:
        previous_policy_action = policy_action
    if final_policy_action is None:
        final_policy_action = policy_action

    frames = _display_frames_or_fallback(display_frames, fallback=env.render())
    frame_control_states, exact_frame_controls = _display_controller_states(
        display_controller_masks,
        frames=frames,
        fallback_previous=previous_control_state,
        fallback_final=final_control_state,
    )
    if frame_interval_seconds is None and target_control_seconds is not None:
        frame_interval_seconds = target_control_seconds / len(frames)
    for index, frame in enumerate(frames):
        is_final_frame = index == len(frames) - 1
        publish_worker_message(
            snapshot_queue,
            _build_snapshot(
                config=config,
                env=env,
                emulator=emulator,
                raw_frame=frame,
                observation=final_observation if is_final_frame else previous_observation,
                info=final_info if is_final_frame else previous_info,
                reset_info=reset_info,
                episode=episode,
                episode_reward=(
                    final_episode_reward if is_final_frame else previous_episode_reward
                ),
                control_fps=control_fps,
                target_control_fps=target_control_fps,
                control_state=frame_control_states[index],
                gas_level=final_gas_level if is_final_frame else previous_gas_level,
                boost_lamp_level=boost_lamp_level,
                action_mask_branches=(
                    final_action_mask_branches
                    if is_final_frame or exact_frame_controls
                    else previous_action_mask_branches
                ),
                telemetry=final_telemetry if is_final_frame else previous_telemetry,
                policy_action=(
                    final_policy_action
                    if is_final_frame or exact_frame_controls
                    else previous_policy_action
                ),
                policy_runner=policy_runner,
                policy_auxiliary_state_predictions=(
                    final_auxiliary_predictions
                    if is_final_frame
                    else previous_auxiliary_predictions
                ),
                policy_auxiliary_state_targets=(
                    final_auxiliary_targets if is_final_frame else previous_auxiliary_targets
                ),
                deterministic_policy=deterministic_policy,
                manual_control_enabled=manual_control_enabled,
                policy_reload_error=policy_reload_error,
                cnn_activations=cnn_activations,
                active_track_sampling=active_track_sampling,
                best_finish_position=best_finish_position,
                best_finish_ranks=best_finish_ranks,
                best_finish_times=best_finish_times,
                latest_finish_times=latest_finish_times,
                latest_finish_deltas_ms=latest_finish_deltas_ms,
                failed_track_attempts=failed_track_attempts,
                action_hold_frame=index + 1,
                action_hold_frames=len(frames),
                policy_decision_frame=is_final_frame,
                live_episode_series=live_episode_series if is_final_frame else None,
            ),
        )
        if frame_interval_seconds is not None and not is_final_frame:
            time.sleep(frame_interval_seconds)


def _display_frames_or_fallback(
    display_frames: DisplayFrames,
    *,
    fallback: RgbFrame,
) -> DisplayFrames:
    if isinstance(display_frames, tuple):
        return display_frames if display_frames else (fallback,)
    return display_frames if len(display_frames) > 0 else (fallback,)


def _display_controller_states(
    display_controller_masks: ControllerMaskBatch,
    *,
    frames: DisplayFrames,
    fallback_previous: RaceControlState,
    fallback_final: RaceControlState,
) -> tuple[tuple[RaceControlState, ...], bool]:
    frame_count = len(frames)
    masks = tuple(int(mask) for mask in display_controller_masks)
    if len(masks) != frame_count:
        return (
            tuple(
                fallback_final if index == frame_count - 1 else fallback_previous
                for index in range(frame_count)
            ),
            False,
        )
    return (
        tuple(
            RaceControlState.from_mask(
                mask,
                stick_x=fallback_final.stick_x,
                pitch=fallback_final.pitch,
            )
            for mask in masks
        ),
        True,
    )


def _build_snapshot(
    *,
    config: WatchAppConfig,
    env: _SnapshotEnv,
    emulator: TelemetryReader,
    raw_frame: RgbFrame | None = None,
    observation: ObservationValue | None,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode: int,
    episode_reward: float,
    control_fps: float,
    target_control_fps: float | None,
    control_state: RaceControlState,
    gas_level: float,
    boost_lamp_level: float,
    action_mask_branches: ActionMaskBranches | None = None,
    policy_action: ActionValue | None,
    policy_runner: PolicyRunner | None,
    deterministic_policy: bool,
    manual_control_enabled: bool,
    policy_reload_error: str | None,
    cnn_activations: CnnActivationSnapshot | None,
    best_finish_position: int | None,
    best_finish_ranks: dict[str, int],
    best_finish_times: dict[str, int],
    latest_finish_times: dict[str, int],
    latest_finish_deltas_ms: dict[str, int],
    failed_track_attempts: frozenset[str],
    action_repeat: int | None = None,
    active_track_sampling: TrackSamplingConfig | None = None,
    telemetry: FZeroXTelemetry | None = None,
    policy_auxiliary_state_predictions: dict[str, object] | None = None,
    policy_auxiliary_state_targets: dict[str, object] | None = None,
    include_auxiliary_state: bool = False,
    auxiliary_target_names: tuple[AuxiliaryStateTargetName, ...] = (),
    action_hold_frame: int = 1,
    action_hold_frames: int = 1,
    policy_decision_frame: bool = True,
    live_episode_series: EpisodeLiveSeriesSnapshot | None = None,
) -> WatchSnapshot:
    if telemetry is None:
        telemetry = _read_live_telemetry(emulator)
    best_finish_position = _update_best_finish_position(best_finish_position, info, telemetry)
    policy_observation = _policy_observation_snapshot(
        config=config,
        observation=observation,
        telemetry=telemetry,
        info=info,
    )
    return WatchSnapshot(
        raw_frame=env.render() if raw_frame is None else raw_frame,
        policy_observation_shape=_policy_observation_shape(config, observation),
        policy_observation=policy_observation,
        info=dict(info),
        reset_info=dict(reset_info),
        episode=episode,
        episode_reward=episode_reward,
        control_fps=control_fps,
        target_control_fps=target_control_fps,
        action_repeat=max(
            1,
            int(config.env.action_repeat if action_repeat is None else action_repeat),
        ),
        native_fps=float(env.backend.native_fps),
        control_state=control_state,
        gas_level=gas_level,
        boost_lamp_level=max(0.0, min(1.0, boost_lamp_level)),
        action_mask_branches=(
            env.action_mask_branches() if action_mask_branches is None else action_mask_branches
        ),
        policy_action=policy_action,
        policy_label=_policy_label(policy_runner),
        policy_curriculum_stage=_policy_curriculum_stage(policy_runner),
        policy_num_timesteps=_policy_num_timesteps(policy_runner),
        policy_experience_frames=_policy_experience_frames(
            policy_runner,
            action_repeat=config.env.action_repeat,
            lineage_frame_offset=config.watch.lineage_frame_offset,
        ),
        policy_deterministic=_policy_deterministic(
            policy_runner,
            deterministic_policy,
        ),
        manual_control_enabled=manual_control_enabled,
        policy_reload_age_seconds=_policy_reload_age_seconds(policy_runner),
        policy_reload_error=policy_reload_error,
        cnn_activations=cnn_activations,
        best_finish_position=best_finish_position,
        best_finish_ranks=dict(best_finish_ranks),
        best_finish_times=dict(best_finish_times),
        latest_finish_times=dict(latest_finish_times),
        latest_finish_deltas_ms=dict(latest_finish_deltas_ms),
        failed_track_attempts=failed_track_attempts,
        continuous_air_brake_disabled=_continuous_air_brake_disabled(config, telemetry),
        telemetry_data=_telemetry_to_data(telemetry),
        active_track_sampling=active_track_sampling,
        policy_auxiliary_state_predictions=(
            _policy_auxiliary_state_predictions(
                policy_runner=policy_runner,
                observation=observation,
                target_names=auxiliary_target_names,
            )
            if (
                include_auxiliary_state
                and observation is not None
                and policy_auxiliary_state_predictions is None
            )
            else policy_auxiliary_state_predictions
        ),
        policy_auxiliary_state_targets=(
            _policy_auxiliary_state_targets(telemetry, target_names=auxiliary_target_names)
            if include_auxiliary_state and policy_auxiliary_state_targets is None
            else policy_auxiliary_state_targets
        ),
        action_hold_frame=max(1, int(action_hold_frame)),
        action_hold_frames=max(1, int(action_hold_frames)),
        policy_decision_frame=bool(policy_decision_frame),
        live_episode_series=live_episode_series,
    )


def _policy_observation_snapshot(
    *,
    config: WatchAppConfig,
    observation: ObservationValue | None,
    telemetry: FZeroXTelemetry | None,
    info: dict[str, object],
) -> PolicyObservationSnapshot | None:
    if observation is None:
        return None
    return PolicyObservationSnapshot(
        image=observation_access.observation_image(observation),
        state=observation_access.observation_state(observation),
        state_reference=_reference_observation_state(
            config=config,
            observation=observation,
            telemetry=telemetry,
            info=info,
        ),
    )


def _policy_observation_shape(
    config: WatchAppConfig,
    observation: ObservationValue | None,
) -> tuple[int, ...] | None:
    if observation is not None:
        return tuple(int(value) for value in observation_access.observation_image(observation).shape)
    if config.watch.policy_observation_shape_hint is not None:
        return tuple(int(value) for value in config.watch.policy_observation_shape_hint)
    if config.watch.managed_save_game_id is not None:
        return None
    height, width = config.env.observation.image_geometry(renderer=config.emulator.renderer)
    channels = stacked_observation_channels(
        3,
        frame_stack=config.env.observation.frame_stack,
        stack_mode=config.env.observation.stack_mode,
        minimap_layer=config.env.observation.minimap_layer,
    )
    return int(height), int(width), int(channels)


def _reference_observation_state(
    *,
    config: WatchAppConfig,
    observation: ObservationValue,
    telemetry: FZeroXTelemetry | None,
    info: dict[str, object],
) -> StateVector | None:
    observed_state = observation_access.observation_state(observation)
    if config.env.observation.mode != "image_state":
        return observed_state
    state_components = config.env.observation.state_components_data()
    if state_components is None:
        return observed_state
    return observation_access.telemetry_state_vector(
        telemetry,
        state_components=state_components,
        action_history=_reference_action_history(
            observed_state=observed_state,
            info=info,
        ),
        split_lean_history=config.env.action.runtime().split_lean_history,
    )


def _reference_action_history(
    *,
    observed_state: StateVector | None,
    info: dict[str, object],
) -> dict[str, float]:
    if observed_state is None:
        return {}
    raw_feature_names = info.get("observation_state_features")
    if isinstance(raw_feature_names, list):
        feature_names = tuple(str(name) for name in raw_feature_names)
    elif isinstance(raw_feature_names, tuple):
        feature_names = tuple(str(name) for name in raw_feature_names)
    else:
        return {}
    flat_state = observed_state.reshape(-1)
    if len(feature_names) != int(flat_state.size):
        return {}
    return {
        name.removeprefix("control_history."): float(value)
        for name, value in zip(feature_names, flat_state, strict=True)
        if name.startswith("control_history.")
    }


def _continuous_air_brake_disabled(
    config: WatchAppConfig,
    telemetry: FZeroXTelemetry | None,
) -> bool:
    if config.env.action.runtime().continuous_air_brake_mode != "disable_on_ground":
        return False
    return telemetry is not None and not telemetry.player.airborne


def _policy_auxiliary_state_predictions(
    *,
    policy_runner: PolicyRunner | None,
    observation: ObservationValue,
    target_names: tuple[AuxiliaryStateTargetName, ...],
) -> dict[str, object] | None:
    if policy_runner is None:
        return None
    return policy_runner.auxiliary_state_predictions(
        observation,
        target_names=target_names,
    )


def _policy_auxiliary_state_targets(
    telemetry: FZeroXTelemetry | None,
    *,
    target_names: tuple[AuxiliaryStateTargetName, ...],
) -> dict[str, object]:
    if telemetry is None:
        return {}
    all_targets = auxiliary_state_target_values(telemetry)
    return {str(name): value for name, value in all_targets.items() if name in target_names}


def _next_boost_lamp_level(
    *,
    previous: float,
    control_state: RaceControlState,
    boost_active: bool,
    action_repeat: int,
) -> float:
    if control_state.boost:
        return BOOST_LAMP_CONFIG.manual_level

    target = BOOST_LAMP_CONFIG.active_level if boost_active else 0.0
    if previous <= target:
        return target

    decay = max(1, action_repeat) / BOOST_LAMP_CONFIG.fade_frames
    return max(target, previous - decay)
