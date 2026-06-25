# src/rl_fzerox/ui/watch/runtime/snapshots/build.py
"""Build and publish watch snapshots across the worker/viewer process boundary.

Workers call these helpers after emulator steps. The module keeps frame
fallbacks, action-hold interpolation, policy metadata, telemetry conversion,
and optional recording frame info in one snapshot boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from fzerox_emulator import FZeroXTelemetry, RaceControlState
from fzerox_emulator.arrays import (
    AudioFrameCounts,
    ControllerMaskBatch,
    DisplayFrames,
    Pcm16Samples,
    RgbFrame,
)
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.policy.auxiliary_state import AuxiliaryStateTargetName
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, WatchAppConfig
from rl_fzerox.ui.watch.live_series import EpisodeLiveSeriesSnapshot
from rl_fzerox.ui.watch.records import TrackRecordBook
from rl_fzerox.ui.watch.runtime.ipc import (
    WatchSnapshot,
    WorkerMessageQueue,
    publish_worker_message,
)
from rl_fzerox.ui.watch.runtime.policy.cnn import CnnActivationSnapshot
from rl_fzerox.ui.watch.runtime.policy.runner import (
    _policy_deterministic,
    _policy_experience_frames,
    _policy_label,
    _policy_num_timesteps,
    _policy_reload_age_seconds,
)
from rl_fzerox.ui.watch.runtime.snapshots.frames import (
    _audio_chunks_for_frames,
    _display_controller_states,
    _display_frames_or_fallback,
    _recording_frame_info,
)
from rl_fzerox.ui.watch.runtime.snapshots.observation import (
    _policy_auxiliary_state_predictions,
    _policy_auxiliary_state_targets,
    _policy_observation_shape,
    _policy_observation_snapshot,
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


class _FrameRecorder(Protocol):
    def record_frame(
        self,
        frame: RgbFrame,
        *,
        info: dict[str, object],
        audio_samples: Pcm16Samples = (),
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _BoostLampConfig:
    active_level: float = 0.55
    manual_level: float = 1.0
    fade_frames: int = 18


BOOST_LAMP_CONFIG = _BoostLampConfig()


@dataclass(frozen=True, slots=True)
class _StepSnapshotDisplay:
    display_frames: DisplayFrames
    display_controller_masks: ControllerMaskBatch = ()
    audio_samples: Pcm16Samples = ()
    audio_frame_counts: AudioFrameCounts = ()


@dataclass(frozen=True, slots=True)
class _StepSnapshotFrameState:
    observation: ObservationValue
    info: dict[str, object]
    episode_reward: float
    telemetry: FZeroXTelemetry | None
    control_state: RaceControlState
    gas_level: float
    action_mask_branches: ActionMaskBranches | None = None
    policy_action: ActionValue | None = None
    auxiliary_predictions: dict[str, object] | None = None
    auxiliary_targets: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class _StepSnapshotPublishRequest:
    config: WatchAppConfig
    env: _SnapshotEnv
    emulator: TelemetryReader
    snapshot_queue: WorkerMessageQueue
    display: _StepSnapshotDisplay
    previous: _StepSnapshotFrameState
    final: _StepSnapshotFrameState
    reset_info: dict[str, object]
    episode: int
    control_fps: float
    target_control_fps: float | None
    target_control_seconds: float | None
    boost_lamp_level: float
    policy_runner: PolicyRunner | None
    deterministic_policy: bool
    manual_control_enabled: bool
    policy_reload_error: str | None
    cnn_activations: CnnActivationSnapshot | None
    active_track_sampling: TrackSamplingConfig | None
    track_record_book: TrackRecordBook
    include_track_sampling: bool = False
    live_episode_series: EpisodeLiveSeriesSnapshot | None = None
    frame_interval_seconds: float | None = None
    frame_recorder: _FrameRecorder | None = None


def _publish_step_snapshots(request: _StepSnapshotPublishRequest) -> None:
    default_action_mask_branches = request.env.action_mask_branches()
    previous_action_mask_branches = (
        default_action_mask_branches
        if request.previous.action_mask_branches is None
        else request.previous.action_mask_branches
    )
    final_action_mask_branches = (
        default_action_mask_branches
        if request.final.action_mask_branches is None
        else request.final.action_mask_branches
    )

    frames = _display_frames_or_fallback(
        request.display.display_frames,
        fallback=request.env.render(),
    )
    frame_control_states, exact_frame_controls = _display_controller_states(
        request.display.display_controller_masks,
        frames=frames,
        fallback_previous=request.previous.control_state,
        fallback_final=request.final.control_state,
    )
    audio_chunks = _audio_chunks_for_frames(
        request.display.audio_samples,
        request.display.audio_frame_counts,
        frame_count=len(frames),
    )
    frame_interval_seconds = request.frame_interval_seconds
    if frame_interval_seconds is None and request.target_control_seconds is not None:
        frame_interval_seconds = request.target_control_seconds / len(frames)
    for index, frame in enumerate(frames):
        is_final_frame = index == len(frames) - 1
        frame_state = request.final if is_final_frame else request.previous
        if request.frame_recorder is not None:
            request.frame_recorder.record_frame(
                frame,
                info=_recording_frame_info(
                    frame_state.info,
                    control_state=frame_control_states[index],
                    render_input_hud=request.config.watch.recording.render_input_hud,
                    policy_active=(
                        request.policy_runner is not None and not request.manual_control_enabled
                    ),
                ),
                audio_samples=audio_chunks[index],
            )
        publish_worker_message(
            request.snapshot_queue,
            _build_snapshot(
                config=request.config,
                env=request.env,
                emulator=request.emulator,
                raw_frame=frame,
                observation=frame_state.observation,
                info=frame_state.info,
                reset_info=request.reset_info,
                episode=request.episode,
                episode_reward=frame_state.episode_reward,
                control_fps=request.control_fps,
                target_control_fps=request.target_control_fps,
                control_state=frame_control_states[index],
                gas_level=frame_state.gas_level,
                boost_lamp_level=request.boost_lamp_level,
                action_mask_branches=(
                    final_action_mask_branches
                    if is_final_frame or exact_frame_controls
                    else previous_action_mask_branches
                ),
                telemetry=frame_state.telemetry,
                policy_action=(
                    request.final.policy_action
                    if is_final_frame or exact_frame_controls
                    else request.previous.policy_action
                ),
                policy_runner=request.policy_runner,
                policy_auxiliary_state_predictions=frame_state.auxiliary_predictions,
                policy_auxiliary_state_targets=frame_state.auxiliary_targets,
                deterministic_policy=request.deterministic_policy,
                manual_control_enabled=request.manual_control_enabled,
                policy_reload_error=request.policy_reload_error,
                cnn_activations=request.cnn_activations,
                active_track_sampling=(
                    request.active_track_sampling
                    if request.include_track_sampling and is_final_frame
                    else None
                ),
                track_record_book=request.track_record_book,
                action_hold_frame=index + 1,
                action_hold_frames=len(frames),
                policy_decision_frame=is_final_frame,
                live_episode_series=request.live_episode_series if is_final_frame else None,
            ),
        )
        if frame_interval_seconds is not None and not is_final_frame:
            time.sleep(frame_interval_seconds)


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
    track_record_book: TrackRecordBook,
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
        track_record_book=track_record_book.snapshot(),
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


def _continuous_air_brake_disabled(
    config: WatchAppConfig,
    telemetry: FZeroXTelemetry | None,
) -> bool:
    if config.env.action.runtime().continuous_air_brake_mode != "disable_on_ground":
        return False
    return telemetry is not None and not telemetry.player.airborne


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
