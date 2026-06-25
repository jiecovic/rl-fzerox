# src/rl_fzerox/ui/watch/runtime/live/step.py
"""One live Watch env step plus frame snapshot publication.

The live worker owns episode reset, command draining, pause/wait timing, and
track-sampling state. This module owns the per-step policy/manual execution
boundary: capture previous state, step the env, refresh final state, publish
display-frame snapshots, and return the values the worker carries forward.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from fzerox_emulator import FZeroXTelemetry, RaceControlState, SpinRequest
from fzerox_emulator.arrays import ControllerMaskBatch, DisplayFrames, RgbFrame
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import (
    ActionMaskBranches,
    ActionMaskSnapshot,
    action_mask_violations,
)
from rl_fzerox.core.envs.engine.stepping import WatchEnvStep
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.live_series import (
    LIVE_SERIES_PUBLISH_POLICY,
    EpisodeLiveSeriesSnapshot,
    EpisodeLiveSeriesTracker,
)
from rl_fzerox.ui.watch.records import TrackRecordBook
from rl_fzerox.ui.watch.runtime.ipc import WorkerMessageQueue
from rl_fzerox.ui.watch.runtime.live.notices import _TimedWatchNotice
from rl_fzerox.ui.watch.runtime.observation import apply_watch_state_feature_zeroing
from rl_fzerox.ui.watch.runtime.policy.cnn import (
    CnnActivationNormalizationMode,
    CnnActivationSampler,
    CnnActivationSnapshot,
)
from rl_fzerox.ui.watch.runtime.policy.visualization import (
    current_auxiliary_predictions as _current_auxiliary_predictions,
)
from rl_fzerox.ui.watch.runtime.policy.visualization import (
    current_auxiliary_targets as _current_auxiliary_targets,
)
from rl_fzerox.ui.watch.runtime.snapshots.build import (
    _next_boost_lamp_level,
    _publish_step_snapshots,
    _StepSnapshotDisplay,
    _StepSnapshotFrameState,
    _StepSnapshotPublishRequest,
)
from rl_fzerox.ui.watch.runtime.telemetry import (
    TelemetryReader,
    _read_live_telemetry,
    _telemetry_to_data,
)
from rl_fzerox.ui.watch.runtime.timing import RateMeter

if TYPE_CHECKING:
    from rl_fzerox.core.policy.auxiliary_state import AuxiliaryStateTargetName
    from rl_fzerox.core.training.inference import PolicyRunner


class _LiveStepBackend(Protocol):
    @property
    def native_fps(self) -> float: ...


class _LiveStepEnv(Protocol):
    @property
    def backend(self) -> _LiveStepBackend: ...

    @property
    def last_requested_control_state(self) -> RaceControlState: ...

    @property
    def last_gas_level(self) -> float: ...

    def render(self) -> RgbFrame: ...

    def action_mask_branches(self) -> ActionMaskBranches: ...

    def action_mask_snapshot(self) -> ActionMaskSnapshot: ...

    def step_watch(self, action: ActionValue) -> WatchEnvStep: ...

    def step_control_watch(
        self,
        control_state: RaceControlState,
        *,
        spin_request: SpinRequest = "none",
    ) -> WatchEnvStep: ...

    def step_frame(
        self,
        control_state: RaceControlState | None = None,
        *,
        spin_request: SpinRequest = "none",
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]: ...


@dataclass(frozen=True, slots=True)
class LiveStepRequest:
    config: WatchAppConfig
    env: _LiveStepEnv
    emulator: TelemetryReader
    snapshot_queue: WorkerMessageQueue
    policy_runner: PolicyRunner | None
    observation: ObservationValue
    info: dict[str, object]
    reset_info: dict[str, object]
    episode: int
    episode_reward: float
    control_rate: RateMeter
    target_control_fps: float | None
    target_control_seconds: float | None
    deterministic_policy: bool
    manual_control_enabled: bool
    single_frame_manual: bool
    current_control_state: RaceControlState
    current_gas_level: float
    spin_request: SpinRequest
    boost_lamp_level: float
    committed_policy_action: ActionValue | None
    committed_action_mask_branches: ActionMaskBranches
    current_auxiliary_predictions: dict[str, object] | None
    current_auxiliary_targets: dict[str, object] | None
    cnn_visualization_enabled: bool
    cnn_normalization: CnnActivationNormalizationMode
    cnn_sampler: CnnActivationSampler
    auxiliary_visualization_enabled: bool
    auxiliary_target_names: tuple[AuxiliaryStateTargetName, ...]
    watch_zeroed_state_features: frozenset[str]
    live_visualization_enabled: bool
    live_series: EpisodeLiveSeriesTracker
    last_live_series_publish_time: float
    track_record_book: TrackRecordBook
    save_notice: _TimedWatchNotice
    policy_reload_error: str | None


@dataclass(frozen=True, slots=True)
class LiveStepResult:
    raw_observation: ObservationValue
    raw_info: dict[str, object]
    observation: ObservationValue
    info: dict[str, object]
    terminated: bool
    truncated: bool
    episode_reward: float
    control_state: RaceControlState
    gas_level: float
    boost_lamp_level: float
    policy_action: ActionValue | None
    cnn_activations: CnnActivationSnapshot | None
    telemetry: FZeroXTelemetry | None
    auxiliary_predictions: dict[str, object] | None
    auxiliary_targets: dict[str, object] | None
    committed_action_mask_branches: ActionMaskBranches
    track_record_book: TrackRecordBook
    last_live_series_publish_time: float


def step_policy_or_manual(request: LiveStepRequest) -> LiveStepResult:
    previous_observation = request.observation
    previous_info = dict(request.info)
    previous_episode_reward = request.episode_reward
    previous_telemetry = _read_live_telemetry(request.emulator)
    previous_control_state = request.current_control_state
    previous_gas_level = request.current_gas_level
    previous_policy_action = request.committed_policy_action
    previous_action_mask_branches = request.committed_action_mask_branches
    previous_auxiliary_predictions = request.current_auxiliary_predictions
    previous_auxiliary_targets = request.current_auxiliary_targets

    if request.manual_control_enabled:
        raw_observation, reward, terminated, truncated, raw_info, display = _manual_step(request)
        current_policy_action = None
        cnn_activations = None
    else:
        raw_observation, reward, terminated, truncated, raw_info, display = _policy_step(request)
        current_policy_action = display.policy_action
        cnn_activations = display.cnn_activations

    observation, info = apply_watch_state_feature_zeroing(
        raw_observation,
        raw_info,
        watch_zeroed_features=request.watch_zeroed_state_features,
    )
    current_control_state = request.env.last_requested_control_state
    current_gas_level = request.env.last_gas_level
    final_action_mask_branches = request.env.action_mask_branches()
    live_telemetry = _read_live_telemetry(request.emulator)
    final_auxiliary_predictions = _current_auxiliary_predictions(
        policy_runner=request.policy_runner,
        enabled=request.auxiliary_visualization_enabled,
        observation=observation,
        target_names=request.auxiliary_target_names,
    )
    final_auxiliary_targets = _current_auxiliary_targets(
        telemetry=live_telemetry,
        enabled=request.auxiliary_visualization_enabled,
        target_names=request.auxiliary_target_names,
    )
    boost_lamp_level = _next_boost_lamp_level(
        previous=request.boost_lamp_level,
        control_state=current_control_state,
        boost_active=telemetry_boost_active(live_telemetry),
        action_repeat=request.config.env.action_repeat,
    )
    request.control_rate.tick()
    episode_reward = request.episode_reward + reward
    request.live_series.observe_decision(
        episode=request.episode,
        info=info,
        episode_reward=episode_reward,
        telemetry_data=_telemetry_to_data(live_telemetry),
        action_repeat=request.config.env.action_repeat,
    )
    live_episode_series, last_live_series_publish_time = _live_episode_series(
        enabled=request.live_visualization_enabled,
        live_series=request.live_series,
        last_live_series_publish_time=request.last_live_series_publish_time,
    )
    track_record_book = request.track_record_book.update(
        info,
        live_telemetry,
        episode_done=terminated or truncated,
    )
    _publish_step_snapshots(
        _StepSnapshotPublishRequest(
            config=request.config,
            env=request.env,
            emulator=request.emulator,
            snapshot_queue=request.snapshot_queue,
            display=_StepSnapshotDisplay(
                display_frames=display.display_frames,
                display_controller_masks=display.display_controller_masks,
            ),
            previous=_StepSnapshotFrameState(
                observation=previous_observation,
                info=request.save_notice.apply(previous_info, now=time.perf_counter()),
                episode_reward=previous_episode_reward,
                telemetry=previous_telemetry,
                control_state=previous_control_state,
                gas_level=previous_gas_level,
                action_mask_branches=previous_action_mask_branches,
                policy_action=previous_policy_action,
                auxiliary_predictions=previous_auxiliary_predictions,
                auxiliary_targets=previous_auxiliary_targets,
            ),
            final=_StepSnapshotFrameState(
                observation=observation,
                info=request.save_notice.apply(info, now=time.perf_counter()),
                episode_reward=episode_reward,
                telemetry=live_telemetry,
                control_state=current_control_state,
                gas_level=current_gas_level,
                action_mask_branches=final_action_mask_branches,
                policy_action=current_policy_action,
                auxiliary_predictions=final_auxiliary_predictions,
                auxiliary_targets=final_auxiliary_targets,
            ),
            reset_info=request.reset_info,
            episode=request.episode,
            control_fps=request.control_rate.rate_hz(),
            target_control_fps=request.target_control_fps,
            target_control_seconds=request.target_control_seconds,
            boost_lamp_level=boost_lamp_level,
            policy_runner=request.policy_runner,
            deterministic_policy=request.deterministic_policy,
            manual_control_enabled=request.manual_control_enabled,
            policy_reload_error=request.policy_reload_error,
            cnn_activations=cnn_activations,
            # The full track-sampling config can be hundreds of KB when
            # race-start variants are enabled. Initial/reset snapshots carry it
            # for UI metadata; repeated display-frame snapshots omit it so
            # Watch does not pickle it every frame.
            active_track_sampling=None,
            track_record_book=track_record_book,
            live_episode_series=live_episode_series,
        )
    )
    return LiveStepResult(
        raw_observation=raw_observation,
        raw_info=raw_info,
        observation=observation,
        info=info,
        terminated=terminated,
        truncated=truncated,
        episode_reward=episode_reward,
        control_state=current_control_state,
        gas_level=current_gas_level,
        boost_lamp_level=boost_lamp_level,
        policy_action=current_policy_action,
        cnn_activations=cnn_activations,
        telemetry=live_telemetry,
        auxiliary_predictions=final_auxiliary_predictions,
        auxiliary_targets=final_auxiliary_targets,
        committed_action_mask_branches=final_action_mask_branches,
        track_record_book=track_record_book,
        last_live_series_publish_time=last_live_series_publish_time,
    )


@dataclass(frozen=True, slots=True)
class _LiveStepDisplay:
    display_frames: DisplayFrames
    display_controller_masks: ControllerMaskBatch
    policy_action: ActionValue | None = None
    cnn_activations: CnnActivationSnapshot | None = None


def _manual_step(
    request: LiveStepRequest,
) -> tuple[ObservationValue, float, bool, bool, dict[str, object], _LiveStepDisplay]:
    if request.single_frame_manual:
        observation, reward, terminated, truncated, info = request.env.step_frame(
            request.current_control_state,
            spin_request=request.spin_request,
        )
        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
            _LiveStepDisplay(
                display_frames=(request.env.render(),),
                display_controller_masks=(request.current_control_state.control_mask,),
            ),
        )

    watch_step = request.env.step_control_watch(
        request.current_control_state,
        spin_request=request.spin_request,
    )
    observation, reward, terminated, truncated, info = watch_step.gym_result()
    return (
        observation,
        reward,
        terminated,
        truncated,
        info,
        _LiveStepDisplay(
            display_frames=watch_step.display_frames,
            display_controller_masks=watch_step.display_controller_masks,
        ),
    )


def _policy_step(
    request: LiveStepRequest,
) -> tuple[ObservationValue, float, bool, bool, dict[str, object], _LiveStepDisplay]:
    policy_runner = request.policy_runner
    assert policy_runner is not None
    policy_runner.refresh_if_due(interval_seconds=10.0)
    decision_action_mask = request.env.action_mask_snapshot()
    policy_action_mask = decision_action_mask if policy_runner.supports_action_masks else None
    action = policy_runner.predict(
        request.observation,
        deterministic=request.deterministic_policy,
        action_masks=policy_action_mask.flat if policy_action_mask is not None else None,
        refresh=False,
    )
    if policy_action_mask is not None:
        violations = action_mask_violations(policy_action_mask.branches, action)
        if violations:
            details = ", ".join(violations)
            raise RuntimeError(f"Policy selected masked action values: {details}")
    cnn_activations = request.cnn_sampler.capture(
        enabled=request.cnn_visualization_enabled,
        policy_runner=policy_runner,
        observation=request.observation,
        normalization=request.cnn_normalization,
    )
    watch_step = request.env.step_watch(action)
    observation, reward, terminated, truncated, info = watch_step.gym_result()
    return (
        observation,
        reward,
        terminated,
        truncated,
        info,
        _LiveStepDisplay(
            display_frames=watch_step.display_frames,
            display_controller_masks=watch_step.display_controller_masks,
            policy_action=action,
            cnn_activations=cnn_activations,
        ),
    )


def _live_episode_series(
    *,
    enabled: bool,
    live_series: EpisodeLiveSeriesTracker,
    last_live_series_publish_time: float,
) -> tuple[EpisodeLiveSeriesSnapshot | None, float]:
    if not enabled:
        return None, last_live_series_publish_time
    current_time = time.perf_counter()
    if current_time - last_live_series_publish_time < LIVE_SERIES_PUBLISH_POLICY.interval_seconds:
        return None, last_live_series_publish_time
    return live_series.snapshot(), current_time
