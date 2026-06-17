# src/rl_fzerox/ui/watch/runtime/career_mode/loop/state.py
from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing.queues import Queue as ProcessQueue

from fzerox_emulator import FZeroXTelemetry, RaceControlState
from rl_fzerox.core.career_mode.controller import CareerModeController
from rl_fzerox.core.career_mode.navigation import RawMenuStep
from rl_fzerox.core.career_mode.runner.policy import CareerModePolicyControl
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.policy.auxiliary_state import AuxiliaryStateTargetName
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.live_series import EpisodeLiveSeriesTracker
from rl_fzerox.ui.watch.records import TrackRecordBook
from rl_fzerox.ui.watch.runtime.career_mode.menu import menu_viewer_info
from rl_fzerox.ui.watch.runtime.career_mode.recording import FrameRecorder
from rl_fzerox.ui.watch.runtime.career_mode.session import CareerModeRuntimeSession
from rl_fzerox.ui.watch.runtime.career_mode.timing import (
    native_frame_seconds as _native_frame_seconds,
)
from rl_fzerox.ui.watch.runtime.ipc import publish_worker_message
from rl_fzerox.ui.watch.runtime.policy.cnn import (
    DEFAULT_CNN_ACTIVATION_NORMALIZATION,
    CnnActivationNormalizationMode,
    CnnActivationSampler,
    CnnActivationSnapshot,
)
from rl_fzerox.ui.watch.runtime.snapshots.build import _build_snapshot
from rl_fzerox.ui.watch.runtime.telemetry import _read_live_telemetry
from rl_fzerox.ui.watch.runtime.timing import RateMeter

_WATCH_RECORDING_NOTICE_SECONDS = 5.0


@dataclass(slots=True)
class TimedRecordingNotice:
    """Short-lived recorder status text included in watch snapshots."""

    message: str | None = None
    expires_at: float = 0.0

    def show(self, message: str, *, now: float) -> None:
        self.message = message
        self.expires_at = now + _WATCH_RECORDING_NOTICE_SECONDS

    def apply(self, info: dict[str, object], *, now: float) -> dict[str, object]:
        if self.message is None or now >= self.expires_at:
            return info
        with_notice = dict(info)
        with_notice["watch_save_notice"] = self.message
        return with_notice


@dataclass(slots=True)
class CareerModeLoopState:
    """Mutable state unpacked by the live Career Mode worker loop."""

    control_rate: RateMeter
    native_control_fps: float
    target_control_fps: float | None
    target_control_seconds: float | None
    native_frame_seconds: float | None
    next_step_time: float
    paused: bool
    deterministic_policy: bool
    manual_control_enabled: bool
    manual_control_state: RaceControlState
    current_control_state: RaceControlState
    current_gas_level: float
    boost_lamp_level: float
    episode: int
    episode_reward: float
    cnn_visualization_enabled: bool
    auxiliary_visualization_enabled: bool
    live_visualization_enabled: bool
    live_series: EpisodeLiveSeriesTracker
    track_record_book: TrackRecordBook
    last_live_series_publish_time: float
    cnn_normalization: CnnActivationNormalizationMode
    cnn_sampler: CnnActivationSampler
    watch_zeroed_state_features: frozenset[str]
    auxiliary_target_names: tuple[AuxiliaryStateTargetName, ...]
    active_policy_control: CareerModePolicyControl | None
    active_policy_started: bool
    current_policy_action: ActionValue | None
    raw_observation: ObservationValue | None
    observation: ObservationValue | None
    raw_info: dict[str, object]
    info: dict[str, object]
    reset_info: dict[str, object]
    current_telemetry: FZeroXTelemetry | None
    current_auxiliary_predictions: dict[str, object] | None
    current_auxiliary_targets: dict[str, object] | None
    cnn_activations: CnnActivationSnapshot | None
    last_menu_step: RawMenuStep | None


def initial_career_mode_loop_state(
    *,
    config: WatchAppConfig,
    session: CareerModeRuntimeSession,
    controller: CareerModeController,
) -> CareerModeLoopState:
    target_control_seconds = session.target_control_seconds
    raw_info = menu_viewer_info(session)
    info = controller.viewer_info(
        info=dict(raw_info),
        active_policy_control=None,
    )
    return CareerModeLoopState(
        control_rate=RateMeter(window=60),
        native_control_fps=session.native_control_fps,
        target_control_fps=session.target_control_fps,
        target_control_seconds=target_control_seconds,
        native_frame_seconds=_native_frame_seconds(target_control_seconds),
        next_step_time=time.perf_counter(),
        paused=False,
        deterministic_policy=bool(config.watch.deterministic_policy),
        manual_control_enabled=False,
        manual_control_state=RaceControlState(),
        current_control_state=RaceControlState(),
        current_gas_level=0.0,
        boost_lamp_level=0.0,
        episode=0,
        episode_reward=0.0,
        cnn_visualization_enabled=False,
        auxiliary_visualization_enabled=False,
        live_visualization_enabled=False,
        live_series=EpisodeLiveSeriesTracker(),
        track_record_book=TrackRecordBook(),
        last_live_series_publish_time=0.0,
        cnn_normalization=DEFAULT_CNN_ACTIVATION_NORMALIZATION,
        cnn_sampler=CnnActivationSampler(refresh_interval_steps=1),
        watch_zeroed_state_features=session.watch_zeroed_state_features,
        auxiliary_target_names=session.auxiliary_target_names,
        active_policy_control=None,
        active_policy_started=False,
        current_policy_action=None,
        raw_observation=None,
        observation=None,
        raw_info=raw_info,
        info=info,
        reset_info=dict(info),
        current_telemetry=_read_live_telemetry(session.emulator),
        current_auxiliary_predictions=None,
        current_auxiliary_targets=None,
        cnn_activations=None,
        last_menu_step=None,
    )


def publish_initial_career_snapshot(
    *,
    config: WatchAppConfig,
    session: CareerModeRuntimeSession,
    snapshot_queue: ProcessQueue,
    state: CareerModeLoopState,
    frame_recorder: FrameRecorder | None = None,
) -> None:
    raw_frame = session.render()
    if frame_recorder is not None:
        frame_recorder.record_frame(raw_frame, info=state.info)
    publish_worker_message(
        snapshot_queue,
        _build_snapshot(
            config=config,
            env=session,
            emulator=session.emulator,
            raw_frame=raw_frame,
            observation=None,
            info=state.info,
            reset_info=state.reset_info,
            episode=state.episode,
            episode_reward=state.episode_reward,
            control_fps=state.control_rate.rate_hz(),
            target_control_fps=state.target_control_fps,
            action_repeat=1,
            control_state=state.current_control_state,
            gas_level=state.current_gas_level,
            boost_lamp_level=state.boost_lamp_level,
            action_mask_branches=session.action_mask_branches(),
            policy_action=None,
            policy_runner=None,
            policy_auxiliary_state_predictions=None,
            policy_auxiliary_state_targets=None,
            include_auxiliary_state=False,
            auxiliary_target_names=state.auxiliary_target_names,
            deterministic_policy=state.deterministic_policy,
            manual_control_enabled=False,
            policy_reload_error=None,
            cnn_activations=None,
            track_record_book=state.track_record_book,
        ),
    )
