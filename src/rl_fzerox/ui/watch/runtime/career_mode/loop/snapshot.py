# src/rl_fzerox/ui/watch/runtime/career_mode/loop/snapshot.py
from __future__ import annotations

from multiprocessing.queues import Queue as ProcessQueue

from fzerox_emulator import RaceControlState
from rl_fzerox.core.career_mode.controller import CareerModeController
from rl_fzerox.core.career_mode.policy import CareerModePolicyControl
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.policy.auxiliary_state import AuxiliaryStateTargetName
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.records import TrackRecordBook
from rl_fzerox.ui.watch.runtime.career_mode.loop.state import TimedRecordingNotice
from rl_fzerox.ui.watch.runtime.career_mode.menu import reset_race_progress_info
from rl_fzerox.ui.watch.runtime.career_mode.session import CareerModeRuntimeSession
from rl_fzerox.ui.watch.runtime.career_mode.timing import (
    measured_game_fps,
    snapshot_action_repeat,
    snapshot_target_control_fps,
    target_game_fps,
    with_measured_game_fps,
)
from rl_fzerox.ui.watch.runtime.ipc import publish_worker_message
from rl_fzerox.ui.watch.runtime.policy.cnn import CnnActivationSnapshot
from rl_fzerox.ui.watch.runtime.policy.runner import _policy_reload_error
from rl_fzerox.ui.watch.runtime.snapshots.build import _build_snapshot
from rl_fzerox.ui.watch.runtime.timing import RateMeter


def publish_career_loop_snapshot(
    *,
    config: WatchAppConfig,
    session: CareerModeRuntimeSession,
    controller: CareerModeController,
    snapshot_queue: ProcessQueue,
    control_rate: RateMeter,
    native_control_fps: float,
    target_control_fps: float | None,
    policy_visible: bool,
    active_policy_control: CareerModePolicyControl | None,
    observation: ObservationValue | None,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode: int,
    episode_reward: float,
    current_control_state: RaceControlState,
    current_gas_level: float,
    boost_lamp_level: float,
    current_policy_action: ActionValue | None,
    current_auxiliary_predictions: dict[str, object] | None,
    current_auxiliary_targets: dict[str, object] | None,
    auxiliary_visualization_enabled: bool,
    auxiliary_target_names: tuple[AuxiliaryStateTargetName, ...],
    deterministic_policy: bool,
    manual_control_enabled: bool,
    cnn_activations: CnnActivationSnapshot | None,
    track_record_book: TrackRecordBook,
    recording_notice: TimedRecordingNotice,
    now: float,
) -> None:
    policy_active = policy_visible and observation is not None
    snapshot_target_fps = snapshot_target_control_fps(
        config=config,
        session=session,
        native_control_fps=native_control_fps,
        target_control_fps=target_control_fps,
        policy_active=policy_active,
    )
    snapshot_info = controller.viewer_info(
        info=info if policy_active else reset_race_progress_info(info),
        active_policy_control=(active_policy_control if policy_active else None),
    )
    snapshot_config = session.snapshot_config(config) if policy_active else config
    snapshot_repeat = snapshot_action_repeat(
        snapshot_config,
        policy_active=policy_active,
    )
    snapshot_info = with_measured_game_fps(
        snapshot_info,
        game_fps=measured_game_fps(
            control_fps=control_rate.rate_hz(),
            action_repeat=snapshot_repeat,
        ),
        game_fps_target=target_game_fps(
            target_control_fps=snapshot_target_fps,
            action_repeat=snapshot_repeat,
        ),
    )
    snapshot_info = recording_notice.apply(snapshot_info, now=now)
    runner = (
        active_policy_control.runner
        if policy_active and active_policy_control is not None
        else None
    )
    publish_worker_message(
        snapshot_queue,
        _build_snapshot(
            config=snapshot_config,
            env=session,
            emulator=session.emulator,
            observation=observation if policy_active else None,
            info=snapshot_info,
            reset_info=reset_info,
            episode=episode,
            episode_reward=episode_reward if policy_active else 0.0,
            control_fps=control_rate.rate_hz(),
            target_control_fps=snapshot_target_fps,
            action_repeat=snapshot_repeat,
            control_state=current_control_state,
            gas_level=current_gas_level,
            boost_lamp_level=boost_lamp_level,
            action_mask_branches=session.action_mask_branches(),
            policy_action=current_policy_action if policy_active else None,
            policy_runner=runner,
            policy_auxiliary_state_predictions=(
                current_auxiliary_predictions if policy_active else None
            ),
            policy_auxiliary_state_targets=(current_auxiliary_targets if policy_active else None),
            include_auxiliary_state=policy_active and auxiliary_visualization_enabled,
            auxiliary_target_names=auxiliary_target_names,
            deterministic_policy=deterministic_policy,
            manual_control_enabled=manual_control_enabled if policy_active else False,
            policy_reload_error=_policy_reload_error(runner),
            cnn_activations=cnn_activations if policy_active else None,
            track_record_book=track_record_book,
        ),
    )
