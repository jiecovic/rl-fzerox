# src/rl_fzerox/ui/watch/runtime/snapshots.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from fzerox_emulator import ControllerState, FZeroXTelemetry
from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.core.envs import observations as observation_utils
from rl_fzerox.core.envs.actions import BOOST_MASK, ActionValue
from rl_fzerox.core.envs.engine.masks import ActionMaskBranches
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.ui.watch.runtime.episode import _update_best_finish_position
from rl_fzerox.ui.watch.runtime.ipc import (
    WatchSnapshot,
    WorkerMessageQueue,
    publish_worker_message,
)
from rl_fzerox.ui.watch.runtime.policy import (
    _policy_curriculum_stage,
    _policy_deterministic,
    _policy_label,
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
    display_frames: tuple[RgbFrame, ...],
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
    control_state: ControllerState,
    gas_level: float,
    boost_lamp_level: float,
    action_mask_branches: ActionMaskBranches,
    policy_action: ActionValue | None,
    policy_runner: PolicyRunner | None,
    deterministic_policy: bool,
    policy_reload_error: str | None,
    best_finish_position: int | None,
    best_finish_times: dict[str, int],
    latest_finish_times: dict[str, int],
) -> None:
    frames = display_frames or (env.render(),)
    frame_interval_seconds = (
        None if target_control_seconds is None else target_control_seconds / len(frames)
    )
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
                control_state=control_state,
                gas_level=gas_level,
                boost_lamp_level=boost_lamp_level,
                action_mask_branches=action_mask_branches,
                telemetry=final_telemetry if is_final_frame else previous_telemetry,
                policy_action=policy_action,
                policy_runner=policy_runner,
                deterministic_policy=deterministic_policy,
                policy_reload_error=policy_reload_error,
                best_finish_position=best_finish_position,
                best_finish_times=best_finish_times,
                latest_finish_times=latest_finish_times,
                action_hold_frame=index + 1,
                action_hold_frames=len(frames),
                policy_decision_frame=is_final_frame,
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
    observation: ObservationValue,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode: int,
    episode_reward: float,
    control_fps: float,
    target_control_fps: float | None,
    control_state: ControllerState,
    gas_level: float,
    boost_lamp_level: float,
    action_mask_branches: ActionMaskBranches | None = None,
    policy_action: ActionValue | None,
    policy_runner: PolicyRunner | None,
    deterministic_policy: bool,
    policy_reload_error: str | None,
    best_finish_position: int | None,
    best_finish_times: dict[str, int],
    latest_finish_times: dict[str, int],
    telemetry: FZeroXTelemetry | None = None,
    action_hold_frame: int = 1,
    action_hold_frames: int = 1,
    policy_decision_frame: bool = True,
) -> WatchSnapshot:
    if telemetry is None:
        telemetry = _read_live_telemetry(emulator)
    best_finish_position = _update_best_finish_position(best_finish_position, info, telemetry)
    return WatchSnapshot(
        raw_frame=env.render() if raw_frame is None else raw_frame,
        observation_image=observation_utils.observation_image(observation),
        observation_state=observation_utils.observation_state(observation),
        info=dict(info),
        reset_info=dict(reset_info),
        episode=episode,
        episode_reward=episode_reward,
        control_fps=control_fps,
        target_control_fps=target_control_fps,
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
        policy_deterministic=_policy_deterministic(
            policy_runner,
            deterministic_policy,
        ),
        policy_reload_age_seconds=_policy_reload_age_seconds(policy_runner),
        policy_reload_error=policy_reload_error,
        best_finish_position=best_finish_position,
        best_finish_times=dict(best_finish_times),
        latest_finish_times=dict(latest_finish_times),
        continuous_air_brake_disabled=_continuous_air_brake_disabled(config, telemetry),
        telemetry_data=_telemetry_to_data(telemetry),
        action_hold_frame=max(1, int(action_hold_frame)),
        action_hold_frames=max(1, int(action_hold_frames)),
        policy_decision_frame=bool(policy_decision_frame),
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
    control_state: ControllerState,
    boost_active: bool,
    action_repeat: int,
) -> float:
    if control_state.joypad_mask & BOOST_MASK:
        return BOOST_LAMP_CONFIG.manual_level

    target = BOOST_LAMP_CONFIG.active_level if boost_active else 0.0
    if previous <= target:
        return target

    decay = max(1, action_repeat) / BOOST_LAMP_CONFIG.fade_frames
    return max(target, previous - decay)
