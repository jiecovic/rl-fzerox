# src/rl_fzerox/ui/watch/runtime/worker.py
from __future__ import annotations

import time
from multiprocessing.queues import Queue as ProcessQueue
from typing import TYPE_CHECKING

from fzerox_emulator import ControllerState, Emulator, FZeroXTelemetry
from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs import observations as observation_utils
from rl_fzerox.core.envs.actions import BOOST_MASK, ActionValue
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.envs.telemetry import telemetry_boost_active
from rl_fzerox.core.seed import seed_process
from rl_fzerox.ui.watch.runtime.episode_result import _update_best_finish_position
from rl_fzerox.ui.watch.runtime.process import (
    WatchSnapshot,
    WorkerClosed,
    WorkerError,
    drain_worker_commands,
    publish_worker_message,
)
from rl_fzerox.ui.watch.runtime.telemetry import _telemetry_to_data
from rl_fzerox.ui.watch.runtime.timing import (
    RateMeter,
    _adjust_control_fps,
    _resolve_control_fps,
    _target_seconds,
)
from rl_fzerox.ui.watch.session import (
    _load_policy_runner,
    _persist_reload_error,
    _policy_curriculum_stage,
    _policy_deterministic,
    _policy_label,
    _policy_reload_age_seconds,
    _policy_reload_error,
    _read_live_telemetry,
    _reset_policy_runner,
    _save_baseline_state,
    _sync_policy_curriculum_stage,
)

if TYPE_CHECKING:
    from rl_fzerox.core.training.inference import PolicyRunner

BOOST_ACTIVE_LAMP_LEVEL = 0.55
BOOST_MANUAL_LAMP_LEVEL = 1.0
BOOST_LAMP_FADE_FRAMES = 18


def run_simulation_worker(
    config: WatchAppConfig,
    command_queue: ProcessQueue,
    snapshot_queue: ProcessQueue,
) -> None:
    """Run emulator and policy stepping in a process owned by the watch UI."""

    try:
        _run_simulation_loop(config, command_queue, snapshot_queue)
    except Exception as exc:
        publish_worker_message(snapshot_queue, WorkerError(message=str(exc)))
    finally:
        publish_worker_message(snapshot_queue, WorkerClosed())


def _run_simulation_loop(
    config: WatchAppConfig,
    command_queue: ProcessQueue,
    snapshot_queue: ProcessQueue,
) -> None:
    seed_process(config.seed)
    emulator = Emulator(
        core_path=config.emulator.core_path,
        rom_path=config.emulator.rom_path,
        runtime_dir=config.emulator.runtime_dir,
        baseline_state_path=config.emulator.baseline_state_path,
        renderer=config.emulator.renderer,
    )
    env = FZeroXEnv(
        backend=emulator,
        config=config.env,
        reward_config=config.reward,
        curriculum_config=config.curriculum,
    )
    try:
        policy_runner = _load_policy_runner(
            config.watch.policy_run_dir,
            artifact=config.watch.policy_artifact,
            device=config.watch.device,
        )
        _sync_policy_curriculum_stage(policy_runner, env)
        native_control_fps = env.backend.native_fps / config.env.action_repeat
        target_control_fps = _resolve_control_fps(
            config.watch.control_fps,
            native_control_fps=native_control_fps,
        )
        target_control_seconds = _target_seconds(target_control_fps)
        control_rate = RateMeter(window=60)
        last_logged_reload_error: str | None = None
        episode = 0
        best_finish_position: int | None = None
        paused = False
        manual_control_state = ControllerState()

        while config.watch.episodes is None or episode < config.watch.episodes:
            reset_seed = config.seed if episode == 0 else None
            observation, info = env.reset(seed=reset_seed)
            _reset_policy_runner(policy_runner)
            reset_info = dict(info)
            current_control_state = env.last_requested_control_state
            current_gas_level = env.last_gas_level
            boost_lamp_level = 0.0
            current_policy_action: ActionValue | None = None
            terminated = False
            truncated = False
            episode_reward = 0.0
            next_step_time = time.perf_counter()
            policy_reload_error = _policy_reload_error(policy_runner)

            publish_worker_message(
                snapshot_queue,
                _build_snapshot(
                    config=config,
                    env=env,
                    emulator=emulator,
                    observation=observation,
                    info=info,
                    reset_info=reset_info,
                    episode=episode,
                    episode_reward=episode_reward,
                    control_fps=control_rate.rate_hz(),
                    target_control_fps=target_control_fps,
                    control_state=current_control_state,
                    gas_level=current_gas_level,
                    boost_lamp_level=boost_lamp_level,
                    policy_action=current_policy_action,
                    policy_runner=policy_runner,
                    policy_reload_error=policy_reload_error,
                    best_finish_position=best_finish_position,
                ),
            )

            while not (terminated or truncated):
                commands, paused, manual_control_state = drain_worker_commands(
                    command_queue,
                    paused=paused,
                    control_state=manual_control_state,
                )
                if commands.quit_requested:
                    return
                if commands.reset_requested:
                    break
                if commands.control_fps_delta:
                    target_control_fps = _adjust_control_fps(
                        target_control_fps,
                        commands.control_fps_delta,
                        native_control_fps=native_control_fps,
                    )
                    target_control_seconds = _target_seconds(target_control_fps)
                    control_rate.trim_to_recent()
                    next_step_time = time.perf_counter()
                if commands.save_requests:
                    _save_baseline_state(
                        emulator=emulator,
                        baseline_state_path=config.emulator.baseline_state_path,
                    )
                if policy_runner is None:
                    current_control_state = commands.control_state

                policy_reload_error = _policy_reload_error(policy_runner)
                last_logged_reload_error = _persist_reload_error(
                    reload_error=policy_reload_error,
                    runtime_dir=config.emulator.runtime_dir,
                    last_logged_reload_error=last_logged_reload_error,
                )

                if commands.paused and commands.step_requests <= 0:
                    time.sleep(0.01)
                    continue
                if not commands.paused and target_control_seconds is not None:
                    wait_seconds = next_step_time - time.perf_counter()
                    if wait_seconds > 0.0:
                        time.sleep(min(wait_seconds, 0.005))
                        continue

                single_frame_manual = commands.paused and commands.step_requests > 0
                previous_observation = observation
                previous_info = dict(info)
                previous_episode_reward = episode_reward
                previous_telemetry = _read_live_telemetry(emulator)
                if policy_runner is None:
                    if single_frame_manual:
                        observation, reward, terminated, truncated, info = env.step_frame(
                            current_control_state
                        )
                        display_frames = (env.render(),)
                    else:
                        watch_step = env.step_control_watch(current_control_state)
                        observation, reward, terminated, truncated, info = watch_step.gym_result()
                        display_frames = watch_step.display_frames
                    current_policy_action = None
                    current_control_state = env.last_requested_control_state
                    current_gas_level = env.last_gas_level
                else:
                    _sync_policy_curriculum_stage(policy_runner, env)
                    action = policy_runner.predict(
                        observation,
                        deterministic=config.watch.deterministic_policy,
                        action_masks=env.action_masks(),
                    )
                    current_policy_action = action
                    watch_step = env.step_watch(action)
                    observation, reward, terminated, truncated, info = watch_step.gym_result()
                    display_frames = watch_step.display_frames
                    current_control_state = env.last_requested_control_state
                    current_gas_level = env.last_gas_level
                live_telemetry = _read_live_telemetry(emulator)
                boost_lamp_level = _next_boost_lamp_level(
                    previous=boost_lamp_level,
                    control_state=current_control_state,
                    boost_active=telemetry_boost_active(live_telemetry),
                    action_repeat=config.env.action_repeat,
                )
                control_rate.tick()
                episode_reward += reward
                best_finish_position = _update_best_finish_position(
                    best_finish_position,
                    info,
                    None,
                )
                _publish_step_snapshots(
                    config=config,
                    env=env,
                    emulator=emulator,
                    snapshot_queue=snapshot_queue,
                    display_frames=display_frames,
                    previous_observation=previous_observation,
                    previous_info=previous_info,
                    previous_episode_reward=previous_episode_reward,
                    previous_telemetry=previous_telemetry,
                    final_observation=observation,
                    final_info=info,
                    final_episode_reward=episode_reward,
                    final_telemetry=live_telemetry,
                    reset_info=reset_info,
                    episode=episode,
                    control_fps=control_rate.rate_hz(),
                    target_control_fps=target_control_fps,
                    target_control_seconds=target_control_seconds,
                    control_state=current_control_state,
                    gas_level=current_gas_level,
                    boost_lamp_level=boost_lamp_level,
                    policy_action=current_policy_action,
                    policy_runner=policy_runner,
                    policy_reload_error=policy_reload_error,
                    best_finish_position=best_finish_position,
                )
                if target_control_seconds is not None:
                    now = time.perf_counter()
                    next_step_time = max(next_step_time + target_control_seconds, now)
            episode += 1
    finally:
        env.close()


def _publish_step_snapshots(
    *,
    config: WatchAppConfig,
    env: FZeroXEnv,
    emulator: Emulator,
    snapshot_queue: ProcessQueue,
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
    policy_action: ActionValue | None,
    policy_runner: PolicyRunner | None,
    policy_reload_error: str | None,
    best_finish_position: int | None,
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
                telemetry=final_telemetry if is_final_frame else previous_telemetry,
                policy_action=policy_action,
                policy_runner=policy_runner,
                policy_reload_error=policy_reload_error,
                best_finish_position=best_finish_position,
            ),
        )
        if frame_interval_seconds is not None and not is_final_frame:
            time.sleep(frame_interval_seconds)


def _build_snapshot(
    *,
    config: WatchAppConfig,
    env: FZeroXEnv,
    emulator: Emulator,
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
    policy_action: ActionValue | None,
    policy_runner: PolicyRunner | None,
    policy_reload_error: str | None,
    best_finish_position: int | None,
    telemetry: FZeroXTelemetry | None = None,
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
        policy_action=policy_action,
        policy_label=_policy_label(policy_runner),
        policy_curriculum_stage=_policy_curriculum_stage(policy_runner),
        policy_deterministic=_policy_deterministic(
            policy_runner,
            config.watch.deterministic_policy,
        ),
        policy_reload_age_seconds=_policy_reload_age_seconds(policy_runner),
        policy_reload_error=policy_reload_error,
        best_finish_position=best_finish_position,
        continuous_air_brake_disabled=_continuous_air_brake_disabled(config, telemetry),
        telemetry_data=_telemetry_to_data(telemetry),
    )


def _continuous_air_brake_disabled(
    config: WatchAppConfig,
    telemetry: FZeroXTelemetry | None,
) -> bool:
    if config.env.action.continuous_air_brake_mode != "disable_on_ground":
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
        return BOOST_MANUAL_LAMP_LEVEL

    target = BOOST_ACTIVE_LAMP_LEVEL if boost_active else 0.0
    if previous <= target:
        return target

    decay = max(1, action_repeat) / BOOST_LAMP_FADE_FRAMES
    return max(target, previous - decay)
